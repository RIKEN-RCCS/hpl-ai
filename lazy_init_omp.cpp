#include "lazy_init.hpp"

#include <assert.h>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>

extern "C" void print_fp16(fp16 *p){
	double f = *p;
	printf("%f\n", f);
}

extern "C" void conv_scale_copy(fp16 * __restrict dst, const float * __restrict src, int n, const float scale){
	if(1.0f == scale){
		// for(int i=0; i<n; i++) dst[i] = (fp16)src[i];
		int len = svcnth();
		svbool_t t32 = svptrue_b32();

		if(288==n & 32==len){
			for(int i=0; i<288; i+=32){
				svfloat32x2_t f2 = svld2_f32(t32, src+i);
				svfloat16_t evn = svcvt_f16_f32_x(t32, svget2_f32(f2,0));
				svfloat16_t odd = svcvt_f16_f32_x(t32, svget2_f32(f2,1));
				svfloat16_t vec = svtrn1_f16(evn, odd);
				svst1_f16(svptrue_b16(), dst+i, vec);
			}
		}else{
			for(int i=0; i<n; i+=len){
				svfloat32x2_t f2 = svld2_f32(t32, src+i);
				svfloat16_t evn = svcvt_f16_f32_x(t32, svget2_f32(f2,0));
				svfloat16_t odd = svcvt_f16_f32_x(t32, svget2_f32(f2,1));
				svfloat16_t vec = svtrn1_f16(evn, odd);
				svbool_t p = svwhilelt_b16_s32(i, n);
				svst1_f16(p, dst+i, vec);
			}
		}
	}else{
		// for(int i=0; i<n; i++) dst[i] = (fp16)(scale * src[i]);
		int len = svcnth();
		svbool_t t32 = svptrue_b32();
		svfloat32_t vscale = svdup_f32(scale);
		for(int i=0; i<n; i+=len){
			svfloat32x2_t f2 = svld2_f32(t32, src+i);

			f2 = svset2_f32(f2,0,svmul_f32_x(t32, svget2_f32(f2,0), vscale));
			f2 = svset2_f32(f2,1,svmul_f32_x(t32, svget2_f32(f2,1), vscale));

			svfloat16_t evn = svcvt_f16_f32_x(t32, svget2_f32(f2,0));
			svfloat16_t odd = svcvt_f16_f32_x(t32, svget2_f32(f2,1));

			svfloat16_t vec = svtrn1_f16(evn, odd);
			svbool_t p = svwhilelt_b16_s32(i, n);
			svst1_f16(p, dst+i, vec);
		}
	}
}


inline svuint64_t update(RandCoeff ac, svuint64_t stat){
	svbool_t t64 = svptrue_b64();
	svuint64_t va = svdup_u64(ac.a);
	svuint64_t vc = svdup_u64(ac.c);
	return svmad_u64_x(t64, va, stat, vc);
}

inline svuint64_t update(svuint64x2_t ac, RandStat stat){
	svbool_t t64 = svptrue_b64();
	svuint64_t vstat = svdup_u64(stat.x);
	return svmad_u64_x(t64, svget2_u64(ac,0), vstat, svget2_u64(ac,1));
	// return svmad_u64_x(t64, vstat, ac.v0, ac.v1);
}

inline svuint64_t update(svuint64_t ac0,svuint64_t ac1, RandStat stat){
	svbool_t t64 = svptrue_b64();
	svuint64_t vstat = svdup_u64(stat.x);
	return svmad_u64_x(t64, ac0, vstat, ac1); 
}

static inline void copy_xx(float * __restrict__ dst, float const* __restrict__ src, int b)
{
	for(int i=0; i<b; ++i) dst[i] = src[i];
}
static inline void copy_xx(fp16* __restrict__ dst, fp16 const* __restrict__ src, int b)
{
	for(int i=0; i<b; ++i) dst[i] = src[i];
}

extern "C"
void lazy_init_f32_f16_in_omp(
		Matgen<float>const& mg, Panels<float>& p, float scale, 
		int rowstart, int rowend, int colstart, int colend,
		int js, int je)
{
	svbool_t t64 = svptrue_b64();
	svbool_t t32 = svptrue_b32();

	const uint64_t *pow_ptr = &mg.powers[0].a;
	svuint64x4_t coef_x4 = svld4_u64(t64, pow_ptr);
	svuint64_t a_evn = svget4_u64(coef_x4,0);
	svuint64_t c_evn = svget4_u64(coef_x4,1);
	svuint64_t a_odd = svget4_u64(coef_x4,2);
	svuint64_t c_odd = svget4_u64(coef_x4,3);

	typedef DDAdaptor<float, fp16, true> DDA;
	size_t const lda = p.lda;
	size_t const ldl = DDA::get_ldl(p);

	int const b = p.b;
	assert(0 == b%16);
	int const i1 = p.i1;
	int const j1 = p.j1;
	int const istride = p.istride;
	int const jstride = p.jstride;
	int const istart = i1 + rowstart*istride;
	int const jstart = j1 + colstart*jstride;

	RandCoeff incl1 = mg.incl1;
	RandCoeff jumpi = mg.jumpi;
	RandCoeff jumpj = mg.jumpj; // jump (Q-1)*b columns
	RandCoeff jumpn = mg.jumpn; // jump (P-1)*b rows

	RandCoeff jumpjp = jumpj * mg.jumpn.pow(b); // jump Q*b columnns
	RandCoeff jumpip = jumpi * mg.incl1.pow(b); // jump P*b rows
	RandCoeff incl16 = incl1.pow(16);
	RandStat stat_00 = {0};

	int joff_thread = -1;

	for(int pj=colstart; pj<colend; ++pj){
		RandStat stat_00_save = stat_00;
// #pragma omp for nowait
		// for(int j=0; j<b; ++j){
		for(int j=js; j<je; ++j){
			if(joff_thread < 0){
				joff_thread = j;
				stat_00_save = mg.jump(b*istart, b*jstart + joff_thread) 
					* RandStat::initialize(mg.seed);
				stat_00 = stat_00_save;
			}
			RandStat stat_i = stat_00;
			for(int pi=rowstart; pi<rowend; ++pi){
				float* __restrict to = p(pi, pj);
				fp16 * __restrict from = DDA::get_deck(p, pi, pj);

				// svuint64_t stat_evn = update({a_evn, c_evn}, stat_i);
				// svuint64_t stat_odd = update({a_odd, c_odd}, stat_i);

				svuint64_t stat_evn = update(a_evn, c_evn, stat_i);
				svuint64_t stat_odd = update(a_odd, c_odd, stat_i);
				for(int i=0; i<b; i+=16){
					// float aij = static_cast<float>(stat_i);
					// to[j*lda + i] = scale * static_cast<float>(from[j*ldl + i]) + aij;
					// stat_i = incl1 * stat_i;
					svfloat32_t fevn = svcvt_f32_s64_x(t32, svreinterpret_s64_u64(stat_evn));
					svfloat32_t fodd = svcvt_f32_s64_x(t32, svreinterpret_s64_u64(stat_odd));
					svfloat32_t fvec = svtrn1_f32(fevn, fodd);
					svfloat32_t vfac = svdup_f32(0x1.fffffffffffffP-65);
					fvec = svmul_f32_x(t32, fvec, vfac);

					const fp16 *hptr = &from[j*ldl + i];
					svuint32_t  hvec1 = svld1sh_u32(t32,  (int16_t *)hptr);
					svfloat16_t hvec2 = svreinterpret_f16_u32(hvec1);
					svfloat32_t hvec3 = svcvt_f32_f16_x(t32, hvec2);

					svfloat32_t vscale = svdup_f32(scale);
					fvec = svmla_f32_x(t32, fvec, hvec3, vscale);

					float *fptr = &to[j*lda + i];
					svst1_f32(t32, fptr, fvec);

					stat_evn = update(incl16, stat_evn);
					stat_odd = update(incl16, stat_odd);
				}
				stat_i = jumpip * stat_i;
			}
			stat_00 = jumpn * stat_00;
		}
		stat_00 = jumpjp * stat_00_save;
	}
}

extern "C"
void lazy_init_f32_f16_in_omp_bb(
		Matgen<float>const& mg, Panels<float>& p, float scale, 
		int rowstart, int rowend, int colstart, int colend,
		int js, int je, float* buf)
{
	svbool_t t64 = svptrue_b64();
	svbool_t t32 = svptrue_b32();

	const uint64_t *pow_ptr = &mg.powers[0].a;
	svuint64x4_t coef_x4 = svld4_u64(t64, pow_ptr);
	svuint64_t a_evn = svget4_u64(coef_x4,0);
	svuint64_t c_evn = svget4_u64(coef_x4,1);
	svuint64_t a_odd = svget4_u64(coef_x4,2);
	svuint64_t c_odd = svget4_u64(coef_x4,3);


	typedef DDAdaptor<float, fp16, true> DDA;
	size_t const lda = p.lda;
	size_t const ldl = DDA::get_ldl(p);

	int const b = p.b;
	assert(0 == b%16);
	int const i1 = p.i1;
	int const j1 = p.j1;
	int const istride = p.istride;
	int const jstride = p.jstride;
	int const istart = i1 + rowstart*istride;
	int const jstart = j1 + colstart*jstride;

	RandCoeff incl1 = mg.incl1;
	RandCoeff jumpi = mg.jumpi;
	RandCoeff jumpj = mg.jumpj; // jump (Q-1)*b columns
	RandCoeff jumpn = mg.jumpn; // jump (P-1)*b rows

	RandCoeff jumpjp = jumpj * mg.jumpn.pow(b); // jump Q*b columnns
	RandCoeff jumpip = jumpi * mg.incl1.pow(b); // jump P*b rows
	RandCoeff incl16 = incl1.pow(16);
	RandStat stat_00 = {0};

	int joff_thread = -1;
	fp16* lbuf = reinterpret_cast<fp16*>(buf);
	float lbufscale = BB_NCYCLE;
	svfloat32_t svlbs = svdup_f32(lbufscale);
	svfloat32_t vscale = svdup_f32(scale);

	for(int pj=colstart; pj<colend; ++pj){
		RandStat stat_00_save = stat_00;
// #pragma omp for nowait
		// for(int j=0; j<b; ++j){
		for(int j=js; j<je; ++j){
			if(joff_thread < 0){
				joff_thread = j;
				stat_00_save = mg.jump(b*istart, b*jstart + joff_thread) 
					* RandStat::initialize(mg.seed);
				stat_00 = stat_00_save;
			}
			RandStat stat_i = stat_00;
			for(int pi=rowstart; pi<rowend; ++pi){
				float* __restrict to = p(pi, pj);
				fp16 * __restrict from = DDA::get_deck(p, pi, pj);

				// svuint64_t stat_evn = update({a_evn, c_evn}, stat_i);
				// svuint64_t stat_odd = update({a_odd, c_odd}, stat_i);
				svuint64_t stat_evn = update(a_evn, c_evn, stat_i);
				svuint64_t stat_odd = update(a_odd, c_odd, stat_i);

				copy_xx(buf, to+j*lda, b);

				for(int i=0; i<b; i+=16){
					// float aij = static_cast<float>(stat_i);
					// to[j*lda + i] = scale * static_cast<float>(from[j*ldl + i]) + aij;
					// stat_i = incl1 * stat_i;
					svfloat32_t fevn = svcvt_f32_s64_x(t32, svreinterpret_s64_u64(stat_evn));
					svfloat32_t fodd = svcvt_f32_s64_x(t32, svreinterpret_s64_u64(stat_odd));
					svfloat32_t fvec = svtrn1_f32(fevn, fodd);
					svfloat32_t vfac = svdup_f32(0x1.fffffffffffffP-65);
					fvec = svmul_f32_x(t32, fvec, vfac);

					const fp16 *hptr = &from[j*ldl + i];
					svuint32_t  hvec1 = svld1sh_u32(t32,  (int16_t *)hptr);
					svfloat16_t hvec2 = svreinterpret_f16_u32(hvec1);
					svfloat32_t hvec3 = svcvt_f32_f16_x(t32, hvec2);

					svuint32_t  hvec4 = svld1sh_u32(t32,  (int16_t *)(lbuf+i));
					svfloat16_t hvec5 = svreinterpret_f16_u32(hvec4);
					svfloat32_t hvec6 = svcvt_f32_f16_x(t32, hvec5);

					hvec3 = svmla_f32_x(t32, hvec3, hvec6, svlbs);

					fvec = svmla_f32_x(t32, fvec, hvec3, vscale);

					float *fptr = &to[j*lda + i];
					svst1_f32(t32, fptr, fvec);

					stat_evn = update(incl16, stat_evn);
					stat_odd = update(incl16, stat_odd);
				}
				stat_i = jumpip * stat_i;
				if(pi!=p.nprow-1) copy_xx(from+j*ldl, lbuf+b, b);
			}
			stat_00 = jumpn * stat_00;
		}
		stat_00 = jumpjp * stat_00_save;
	}
}

#define LDCVT(T,A) svcvt_f32_f16_x(T, svreinterpret_f16_u32(svld1sh_u32(T,(int16_t*)(A))))
static inline void lishhm_up(int istart, int b, float beta, float ab, float scale, fp16 const* from, float* to)
{
	#if 0
	#pragma clang loop vectorize(enable)
	for(int i=0; i<b; ++i){
		float xij = static_cast<float>(from[i]);
		float aij = beta + ab * (istart + i);
		to[i] = scale * xij + aij;
	}
	#else
	svfloat32_t svab = svdup_f32(ab);
	svfloat32_t svbeta = svdup_f32(beta);
	svfloat32_t svscale = svdup_f32(scale);
	for(int i=0; i<b; i+=svcntw()){
		svbool_t t = svwhilelt_b32(i, b);
		svfloat32_t t0 = LDCVT(t, from+i);
		svint32_t idx = svindex_s32(istart+i, 1);
		svfloat32_t fidx = svcvt_f32_s32_x(t, idx);
		svfloat32_t aij = svmad_f32_x(t, fidx, svab, svbeta);
		svfloat32_t t1 = svmla_f32_x(t, aij, svscale, t0);
		svst1_vnum_f32(t, to+i, 0, t1);
	}
	#endif
}
static inline void lishhm_low(int b, float aij_j, float scale, fp16 const* from, float* to)
{
	#pragma clang loop vectorize(enable)
	for(int i=0; i<b; ++i){
		float xij = static_cast<float>(from[i]);
		to[i] = scale * xij + aij_j;
	}
}
static inline void lishhmbb_up(int istart, int b, float beta, float ab, float scale, fp16 const* from, fp16 const* lbuf, float* to)
{
	#if 0
	float const lbufscale = BB_NCYCLE;
	#pragma clang loop vectorize(enable)
	for(int i=0; i<b; ++i){
		float xij = static_cast<float>(from[i]) + lbufscale*static_cast<float>(lbuf[i]);
		float aij = beta + ab * (istart + i);
		to[i] = scale * xij + aij;
	}
	#else
	svfloat32_t svab = svdup_f32(ab);
	svfloat32_t svbeta = svdup_f32(beta);
	svfloat32_t svscale = svdup_f32(scale);
	svfloat32_t svls = svdup_f32(BB_NCYCLE);
	for(int i=0; i<b; i+=svcntw()){
		svbool_t t = svwhilelt_b32(i, b);
		svfloat32_t t0 = LDCVT(t, from+i);
		svfloat32_t t1 = LDCVT(t, lbuf+i);
		svfloat32_t t2 = svmla_f32_x(t, t0, t1, svls);
		svint32_t idx = svindex_s32(istart+i, 1);
		svfloat32_t fidx = svcvt_f32_s32_x(t, idx);
		svfloat32_t aij = svmad_f32_x(t, fidx, svab, svbeta);
		svfloat32_t t3 = svmla_f32_x(t, aij, svscale, t2);
		svst1_vnum_f32(t, to+i, 0, t3);
	}
	#endif
}
static inline void lishhmbb_low(int b, float aij_j, float scale, fp16 const* from, fp16 const* lbuf, float* to)
{
	#if 0
	float const lbufscale = BB_NCYCLE;
	#pragma clang loop vectorize(enable)
	for(int i=0; i<b; ++i){
		float xij = static_cast<float>(from[i]) + lbufscale*static_cast<float>(lbuf[i]);
		to[i] = scale * xij + aij_j;
	}
	#else
	svfloat32_t svaij_j = svdup_f32(aij_j);
	svfloat32_t svscale = svdup_f32(scale);
	svfloat32_t svls = svdup_f32(BB_NCYCLE);
	for(int i=0; i<b; i+=svcntw()){
		svbool_t t = svwhilelt_b32(i, b);
		svfloat32_t t0 = LDCVT(t, from+i);
		svfloat32_t t1 = LDCVT(t, lbuf+i);
		svfloat32_t t2 = svmla_f32_x(t, t0, t1, svls);
		svfloat32_t t3 = svmad_f32_x(t, t2, svscale, svaij_j);
		svst1_vnum_f32(t, to+i, 0, t3);
	}
	#endif
}
#undef LDCVT

extern "C"
void lazy_init_f32_f16_in_omp_hm(
	HMGen<float>const& mg, Panels<float>& p, float scale, 
	int rowstart, int rowend, int colstart, int colend,
	int js, int je)
{
	typedef float F;
	typedef fp16 FLow;
	typedef DDAdaptor<F, FLow, true> DDA;
	typedef typename DDA::FDeck FDeck;
	size_t const lda = p.lda;
	size_t const ldl = DDA::get_ldl(p);
	int const b = p.b;
	int const i1 = p.i1;
	int const j1 = p.j1;
	int const istride = p.istride;
	int const jstride = p.jstride;
	F const alpha = -mg.alpha;
	F const beta = -mg.beta;
	F const ab = alpha*beta;
	for(int pj=colstart; pj<colend; ++pj){
		for(int j=js; j<je; ++j){
			int jstart = b*(j1 + pj*jstride);
			F const fpjj = jstart + j;
			F aij_j = alpha + ab * fpjj;
			for(int pi=rowstart; pi<rowend; ++pi){
				int istart = b*(i1 + pi*istride);
				F* __restrict__ to = p(pi, pj);
				FDeck* __restrict__ from = DDA::get_deck(p, pi, pj);
				assert(istart!=jstart);
				if(istart<jstart){
					for(int i=0; i<b; ++i){
						F aij = beta + ab * (istart + i);
						to[j*lda + i] = scale * static_cast<F>(from[j*ldl + i]) + aij;
					}
				}
				else {
					for(int i=0; i<b; ++i){
						to[j*lda + i] = scale * static_cast<F>(from[j*ldl + i]) + aij_j;
					}
				}
			}
		}
	}
}
extern "C"
void lazy_init_diag_f32_f16_in_omp_hm(
	HMGen<float>const& mg, Panels<float>& p, float scale, 
	int row, int col, int js, int je)
{
	typedef float F;
	typedef fp16 FLow;
	typedef DDAdaptor<F, FLow, true> DDA;
	typedef typename DDA::FDeck FDeck;
	size_t const lda = p.lda;
	size_t const ldl = DDA::get_ldl(p);
	int const b = p.b;
	int const i1 = p.i1;
	int const j1 = p.j1;
	int const istride = p.istride;
	int const jstride = p.jstride;
	int istart = b*(i1 + row*istride);
	int jstart = b*(j1 + col*jstride);
	F const alpha = -mg.alpha;
	F const beta = -mg.beta;
	F const ab = alpha*beta;
	F const done = 1;
	F* __restrict__ to = p(row, col);
	FDeck* __restrict__ from = DDA::get_deck(p, row, col);
	for(int j=js; j<je; ++j){
		F const fpjj = jstart + j;
		for(int i=0; i<j; ++i){
			F aij = beta + ab * (istart + i);
			to[j*lda + i] = scale * static_cast<F>(from[j*ldl + i]) + aij;
		}
		{
			to[j*lda + j] = scale * static_cast<F>(from[j*ldl + j]) + (done + ab * fpjj);
		}
		F aij_j = alpha + ab * fpjj;
		for(int i=j+1; i<b; ++i){
			to[j*lda + i] = scale * static_cast<F>(from[j*ldl + i]) + aij_j;
		}
		
	}
}

extern "C"
void lazy_init_f32_f16_in_omp_hm_bb(
	HMGen<float>const& mg, Panels<float>& p, float scale, 
	int rowstart, int rowend, int colstart, int colend,
	int js, int je, float* buf)
{
	typedef float F;
	typedef fp16 FLow;
	typedef DDAdaptor<F, FLow, true> DDA;
	typedef typename DDA::FDeck FDeck;
	size_t const lda = p.lda;
	size_t const ldl = DDA::get_ldl(p);
	int const b = p.b;
	int const i1 = p.i1;
	int const j1 = p.j1;
	int const istride = p.istride;
	int const jstride = p.jstride;
	F const alpha = -mg.alpha;
	F const beta = -mg.beta;
	F const ab = alpha*beta;
	FLow* __restrict__ lbuf = reinterpret_cast<FLow*>(buf);
	for(int pj=colstart; pj<colend; ++pj){
		int jstart = b*(j1 + pj*jstride);
		for(int j=js; j<je; ++j){
			F const fpjj = jstart + j;
			F aij_j = alpha + ab * fpjj;
			for(int pi=rowstart; pi<rowend; ++pi){
				int istart = b*(i1 + pi*istride);
				assert(istart!=jstart);
				F* __restrict__ to = p(pi, pj) + j*lda;
				FDeck* __restrict__ from = DDA::get_deck(p, pi, pj) + j * ldl;
				copy_xx(buf, to, b);
				if(istart<jstart)
					lishhmbb_up(istart, b, beta, ab, scale, from, lbuf, to);
				else 
					lishhmbb_low(b, aij_j, scale, from, lbuf, to);
				if(pi!=p.nprow-1) copy_xx(from, lbuf+b, b);
			}
		}
	}
}

extern "C"
void lazy_init_diag_f32_f16_in_omp_hm_bb(
	HMGen<float>const& mg, Panels<float>& p, float scale, 
	int row, int col, int js, int je, float* buf)
{
	typedef float F;
	typedef fp16 FLow;
	typedef DDAdaptor<F, FLow, true> DDA;
	typedef typename DDA::FDeck FDeck;
	size_t const lda = p.lda;
	size_t const ldl = DDA::get_ldl(p);
	int const b = p.b;
	int const i1 = p.i1;
	int const j1 = p.j1;
	int const istride = p.istride;
	int const jstride = p.jstride;
	int istart = b*(i1 + row*istride);
	int jstart = b*(j1 + col*jstride);
	F const alpha = -mg.alpha;
	F const beta = -mg.beta;
	F const ab = alpha*beta;
	F const done = 1;
	F* to1 = p(row, col);
	FDeck* from1 = DDA::get_deck(p, row, col);
	FLow* __restrict__ lbuf = reinterpret_cast<FLow*>(buf);
	F lbufscale = BB_NCYCLE;
	for(int j=js; j<je; ++j){
		F* __restrict__ to = to1 + j * lda;
		FDeck* __restrict__ from = from1 + j * ldl;
		F const fpjj = jstart + j;
		copy_xx(buf, to, b);
		lishhmbb_up(istart, j, beta, ab, scale, from, lbuf, to);
		{
			F xij = static_cast<F>(from[j]) + lbufscale*static_cast<F>(lbuf[j]);
			to[j] = scale * xij + (done + ab * fpjj);
		}
		F aij_j = alpha + ab * fpjj;
		if(j+1<b) lishhmbb_low(b-j-1, aij_j, scale, from+j+1, lbuf+j+1, to+j+1);
		if(row!=p.nprow-1) copy_xx(from, lbuf+b, b);
	}
}

#endif

extern "C" void fjtrad_omp_barrer(); // defined in main.cpp (to be compiled in trad mode)

extern "C"
void lazy_init_f32_f16_in_omp(
		Matgen<float>const& mg, Panels<float>& p, float scale, 
		int rowstart, int rowend, int colstart, int colend,
		int js, int je);

extern "C"
void lazy_init_diag_f32_f16_in_omp(
		Matgen<float>const& mg, Panels<float>& p, float scale, 
		int row, int col, const float *diag,
		int js, int je)
{
	typedef DDAdaptor<float, fp16, true> DDA;
	size_t const ldl = DDA::get_ldl(p);
	int const b = p.b;

	float dbuf[b];

	// Save diagonal part first
	fp16 * __restrict from = DDA::get_deck(p, row, col);
// #pragma omp for
	// for(int j=0; j<b; ++j){
	for(int j=js; j<je; ++j){
		float ajj = diag[b*row + j];
		ajj += scale * (float)from[j*ldl + j];
		dbuf[j] = ajj;
	}
// #pragma omp barrier
	fjtrad_omp_barrer();

	// Usual Lazy-init
	lazy_init_f32_f16_in_omp(mg, p, scale, row, row+1, col, col+1, js, je);
// #pragma omp barrier
	fjtrad_omp_barrer();

	// Overwrite the diagonal part
	float* __restrict to = p(row, col);
	size_t const lda = p.lda;
// #pragma omp for nowait
	// for(int j=0; j<b; ++j){
	for(int j=js; j<je; ++j){
		to[j*lda + j] = dbuf[j];
	}
}

extern "C"
void lazy_init_f32_f16_in_omp_bb(
		Matgen<float>const& mg, Panels<float>& p, float scale, 
		int rowstart, int rowend, int colstart, int colend,
		int js, int je, float* buf);
extern "C"
void lazy_init_diag_f32_f16_in_omp_bb(
		Matgen<float>const& mg, Panels<float>& p, float scale, 
		int row, int col, const float *diag,
		int js, int je, float *buf)
{
	typedef DDAdaptor<float, fp16, true> DDA;
	size_t const ldl = DDA::get_ldl(p);
	int const b = p.b;

	float dbuf[b];

	// Save diagonal part first
	fp16 * __restrict from = DDA::get_deck(p, row, col);
	float* __restrict to = p(row, col);
	fp16* lto = reinterpret_cast<fp16*>(to);
	size_t const lda = p.lda;
	float lbufscale = BB_NCYCLE;
// #pragma omp for
	// for(int j=0; j<b; ++j){
	for(int j=js; j<je; ++j){
		float ajj = diag[b*row + j];
		float xij = (float)from[j*ldl+j] + lbufscale * (float)lto[j*ldl+j];
		ajj += scale * xij;
		dbuf[j] = ajj;
	}
// #pragma omp barrier
	fjtrad_omp_barrer();

	// Usual Lazy-init
	lazy_init_f32_f16_in_omp_bb(mg, p, scale, row, row+1, col, col+1, js, je, buf);
// #pragma omp barrier
	fjtrad_omp_barrer();

	// Overwrite the diagonal part
// #pragma omp for nowait
	// for(int j=0; j<b; ++j){
	for(int j=js; j<je; ++j){
		to[j*lda + j] = dbuf[j];
	}
}

#if (!defined __aarch64__) || (defined UNIT_TEST)

#ifdef UNIT_TEST
#define lazy_init_f32_f16_in_omp lazy_init_f32_f16_in_omp_nosve
#endif
extern "C"
void lazy_init_f32_f16_in_omp(
		Matgen<float>const& mg, Panels<float>& p, float scale, 
		int rowstart, int rowend, int colstart, int colend,
		int js, int je)
{
	typedef DDAdaptor<float, fp16, true> DDA;
	size_t const lda = p.lda;
	size_t const ldl = DDA::get_ldl(p);

	int const b = p.b;
	int const i1 = p.i1;
	int const j1 = p.j1;
	int const istride = p.istride;
	int const jstride = p.jstride;
	int const istart = i1 + rowstart*istride;
	int const jstart = j1 + colstart*jstride;

	RandCoeff incl1 = mg.incl1;
	RandCoeff jumpi = mg.jumpi;
	RandCoeff jumpj = mg.jumpj; // jump (Q-1)*b columnst
	RandCoeff jumpn = mg.jumpn;

	RandCoeff jumpjp = jumpj * mg.jumpn.pow(b); // jump Q*b columnns
	RandStat stat_00 = {0};

	int joff_thread = -1;

	for(int pj=colstart; pj<colend; ++pj){
		RandStat stat_00_save = stat_00;
// #pragma omp for nowait
		// for(int j=0; j<b; ++j){
		for(int j=js; j<je; ++j){
			if(joff_thread < 0){
				joff_thread = j;
				stat_00_save = mg.jump(b*istart, b*jstart + joff_thread) 
					* RandStat::initialize(mg.seed);
				stat_00 = stat_00_save;
			}
			RandStat stat_i = stat_00;
			for(int pi=rowstart; pi<rowend; ++pi){
				float* __restrict to = p(pi, pj);
				fp16 * __restrict from = DDA::get_deck(p, pi, pj);
				for(int i=0; i<b; ++i){
					float aij = static_cast<float>(stat_i);
					float xij = static_cast<float>(from[j*ldl+i]);
					to[j*lda + i] = scale * xij + aij;
					stat_i = incl1 * stat_i;
				}
				stat_i = jumpi * stat_i;
			}
			stat_00 = jumpn * stat_00;
		}
		stat_00 = jumpjp * stat_00_save;
	}
}

extern "C"
void lazy_init_f32_f16_in_omp_bb(
		Matgen<float>const& mg, Panels<float>& p, float scale, 
		int rowstart, int rowend, int colstart, int colend,
		int js, int je, float* buf)
{
	typedef DDAdaptor<float, fp16, true> DDA;
	size_t const lda = p.lda;
	size_t const ldl = DDA::get_ldl(p);

	int const b = p.b;
	int const i1 = p.i1;
	int const j1 = p.j1;
	int const istride = p.istride;
	int const jstride = p.jstride;
	int const istart = i1 + rowstart*istride;
	int const jstart = j1 + colstart*jstride;

	RandCoeff incl1 = mg.incl1;
	RandCoeff jumpi = mg.jumpi;
	RandCoeff jumpj = mg.jumpj; // jump (Q-1)*b columnst
	RandCoeff jumpn = mg.jumpn;

	RandCoeff jumpjp = jumpj * mg.jumpn.pow(b); // jump Q*b columnns
	RandStat stat_00 = {0};

	fp16* lbuf = reinterpret_cast<fp16*>(buf);
	float lbufscale = BB_NCYCLE;
	int joff_thread = -1;

	for(int pj=colstart; pj<colend; ++pj){
		RandStat stat_00_save = stat_00;
// #pragma omp for nowait
		// for(int j=0; j<b; ++j){
		for(int j=js; j<je; ++j){
			if(joff_thread < 0){
				joff_thread = j;
				stat_00_save = mg.jump(b*istart, b*jstart + joff_thread) 
					* RandStat::initialize(mg.seed);
				stat_00 = stat_00_save;
			}
			RandStat stat_i = stat_00;
			for(int pi=rowstart; pi<rowend; ++pi){
				float* __restrict to = p(pi, pj);
				fp16 * __restrict from = DDA::get_deck(p, pi, pj);
				for(int i=0; i<b; ++i) buf[i] = to[j*lda + i];
				for(int i=0; i<b; ++i){
					float aij = static_cast<float>(stat_i);
					float xij = static_cast<float>(from[j*ldl+i]) + lbufscale * static_cast<float>(lbuf[i]);
					to[j*lda + i] = scale * xij + aij;
					stat_i = incl1 * stat_i;
				}
				stat_i = jumpi * stat_i;
				if(pi != p.nprow-1) for(int i=0; i<b; ++i) from[j*lda+i] = lbuf[b+i];
			}
			stat_00 = jumpn * stat_00;
		}
		stat_00 = jumpjp * stat_00_save;
	}
}
#endif

#ifdef UNIT_TEST
#include <stdlib.h>
#include <string.h>
int main(){
	enum{
		B = 16,
		N = 64,
		M = 48,
		LDA = N,
		LDL = 2*N,
	};
	float mat1[M][N]; // N x M col-major
	float mat2[M][N];

	// initialize
	puts("enter initialize");
	fflush(stdout);
	memset(mat1[0], 0, sizeof(mat1));
	fp16 *hptr = (fp16 *)mat1[0] + LDA;
	for(int j=0; j<M; j++, hptr+=LDL){
		for(int i=0; i<N; i++){
			hptr[i] = fp16(drand48() - 0.5);
		}
	}
	puts("enter memcpy");
	fflush(stdout);
	memcpy(mat2, mat1, sizeof(mat1));

	int istride = 3;
	int jstride = 2;
	Matgen<float> mg(42, N, B*(istride-1), B*(jstride-1), nullptr);
	Panels<float> p;
	p.b    = B;
	p.lda  = LDA;
	p.ldp  = B;
	p.ldpp = B * LDA;
	p.i1 = 16;
	p.j1 = 16;
	p.istride = istride;
	p.jstride = jstride;
	float scale = 3.0f;

	puts("enter nosve");
	fflush(stdout);
	p.p = mat1[0];
#pragma omp parallel
	{
		lazy_init_f32_f16_in_omp_nosve(mg, p, scale, 0, N/B, 0, M/B);
	}

	puts("enter sve");
	fflush(stdout);
	p.p = mat2[0];
#pragma omp parallel
	{
		lazy_init_f32_f16_in_omp(mg, p, scale, 0, N/B, 0, M/B);
	}

	puts("enter verify");
	fflush(stdout);
	for(int j=0; j<M; j++){
		for(int i=0; i<N; i++){
			if(!(mat1[j][i] == mat2[j][i])){
				printf("(%d,%d), %e, %e\n", i, j, mat1[j][i], mat2[j][i]);
			}
		}
	}

	return 0;
}
#endif
