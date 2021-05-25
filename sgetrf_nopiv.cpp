#include <cstdio>
#include <cstdint>
#include <cassert>
#include <omp.h>

#ifdef __FUJITSU
#define AVOID_LAMBDA_INDEX
#endif

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

namespace{

enum{
	NMAX = 1280,
};

float *thread_buffer(int tid, int nth){
	static float buf[NMAX * NMAX];
	size_t off = (size_t)tid * NMAX * NMAX / nth;
	return &buf[off];
}

// #ifdef UNIT_TEST
void sgetrf_nopiv_naive(int m, int n, float *aptr, size_t lda){
	auto a = [aptr, lda](int i, int j) -> float & {
		return aptr[i + lda*j];
	};

	const int kend = n<m ? n : m;
	for(int k=0; k<kend; k++){
		float inv = 1.0f / a(k, k);
		#if 0
		for(int i=k+1; i<m; i++){
			float aik = (a(i, k) *= inv);
			for(int j=k+1; j<n; j++){
				a(i, j) -= aik * a(k, j);
			}
		}
		#else
		for(int i=k+1; i<m; i++){
			a(i, k) *= inv;
		}
		for(int j=k+1; j<n; j++){
			float akj = a(k, j);
			#if 0
			for(int i=k+1; i<m; i++){
				a(i, j) -= a(i, k) * akj;
			}
			#else
			// saxpy(m-k-1, -akj, &a(k+1, k), &a(k+1, j));
			float * __restrict__ aj = &a(0,j);
			float * __restrict__ ak = &a(0,k);
			for(int i=k+1; i<m; i++){
				aj[i] -= ak[i] * akj;
			}
			#endif
		}
#endif
	}
}

//void sgetrf_nopiv_naive(int n, float *a, size_t lda){
//	sgetrf_nopiv_naive(n, n, a, lda);
//}
// #endif

template <int n>
void sgetrf_nopiv_inner(int m, float *a, size_t lda){
	sgetrf_nopiv_naive(m, n, a, lda);
}

template <int m>
void strsmLL_inner(const int n, const float *ap, float *bp, size_t ld){
	auto a = [ap, ld](int i, int j)->const float &{
		return ap[i + ld*j];
	};
	auto b = [bp, ld](int i, int j)->float &{
		return bp[i + ld*j];
	};
	for(int j=0; j<n; j++){
		// printf("%d: %d\n", omp_get_thread_num(), j);
		for(int k=0; k<m; k++){
			for(int i=k+1; i<m; i++){
				b(i,j) -= a(i,k) * b(k,j);
			}
		}
	}
}

template <int K>
void sgemm_inner(const int m, const int n, const float *ap, const float *bp, float *cp, size_t ld){
	if(0 == n) return;

	auto a = [ap, ld](int i, int j)->const float &{
		const float * __restrict__ arp = ap;
		return arp[i + ld*j];
	};
	auto b = [bp, ld](int i, int j)->const float &{
		const float * __restrict__ brp = bp;
		return brp[i + ld*j];
	};
	auto c = [cp, ld](int i, int j)->float &{
		float * __restrict__ crp = cp;
		return crp[i + ld*j];
	};
	for(int i=0; i<m; i++){
		for(int j=0; j<n; j++){
#if 0
			float sum = 0.0;
			for(int k=0; k<K; k++){
				sum += a(i,k) * b(k,j);
			}
			c(i,j) -= sum;
#else
			float sum = c(i,j);
			for(int k=0; k<K; k++){
				sum -= a(i,k) * b(k,j);
			}
			c(i,j) = sum;
#endif
		}
	}
}

// TEMPLATE SPECIALIZATIONS
template <>
void strsmLL_inner<3>(const int n, const float *ap, float *bp, size_t ld){
#ifndef AVOID_LAMBDA_INDEX
	auto a = [ap, ld](int i, int j)->const float &{
		return ap[i + ld*j];
	};
	auto b = [bp, ld](int i, int j)->float &{
		return bp[i + ld*j];
	};
#else
#  define a(i,j) ap[(i)+ld*(j)]
#  define b(i,j) bp[(i)+ld*(j)]
#endif

	float a10 = a(1,0);
	float a20 = a(2,0), a21 = a(2,1);
	for(int j=0; j<n; j++){
		float b0  = b(0,j);
		float b1  = b(1,j);
		float b2  = b(2,j);

		// k=0, i in [1,2]
		b1 -= a10 * b0;
		b2 -= a20 * b0;
		// k=1, i in [2]
		b2 -= a21 * b1;

		b(1,j) = b1;
		b(2,j) = b2;
	}
}

template <>
void strsmLL_inner<6>(const int n, const float *ap, float *bp, size_t ld){
#ifndef AVOID_LAMBDA_INDEX
	auto a = [ap, ld](int i, int j)->const float &{
		return ap[i + ld*j];
	};
	auto b = [bp, ld](int i, int j)->float &{
		return bp[i + ld*j];
	};
#endif

	float a10 = a(1,0);
	float a20 = a(2,0), a21 = a(2,1);
	float a30 = a(3,0), a31 = a(3,1), a32 = a(3,2);
	float a40 = a(4,0), a41 = a(4,1), a42 = a(4,2), a43 = a(4,3);
	float a50 = a(5,0), a51 = a(5,1), a52 = a(5,2), a53 = a(5,3), a54 = a(5,4);

	for(int j=0; j<n; j++){
		float b0  = b(0,j);
		float b1  = b(1,j);
		float b2  = b(2,j);
		float b3  = b(3,j);
		float b4  = b(4,j);
		float b5  = b(5,j);

		// k=0, i in [1,2,3,4,5]
		b1 -= a10 * b0;
		b2 -= a20 * b0;
		b3 -= a30 * b0;
		b4 -= a40 * b0;
		b5 -= a50 * b0;
		// k=1, i in [2,3,4,5]
		b2 -= a21 * b1;
		b3 -= a31 * b1;
		b4 -= a41 * b1;
		b5 -= a51 * b1;
		// k=2, i in [3,4,5]
		b3 -= a32 * b2;
		b4 -= a42 * b2;
		b5 -= a52 * b2;
		// k=3, i in [4,5]
		b4 -= a43 * b3;
		b5 -= a53 * b3;
		// k=4, i in [5]
		b5 -= a54 * b4;

		b(1,j) = b1;
		b(2,j) = b2;
		b(3,j) = b3;
		b(4,j) = b4;
		b(5,j) = b5;
	}
#ifdef AVOID_LAMBDA_INDEX
#undef a
#undef b
#endif
}

#ifdef __ARM_FEATURE_SVE

#ifdef __CLANG_FUJITSU
#define SVLD1_VNUM_F32(p, ptr, num)      svld1_f32(p, ptr + (num*vlen))
#define SVST1_VNUM_F32(p, ptr, num, val) svst1_f32(p, ptr + (num*vlen), val)
#endif
 
#ifdef __FUJITSU
#define SVLD1_VNUM_F32(p, ptr, num)      svld1_vnum_f32(p, ptr, num)
#define SVST1_VNUM_F32(p, ptr, num, val) svst1_vnum_f32(p, ptr, num, val) 
#endif

#if (!defined __CLANG_FUJITSU) || (!defined __FUJITSU)
//#define SVLD1_VNUM_F32(p, ptr, num)      svld1_f32(p, ptr + (num*vlen))
//#define SVST1_VNUM_F32(p, ptr, num, val) svst1_f32(p, ptr + (num*vlen), val)
#define SVLD1_VNUM_F32(p, ptr, num)      svld1_vnum_f32(p, ptr, num)
#define SVST1_VNUM_F32(p, ptr, num, val) svst1_vnum_f32(p, ptr, num, val) 
#endif
 

template <>
void sgemm_inner<3>(const int m, const int n, const float *ap, const float *bp, float *cp, size_t ld){
	if(0 == n) return;

	assert(0 == n%3);

#ifndef AVOID_LAMBDA_INDEX
	auto a = [ap, ld](int i, int j)->const float &{
		const float * __restrict__ arp = ap;
		return arp[i + ld*j];
	};
	auto b = [bp, ld](int i, int j)->const float &{
		const float * __restrict__ brp = bp;
		return brp[i + ld*j];
	};
	auto c = [cp, ld](int i, int j)->float &{
		float * __restrict__ crp = cp;
		return crp[i + ld*j];
	};
#else
#  define a(i,j) ap[(i)+ld*(j)]
#  define b(i,j) bp[(i)+ld*(j)]
#  define c(i,j) cp[(i)+ld*(j)]
#endif

	const int vlen = svcntw(); // assumed to be 16
	if(3 == n){
		int i=0;
		svbool_t p0 = svwhilelt_b32(i+0*vlen, m);
		svbool_t p1 = svwhilelt_b32(i+1*vlen, m);
		// 2x6 blocking for A and C
		svfloat32_t a00 = SVLD1_VNUM_F32(p0, &a(i,0), 0);
		svfloat32_t a10 = SVLD1_VNUM_F32(p1, &a(i,0), 1);
		svfloat32_t a01 = SVLD1_VNUM_F32(p0, &a(i,1), 0);
		svfloat32_t a11 = SVLD1_VNUM_F32(p1, &a(i,1), 1);
		svfloat32_t a02 = SVLD1_VNUM_F32(p0, &a(i,2), 0);
		svfloat32_t a12 = SVLD1_VNUM_F32(p1, &a(i,2), 1);

		for( ; i<m; ){
			int j=0;
			svfloat32_t c00 = SVLD1_VNUM_F32(p0, &c(i,j+0), 0);
			svfloat32_t c10 = SVLD1_VNUM_F32(p1, &c(i,j+0), 1);
			svfloat32_t c01 = SVLD1_VNUM_F32(p0, &c(i,j+1), 0);
			svfloat32_t c11 = SVLD1_VNUM_F32(p1, &c(i,j+1), 1);
			svfloat32_t c02 = SVLD1_VNUM_F32(p0, &c(i,j+2), 0);
			svfloat32_t c12 = SVLD1_VNUM_F32(p1, &c(i,j+2), 1);
			#define MM_KERNEL3(K) {\
				float b ## K ## 0 = b(K,j+0); \
				c00 = svmls_n_f32_x(p0, c00, a0 ## K, b ## K ## 0); \
				c10 = svmls_n_f32_x(p1, c10, a1 ## K, b ## K ## 0); \
				float b ## K ## 1 = b(K,j+1); \
				c01 = svmls_n_f32_x(p0, c01, a0 ## K, b ## K ## 1); \
				c11 = svmls_n_f32_x(p1, c11, a1 ## K, b ## K ## 1); \
				float b ## K ## 2 = b(K,j+2); \
				c02 = svmls_n_f32_x(p0, c02, a0 ## K, b ## K ## 2); \
				c12 = svmls_n_f32_x(p1, c12, a1 ## K, b ## K ## 2); \
			}

			MM_KERNEL3(0);
			MM_KERNEL3(1);
			MM_KERNEL3(2);

			SVST1_VNUM_F32(p0, &c(i,j+0), 0, c00);
			SVST1_VNUM_F32(p1, &c(i,j+0), 1, c10);
			SVST1_VNUM_F32(p0, &c(i,j+1), 0, c01);
			SVST1_VNUM_F32(p1, &c(i,j+1), 1, c11);
			SVST1_VNUM_F32(p0, &c(i,j+2), 0, c02);
			SVST1_VNUM_F32(p1, &c(i,j+2), 1, c12);

			i+=2*vlen;
			p0 = svwhilelt_b32(i+0*vlen, m);
			p1 = svwhilelt_b32(i+1*vlen, m);
			// 2x6 blocking for A and C
			a00 = SVLD1_VNUM_F32(p0, &a(i,0), 0);
			a10 = SVLD1_VNUM_F32(p1, &a(i,0), 1);
			a01 = SVLD1_VNUM_F32(p0, &a(i,1), 0);
			a11 = SVLD1_VNUM_F32(p1, &a(i,1), 1);
			a02 = SVLD1_VNUM_F32(p0, &a(i,2), 0);
			a12 = SVLD1_VNUM_F32(p1, &a(i,2), 1);
		}
	}else{
		for(int i=0; i<m; i+=2*vlen){
			svbool_t p0 = svwhilelt_b32(i+0*vlen, m);
			svbool_t p1 = svwhilelt_b32(i+1*vlen, m);

			// 2x6 blocking for A and C
			svfloat32_t a00 = SVLD1_VNUM_F32(p0, &a(i,0), 0);
			svfloat32_t a10 = SVLD1_VNUM_F32(p1, &a(i,0), 1);
			svfloat32_t a01 = SVLD1_VNUM_F32(p0, &a(i,1), 0);
			svfloat32_t a11 = SVLD1_VNUM_F32(p1, &a(i,1), 1);
			svfloat32_t a02 = SVLD1_VNUM_F32(p0, &a(i,2), 0);
			svfloat32_t a12 = SVLD1_VNUM_F32(p1, &a(i,2), 1);

			for(int j=0; j<n; j+=3){
				svfloat32_t c00 = SVLD1_VNUM_F32(p0, &c(i,j+0), 0);
				svfloat32_t c10 = SVLD1_VNUM_F32(p1, &c(i,j+0), 1);
				svfloat32_t c01 = SVLD1_VNUM_F32(p0, &c(i,j+1), 0);
				svfloat32_t c11 = SVLD1_VNUM_F32(p1, &c(i,j+1), 1);
				svfloat32_t c02 = SVLD1_VNUM_F32(p0, &c(i,j+2), 0);
				svfloat32_t c12 = SVLD1_VNUM_F32(p1, &c(i,j+2), 1);

				MM_KERNEL3(0);
				MM_KERNEL3(1);
				MM_KERNEL3(2);

				#undef MM_KERNEL3

				SVST1_VNUM_F32(p0, &c(i,j+0), 0, c00);
				SVST1_VNUM_F32(p1, &c(i,j+0), 1, c10);
				SVST1_VNUM_F32(p0, &c(i,j+1), 0, c01);
				SVST1_VNUM_F32(p1, &c(i,j+1), 1, c11);
				SVST1_VNUM_F32(p0, &c(i,j+2), 0, c02);
				SVST1_VNUM_F32(p1, &c(i,j+2), 1, c12);
			}

		}
	}
#ifdef AVOID_LAMBDA_INDEX
#undef a
#undef b
#undef c
#endif
}

template <>
void sgemm_inner<6>(const int m, const int n, const float * __restrict__ ap, const float * __restrict__ bp, float * __restrict__ cp, size_t ld){
	if(0 == n) return;

	// assert(0 == n%6);
	// 幅が6の倍数でないときにゴミを書き込むが動くはず

#ifndef AVOID_LAMBDA_INDEX
	auto a = [ap, ld](long i, long j)->const float &{
		const float * __restrict__ arp = ap;
		return arp[i + ld*j];
	};
	auto b = [bp, ld](long i, long j)->const float &{
		const float * __restrict__ brp = bp;
		return brp[i + ld*j];
	};
	auto c = [cp, ld](long i, long j)->float &{
		float * __restrict__ crp = cp;
		return crp[i + ld*j];
	};
#else
#  define a(i,j) ap[(i)+ld*(j)]
#  define b(i,j) bp[(i)+ld*(j)]
#  define c(i,j) cp[(i)+ld*(j)]
#endif

	const int vlen = svcntw(); // assumed to be 16

	if(6==n){ // mostly, n=6 or 12
		int i=0;
		const int j=0;
		svbool_t p0 = svwhilelt_b32(i+0*vlen, m);
		svbool_t p1 = svwhilelt_b32(i+1*vlen, m);
		// 2x6 blocking for A and C
		svfloat32_t a00 = SVLD1_VNUM_F32(p0, &a(i,0), 0);
		svfloat32_t a10 = SVLD1_VNUM_F32(p1, &a(i,0), 1);
		svfloat32_t a01 = SVLD1_VNUM_F32(p0, &a(i,1), 0);
		svfloat32_t a11 = SVLD1_VNUM_F32(p1, &a(i,1), 1);
		svfloat32_t a02 = SVLD1_VNUM_F32(p0, &a(i,2), 0);
		svfloat32_t a12 = SVLD1_VNUM_F32(p1, &a(i,2), 1);
		svfloat32_t a03 = SVLD1_VNUM_F32(p0, &a(i,3), 0);
		svfloat32_t a13 = SVLD1_VNUM_F32(p1, &a(i,3), 1);
		svfloat32_t a04 = SVLD1_VNUM_F32(p0, &a(i,4), 0);
		svfloat32_t a14 = SVLD1_VNUM_F32(p1, &a(i,4), 1);
		svfloat32_t a05 = SVLD1_VNUM_F32(p0, &a(i,5), 0);
		svfloat32_t a15 = SVLD1_VNUM_F32(p1, &a(i,5), 1);
		for( ; i<m; ){
			svfloat32_t c00 = SVLD1_VNUM_F32(p0, &c(i,j+0), 0);
			svfloat32_t c10 = SVLD1_VNUM_F32(p1, &c(i,j+0), 1);
			svfloat32_t c01 = SVLD1_VNUM_F32(p0, &c(i,j+1), 0);
			svfloat32_t c11 = SVLD1_VNUM_F32(p1, &c(i,j+1), 1);
			svfloat32_t c02 = SVLD1_VNUM_F32(p0, &c(i,j+2), 0);
			svfloat32_t c12 = SVLD1_VNUM_F32(p1, &c(i,j+2), 1);
			svfloat32_t c03 = SVLD1_VNUM_F32(p0, &c(i,j+3), 0);
			svfloat32_t c13 = SVLD1_VNUM_F32(p1, &c(i,j+3), 1);
			svfloat32_t c04 = SVLD1_VNUM_F32(p0, &c(i,j+4), 0);
			svfloat32_t c14 = SVLD1_VNUM_F32(p1, &c(i,j+4), 1);
			svfloat32_t c05 = SVLD1_VNUM_F32(p0, &c(i,j+5), 0);
			svfloat32_t c15 = SVLD1_VNUM_F32(p1, &c(i,j+5), 1);

			#define MM_KERNEL6(K) {\
				float b ## K ## 0 = b(K,j+0); \
				c00 = svmls_n_f32_x(p0, c00, a0 ## K, b ## K ## 0); \
				c10 = svmls_n_f32_x(p1, c10, a1 ## K, b ## K ## 0); \
				float b ## K ## 1 = b(K,j+1); \
				c01 = svmls_n_f32_x(p0, c01, a0 ## K, b ## K ## 1); \
				c11 = svmls_n_f32_x(p1, c11, a1 ## K, b ## K ## 1); \
				float b ## K ## 2 = b(K,j+2); \
				c02 = svmls_n_f32_x(p0, c02, a0 ## K, b ## K ## 2); \
				c12 = svmls_n_f32_x(p1, c12, a1 ## K, b ## K ## 2); \
				float b ## K ## 3 = b(K,j+3); \
				c03 = svmls_n_f32_x(p0, c03, a0 ## K, b ## K ## 3); \
				c13 = svmls_n_f32_x(p1, c13, a1 ## K, b ## K ## 3); \
				float b ## K ## 4 = b(K,j+4); \
				c04 = svmls_n_f32_x(p0, c04, a0 ## K, b ## K ## 4); \
				c14 = svmls_n_f32_x(p1, c14, a1 ## K, b ## K ## 4); \
				float b ## K ## 5 = b(K,j+5); \
				c05 = svmls_n_f32_x(p0, c05, a0 ## K, b ## K ## 5); \
				c15 = svmls_n_f32_x(p1, c15, a1 ## K, b ## K ## 5); \
			}
			MM_KERNEL6(0);
			MM_KERNEL6(1);
			MM_KERNEL6(2);
			MM_KERNEL6(3);
			MM_KERNEL6(4);
			MM_KERNEL6(5);

			SVST1_VNUM_F32(p0, &c(i,j+0), 0, c00);
			SVST1_VNUM_F32(p1, &c(i,j+0), 1, c10);
			SVST1_VNUM_F32(p0, &c(i,j+1), 0, c01);
			SVST1_VNUM_F32(p1, &c(i,j+1), 1, c11);
			SVST1_VNUM_F32(p0, &c(i,j+2), 0, c02);
			SVST1_VNUM_F32(p1, &c(i,j+2), 1, c12);
			SVST1_VNUM_F32(p0, &c(i,j+3), 0, c03);
			SVST1_VNUM_F32(p1, &c(i,j+3), 1, c13);
			SVST1_VNUM_F32(p0, &c(i,j+4), 0, c04);
			SVST1_VNUM_F32(p1, &c(i,j+4), 1, c14);
			SVST1_VNUM_F32(p0, &c(i,j+5), 0, c05);
			SVST1_VNUM_F32(p1, &c(i,j+5), 1, c15);

			i+=2*vlen;
			p0 = svwhilelt_b32(i+0*vlen, m);
			p1 = svwhilelt_b32(i+1*vlen, m);
			// 2x6 blocking for A and C
			a00 = SVLD1_VNUM_F32(p0, &a(i,0), 0);
			a10 = SVLD1_VNUM_F32(p1, &a(i,0), 1);
			a01 = SVLD1_VNUM_F32(p0, &a(i,1), 0);
			a11 = SVLD1_VNUM_F32(p1, &a(i,1), 1);
			a02 = SVLD1_VNUM_F32(p0, &a(i,2), 0);
			a12 = SVLD1_VNUM_F32(p1, &a(i,2), 1);
			a03 = SVLD1_VNUM_F32(p0, &a(i,3), 0);
			a13 = SVLD1_VNUM_F32(p1, &a(i,3), 1);
			a04 = SVLD1_VNUM_F32(p0, &a(i,4), 0);
			a14 = SVLD1_VNUM_F32(p1, &a(i,4), 1);
			a05 = SVLD1_VNUM_F32(p0, &a(i,5), 0);
			a15 = SVLD1_VNUM_F32(p1, &a(i,5), 1);
		}
	}else if (12==n){
		for(int i=0; i<m; i+=2*vlen){
			svbool_t p0 = svwhilelt_b32(i+0*vlen, m);
			svbool_t p1 = svwhilelt_b32(i+1*vlen, m);

			// 2x6 blocking for A and C
			svfloat32_t a00 = SVLD1_VNUM_F32(p0, &a(i,0), 0);
			svfloat32_t a10 = SVLD1_VNUM_F32(p1, &a(i,0), 1);
			svfloat32_t a01 = SVLD1_VNUM_F32(p0, &a(i,1), 0);
			svfloat32_t a11 = SVLD1_VNUM_F32(p1, &a(i,1), 1);
			svfloat32_t a02 = SVLD1_VNUM_F32(p0, &a(i,2), 0);
			svfloat32_t a12 = SVLD1_VNUM_F32(p1, &a(i,2), 1);
			svfloat32_t a03 = SVLD1_VNUM_F32(p0, &a(i,3), 0);
			svfloat32_t a13 = SVLD1_VNUM_F32(p1, &a(i,3), 1);
			svfloat32_t a04 = SVLD1_VNUM_F32(p0, &a(i,4), 0);
			svfloat32_t a14 = SVLD1_VNUM_F32(p1, &a(i,4), 1);
			svfloat32_t a05 = SVLD1_VNUM_F32(p0, &a(i,5), 0);
			svfloat32_t a15 = SVLD1_VNUM_F32(p1, &a(i,5), 1);
			// for(int j=0; j<12; j+=6){
			{
				int j=0;
				svfloat32_t c00 = SVLD1_VNUM_F32(p0, &c(i,j+0), 0);
				svfloat32_t c10 = SVLD1_VNUM_F32(p1, &c(i,j+0), 1);
				svfloat32_t c01 = SVLD1_VNUM_F32(p0, &c(i,j+1), 0);
				svfloat32_t c11 = SVLD1_VNUM_F32(p1, &c(i,j+1), 1);
				svfloat32_t c02 = SVLD1_VNUM_F32(p0, &c(i,j+2), 0);
				svfloat32_t c12 = SVLD1_VNUM_F32(p1, &c(i,j+2), 1);
				svfloat32_t c03 = SVLD1_VNUM_F32(p0, &c(i,j+3), 0);
				svfloat32_t c13 = SVLD1_VNUM_F32(p1, &c(i,j+3), 1);
				svfloat32_t c04 = SVLD1_VNUM_F32(p0, &c(i,j+4), 0);
				svfloat32_t c14 = SVLD1_VNUM_F32(p1, &c(i,j+4), 1);
				svfloat32_t c05 = SVLD1_VNUM_F32(p0, &c(i,j+5), 0);
				svfloat32_t c15 = SVLD1_VNUM_F32(p1, &c(i,j+5), 1);

				MM_KERNEL6(0);
				MM_KERNEL6(1);
				MM_KERNEL6(2);
				MM_KERNEL6(3);
				MM_KERNEL6(4);
				MM_KERNEL6(5);

				SVST1_VNUM_F32(p0, &c(i,j+0), 0, c00);
				SVST1_VNUM_F32(p1, &c(i,j+0), 1, c10);
				SVST1_VNUM_F32(p0, &c(i,j+1), 0, c01);
				SVST1_VNUM_F32(p1, &c(i,j+1), 1, c11);
				SVST1_VNUM_F32(p0, &c(i,j+2), 0, c02);
				SVST1_VNUM_F32(p1, &c(i,j+2), 1, c12);
				SVST1_VNUM_F32(p0, &c(i,j+3), 0, c03);
				SVST1_VNUM_F32(p1, &c(i,j+3), 1, c13);
				SVST1_VNUM_F32(p0, &c(i,j+4), 0, c04);
				SVST1_VNUM_F32(p1, &c(i,j+4), 1, c14);
				SVST1_VNUM_F32(p0, &c(i,j+5), 0, c05);
				SVST1_VNUM_F32(p1, &c(i,j+5), 1, c15);
			}
			{
				int j=6;
				svfloat32_t c00 = SVLD1_VNUM_F32(p0, &c(i,j+0), 0);
				svfloat32_t c10 = SVLD1_VNUM_F32(p1, &c(i,j+0), 1);
				svfloat32_t c01 = SVLD1_VNUM_F32(p0, &c(i,j+1), 0);
				svfloat32_t c11 = SVLD1_VNUM_F32(p1, &c(i,j+1), 1);
				svfloat32_t c02 = SVLD1_VNUM_F32(p0, &c(i,j+2), 0);
				svfloat32_t c12 = SVLD1_VNUM_F32(p1, &c(i,j+2), 1);
				svfloat32_t c03 = SVLD1_VNUM_F32(p0, &c(i,j+3), 0);
				svfloat32_t c13 = SVLD1_VNUM_F32(p1, &c(i,j+3), 1);
				svfloat32_t c04 = SVLD1_VNUM_F32(p0, &c(i,j+4), 0);
				svfloat32_t c14 = SVLD1_VNUM_F32(p1, &c(i,j+4), 1);
				svfloat32_t c05 = SVLD1_VNUM_F32(p0, &c(i,j+5), 0);
				svfloat32_t c15 = SVLD1_VNUM_F32(p1, &c(i,j+5), 1);

				MM_KERNEL6(0);
				MM_KERNEL6(1);
				MM_KERNEL6(2);
				MM_KERNEL6(3);
				MM_KERNEL6(4);
				MM_KERNEL6(5);

				SVST1_VNUM_F32(p0, &c(i,j+0), 0, c00);
				SVST1_VNUM_F32(p1, &c(i,j+0), 1, c10);
				SVST1_VNUM_F32(p0, &c(i,j+1), 0, c01);
				SVST1_VNUM_F32(p1, &c(i,j+1), 1, c11);
				SVST1_VNUM_F32(p0, &c(i,j+2), 0, c02);
				SVST1_VNUM_F32(p1, &c(i,j+2), 1, c12);
				SVST1_VNUM_F32(p0, &c(i,j+3), 0, c03);
				SVST1_VNUM_F32(p1, &c(i,j+3), 1, c13);
				SVST1_VNUM_F32(p0, &c(i,j+4), 0, c04);
				SVST1_VNUM_F32(p1, &c(i,j+4), 1, c14);
				SVST1_VNUM_F32(p0, &c(i,j+5), 0, c05);
				SVST1_VNUM_F32(p1, &c(i,j+5), 1, c15);
			}
		}
	}else{ // general case
		for(int i=0; i<m; i+=2*vlen){
			svbool_t p0 = svwhilelt_b32(i+0*vlen, m);
			svbool_t p1 = svwhilelt_b32(i+1*vlen, m);

			// 2x6 blocking for A and C
			svfloat32_t a00 = SVLD1_VNUM_F32(p0, &a(i,0), 0);
			svfloat32_t a10 = SVLD1_VNUM_F32(p1, &a(i,0), 1);
			svfloat32_t a01 = SVLD1_VNUM_F32(p0, &a(i,1), 0);
			svfloat32_t a11 = SVLD1_VNUM_F32(p1, &a(i,1), 1);
			svfloat32_t a02 = SVLD1_VNUM_F32(p0, &a(i,2), 0);
			svfloat32_t a12 = SVLD1_VNUM_F32(p1, &a(i,2), 1);
			svfloat32_t a03 = SVLD1_VNUM_F32(p0, &a(i,3), 0);
			svfloat32_t a13 = SVLD1_VNUM_F32(p1, &a(i,3), 1);
			svfloat32_t a04 = SVLD1_VNUM_F32(p0, &a(i,4), 0);
			svfloat32_t a14 = SVLD1_VNUM_F32(p1, &a(i,4), 1);
			svfloat32_t a05 = SVLD1_VNUM_F32(p0, &a(i,5), 0);
			svfloat32_t a15 = SVLD1_VNUM_F32(p1, &a(i,5), 1);
#pragma loop novrec
			for(int j=0; j<n; j+=6){
				svfloat32_t c00 = SVLD1_VNUM_F32(p0, &c(i,j+0), 0);
				svfloat32_t c10 = SVLD1_VNUM_F32(p1, &c(i,j+0), 1);
				svfloat32_t c01 = SVLD1_VNUM_F32(p0, &c(i,j+1), 0);
				svfloat32_t c11 = SVLD1_VNUM_F32(p1, &c(i,j+1), 1);
				svfloat32_t c02 = SVLD1_VNUM_F32(p0, &c(i,j+2), 0);
				svfloat32_t c12 = SVLD1_VNUM_F32(p1, &c(i,j+2), 1);
				svfloat32_t c03 = SVLD1_VNUM_F32(p0, &c(i,j+3), 0);
				svfloat32_t c13 = SVLD1_VNUM_F32(p1, &c(i,j+3), 1);
				svfloat32_t c04 = SVLD1_VNUM_F32(p0, &c(i,j+4), 0);
				svfloat32_t c14 = SVLD1_VNUM_F32(p1, &c(i,j+4), 1);
				svfloat32_t c05 = SVLD1_VNUM_F32(p0, &c(i,j+5), 0);
				svfloat32_t c15 = SVLD1_VNUM_F32(p1, &c(i,j+5), 1);

				MM_KERNEL6(0);
				MM_KERNEL6(1);
				MM_KERNEL6(2);
				MM_KERNEL6(3);
				MM_KERNEL6(4);
				MM_KERNEL6(5);
				#undef MM_KERNEL6

				SVST1_VNUM_F32(p0, &c(i,j+0), 0, c00);
				SVST1_VNUM_F32(p1, &c(i,j+0), 1, c10);
				SVST1_VNUM_F32(p0, &c(i,j+1), 0, c01);
				SVST1_VNUM_F32(p1, &c(i,j+1), 1, c11);
				SVST1_VNUM_F32(p0, &c(i,j+2), 0, c02);
				SVST1_VNUM_F32(p1, &c(i,j+2), 1, c12);
				SVST1_VNUM_F32(p0, &c(i,j+3), 0, c03);
				SVST1_VNUM_F32(p1, &c(i,j+3), 1, c13);
				SVST1_VNUM_F32(p0, &c(i,j+4), 0, c04);
				SVST1_VNUM_F32(p1, &c(i,j+4), 1, c14);
				SVST1_VNUM_F32(p0, &c(i,j+5), 0, c05);
				SVST1_VNUM_F32(p1, &c(i,j+5), 1, c15);
			}
		}
	}
#ifdef AVOID_LAMBDA_INDEX
#undef a
#undef b
#undef c
#endif
}

template <>
void sgetrf_nopiv_inner<3>(int m, float *aptr, size_t lda){
	if(m < 2) return;

#ifndef AVOID_LAMBDA_INDEX
	auto a = [aptr, lda](int i, int j)->float &{
		return aptr[i + lda*j];
	};
#else
#  define a(i,j) aptr[(i)+lda*(j)]
#endif
	float a00 = a(0,0), a01 = a(0,1), a02 = a(0,2);
	float a10 = a(1,0), a11 = a(1,1), a12 = a(1,2);
	float a20 = a(2,0), a21 = a(2,1), a22 = a(2,2);

	// k=0
	float d00 = 1.0f / a00;
	a10 *= d00;
	a20 *= d00;
	a11 -= a10 * a01, a12 -= a10 * a02;
	a21 -= a20 * a01, a22 -= a20 * a02;
	// k=1
	float d11 = 1.0f / a11;
	a21 *= d11;
	a22 -= a21 * a12;
	// k=2
	float d22 = 1.0f / a22;

	a(1,0) = a10, a(1,1) = a11, a(1,2) = a12;
	if(m < 3) return;
	a(2,0) = a20, a(2,1) = a21, a(2,2) = a22;

	// strsm, Right, Upper, NonDiag
	svfloat32_t d00v = svdup_f32(d00);
	svfloat32_t d11v = svdup_f32(d11);
	svfloat32_t d22v = svdup_f32(d22);

	svfloat32_t a01v = svdup_f32(a01);
	svfloat32_t a02v = svdup_f32(a02);
	svfloat32_t a12v = svdup_f32(a12);

	const int vlen = svcntw(); // assumed to be 16
#if 0
	int i=3;
	svbool_t p = svwhilelt_b32(i, m);
	svfloat32_t a0 = svld1_f32(p, &a(i,0));
	svfloat32_t a1 = svld1_f32(p, &a(i,1));
	svfloat32_t a2 = svld1_f32(p, &a(i,2));
	for(; i<m; ){
		a0 = svmul_f32_x(p, a0, d00v);
		a1 = svmls_f32_x(p, a1, a0, a01v);
		a2 = svmls_f32_x(p, a2, a0, a02v);

		a1 = svmul_f32_x(p, a1, d11v);
		a2 = svmls_f32_x(p, a2, a1, a12v);

		a2 = svmul_f32_x(p, a2, d22v);

		svst1_f32(p, &a(i,0), a0);
		svst1_f32(p, &a(i,1), a1);
		svst1_f32(p, &a(i,2), a2);

		i+=vlen;
		p = svwhilelt_b32(i, m);
		a0 = svld1_f32(p, &a(i,0));
		a1 = svld1_f32(p, &a(i,1));
		a2 = svld1_f32(p, &a(i,2));
	}
#else
	// 3-stage software pipelining (by hand)
	svbool_t p0, p1, p2;
	svfloat32_t a0_i0, a1_i0, a2_i0;
	svfloat32_t a0_i1, a1_i1, a2_i1;
	svfloat32_t a0_i2, a1_i2, a2_i2;
	int i=3;
#define STAGE0(I, II) \
	p    ## I = svwhilelt_b32 (i+II*vlen, m); \
	a0_i ## I = SVLD1_VNUM_F32(p ## I, &a(i,0), II); \
	a1_i ## I = SVLD1_VNUM_F32(p ## I, &a(i,1), II); \
	a0_i ## I = svmul_f32_x   (p ## I, a0_i ## I, d00v); 
#define STAGE1(I, II) \
	a2_i ## I = SVLD1_VNUM_F32(p ## I, &a(i,2), II); \
	SVST1_VNUM_F32            (p ## I, &a(i,0), II, a0_i ## I); \
	a1_i ## I = svmls_f32_x   (p ## I, a1_i ## I, a0_i ## I, a01v); \
	a2_i ## I = svmls_f32_x   (p ## I, a2_i ## I, a0_i ## I, a02v); \
	a1_i ## I = svmul_f32_x   (p ## I, a1_i ## I, d11v);
#define STAGE2(I, II) \
	a2_i ## I = svmls_f32_x   (p ## I, a2_i ## I, a1_i ## I, a12v); \
	a2_i ## I = svmul_f32_x   (p ## I, a2_i ## I, d22v); \
	SVST1_VNUM_F32            (p ## I, &a(i,1), II, a1_i ## I); \
	SVST1_VNUM_F32            (p ## I, &a(i,2), II, a2_i ## I); \

#if 0
	// 0-0 (stage 0 iter 0)
	STAGE0(0,0);

	// 0-1, 1-0
	STAGE0(1, 1);
	STAGE1(0, 0);

	for( ; i<m-2*vlen; i+=3*vlen){
		// 0-2, 1-1, 2-0
		STAGE0(2,2)
		STAGE1(1,1)
		STAGE2(0,0)

		// 0-0', 1-2, 2-1
		STAGE0(0,3)
		STAGE1(2,2)
		STAGE2(1,1)

		// 0-1', 1-0', 2-2
		STAGE0(1,4);
		STAGE1(0,3);
		STAGE2(2,2);
	}
	// 1-1, 2-0
	STAGE1(1,1);
	STAGE2(0,0);

	// 2-1
	STAGE2(1,1);
#else
	const int nn = 1 + ((m-3)-1)/vlen;
	if(1 == nn){
		STAGE0(0,0);
		STAGE1(0,0);
		STAGE2(0,0);
		return;
	}
	// 0-0 (stage 0 iter 0)
	STAGE0(0,0);

	// 0-1, 1-0
	STAGE0(1,1);
	STAGE1(0,0);

	int nend = nn - 4;
	// nend += 1; // !!?

	for(int ii=0 ; ii<nend; ii+=3, i+=3*vlen){
		// 0-2, 1-1, 2-0
		STAGE0(2,2); STAGE1(1,1); STAGE2(0,0);
		// 0-0', 1-2, 2-1
		STAGE0(0,3); STAGE1(2,2); STAGE2(1,1);
		// 0-1', 1-0', 2-2
		STAGE0(1,4); STAGE1(0,3); STAGE2(2,2);
	}

	if(2 == nn%3){
		// 1-1, 2-0
		             STAGE1(1,1); STAGE2(0,0);
		// 2-1
		                          STAGE2(1,1);
	}else if(0 == nn%3){
		// 0-2, 1-1, 2-0
		STAGE0(2,2); STAGE1(1,1); STAGE2(0,0);
		// 1-2,  2-1
		             STAGE1(2,2); STAGE2(1,1);
		// 2-2
		                          STAGE2(2,2);
	}else if(1 == nn%3){
		// 0-2, 1-1, 2-0
		STAGE0(2,2); STAGE1(1,1); STAGE2(0,0);
		// 0-0', 1-2,  2-1
		STAGE0(0,3); STAGE1(2,2); STAGE2(1,1);
		// 1-0', 2-2
		             STAGE1(0,3); STAGE2(2,2);
		// 2-0'
		                          STAGE2(0,3);
	}
#endif
#undef STAGE0
#undef STAGE1
#undef STAGE2
#endif

#ifdef AVOID_LAMBDA_INDEX
#undef a
#endif
}

#endif // SVE

template <>
void sgetrf_nopiv_inner<6>(int m, float *aptr, size_t lda){
	auto a = [aptr, lda](int i, int j)->float &{
		return aptr[i + lda*j];
	};

	sgetrf_nopiv_inner<3>(m, &a(0,0), lda);

	strsmLL_inner<3>(3, &a(0,0), &a(0,3), lda);

	sgemm_inner<3>(m-3, 3, &a(3,0), &a(0,3), &a(3,3), lda);

	sgetrf_nopiv_inner<3>(m-3, &a(3,3), lda);
}

// end of TEMPLATE SPECIALIZATIONS

template <int bs>
void 
__attribute__((noinline))
sgetrf_nopiv_omp(int n, float *aext, size_t lda)
{
	auto ax = [aext, lda](int i, int j) -> float & {
		return aext[i + lda*j];
	};
#pragma omp parallel
	{
		const int tid = omp_get_thread_num();
		const int nth = omp_get_num_threads();
		
		float *aloc = thread_buffer(tid, nth);

		auto athread = [n, nth](int tid, int i=0, int j=0) -> float & {
			float *p = thread_buffer(tid, nth);
			return p[i + n*j];
		};
		auto al = [aloc, n](int i, int j) -> float & {
			return aloc[i + n*j];
		};

		// copy in
		int myjend = 0;
		for(int j=bs*tid; j<n; j+=bs*nth){
			for(int jj=0; jj<bs; jj++){
				if(j+jj >= n){
					j = n;
					break;
				}
				float * __restrict__ dst = &al(0, myjend);
				float * __restrict__ src = &ax(0, j+jj);
				for(int i=0; i<n; i++){
					dst[i] = src[i];
				}
				myjend++;
			}
		}

		// LU decomp.
		int myj = 0;
		for(int k=0, kb=0;  k<n; k+=bs,  kb++){
			const int who   = kb % nth;
			const int where = kb / nth;

			int icol = k;
			int jcol = bs*where;
			float *acol = &athread(who, icol, jcol);
			if(tid == who){
				#if 1
				sgetrf_nopiv_inner<bs>(n-k, acol, n);
				#endif
				myj += bs;
			}
			if(k+bs >= n) break;

			#pragma omp barrier

			float *aright = &al(k, myj);

			// process right part, strsm...,
			#if 1
			strsmLL_inner<bs>(myjend - myj, acol, aright, n);
			#endif


			const float *a_gemm = acol + bs;
			const float *b_gemm = aright;
			float       *c_gemm = aright + bs;

			const int m_gemm = n - (k + bs);
			const int n_gemm = myjend - myj;

			#if 1
			sgemm_inner<bs>(m_gemm, n_gemm, a_gemm, b_gemm, c_gemm, n);
			#endif
		}

		// copy out
		myjend = 0;
		for(int j=bs*tid; j<n; j+=bs*nth){
			for(int jj=0; jj<bs; jj++){
				if(j+jj >= n){
					j = n;
					break;
				}
				float * __restrict__ src = &al(0, myjend);
				float * __restrict__ dst = &ax(0, j+jj);
				for(int i=0; i<n; i++){
					dst[i] = src[i];
				}
				myjend++;
			}
		}
	}

	return;
}

} // anonymous namespace

extern void sgetrf_nopiv_tuned(int n, float *a, size_t lda){
	sgetrf_nopiv_omp<6>(n, a, lda);
}

#ifdef UNIT_TEST

#include <cmath>
#include <cstdlib>
#include <cstring>

#ifdef __aarch64__
static int64_t get_utime(){
	uint64_t tsc;
	asm volatile ("isb; mrs %0, cntvct_el0" : "=r" (tsc));
	return tsc;
}
static double tick2second(uint64_t tick){
	auto frequency = []{
		uint64_t frq;
		asm volatile ("isb; mrs %0, cntfrq_el0" : "=r" (frq));
		return frq;
	};
	static double invfreq = 1.0 / frequency();
	return invfreq * (double)tick;
}
#else
#  ifdef __APPLE__
#  include <sys/time.h>
static int64_t get_utime(){
	timeval tv;
	gettimeofday(&tv, NULL);

	return tv.tv_usec*1000ll + tv.tv_sec*1000000000ll;
}
#  elif defined __linux__
#  include <time.h>
static int64_t get_utime(){
	timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);

	return ts.tv_nsec + ts.tv_sec*1000000000ll;
}
#  else
#  error Timer API not available
#  endif
static double tick2second(uint64_t tick){
	return 1.e-9 * (double)tick;
}
#endif

int test_sgetrf(int n, float *a, size_t lda, float *b, size_t ldb){
	assert(n <= NMAX);

	double t0 = get_utime();
	sgetrf_nopiv_naive(n, a, lda);
	double t1 = get_utime();
	enum{BS=6,};
	sgetrf_nopiv_omp<BS>(n, b, ldb);
	double t2 = get_utime();

	int cnt = 0;
	for(int j=0; j<n; j++){
		for(int i=0; i<n; i++){
			float aa = a[i + lda*j];
			float bb = b[i + ldb*j];
			if(fabs(bb - aa) > 1.e-4){
				printf("! (%d, %d) : %f\t%f\t%e\n", i, j, aa, bb, bb-aa);
				cnt++;
			}
			if(cnt > 100) break;
		}
	}
	cnt ? printf("n=%d, FAIL\n", n) :  printf("n=%d, PASS\n", n);

	auto perf = [n](const char *name, int64_t tbeg, int64_t tend){
		double dt = tick2second(tend - tbeg);
		double flop = 2./3.* n * n * n;

		printf("%s: n = %d, %e usec, %f Gflops\n", name, n, 1.e6*dt, 1.e-9*flop/dt);
	};
	perf("naive", t0, t1);
	perf("tuned", t1, t2);

	return cnt;
}

int main(){
	enum{
		M = 1152,
		N = 1152,
		LDA = 1152 + 144,
		SEED = 42,
	};
	static float buf[LDA * N];
	static float abuf[LDA * N];
	static float bbuf[LDA * N];

	srand48(SEED);
	for(int j=0; j<N; j++){
		for(int i=0; i<M; i++){
			buf[i + LDA*j] = drand48() - 0.5;
			if (i == j) buf[i + LDA*j] = 10.0 * (drand48() + 1.0);
		}
	}

	int ret = 0;

	// dry run
	std::memcpy(abuf, buf, sizeof(buf));
	sgetrf_nopiv_omp<6>(288, abuf, LDA);

#if 1
	std::memcpy(abuf, buf, sizeof(buf));
	std::memcpy(bbuf, buf, sizeof(buf));
	ret += test_sgetrf(288, abuf, LDA, bbuf, LDA);


	std::memcpy(abuf, buf, sizeof(buf));
	std::memcpy(bbuf, buf, sizeof(buf));
	ret += test_sgetrf(320, abuf, LDA, bbuf, LDA);

	std::memcpy(abuf, buf, sizeof(buf));
	std::memcpy(bbuf, buf, sizeof(buf));
	ret += test_sgetrf(576, abuf, LDA, bbuf, LDA);

	std::memcpy(abuf, buf, sizeof(buf));
	std::memcpy(bbuf, buf, sizeof(buf));
	ret += test_sgetrf(640, abuf, LDA, bbuf, LDA);

	std::memcpy(abuf, buf, sizeof(buf));
	std::memcpy(bbuf, buf, sizeof(buf));
	ret += test_sgetrf(1152, abuf, LDA, bbuf, LDA);
#else
	for(int n=310; n<330; n++){
		std::memcpy(abuf, buf, sizeof(buf));
		std::memcpy(bbuf, buf, sizeof(buf));
		ret += test_sgetrf(n, abuf, LDA, bbuf, LDA);
	}
#endif

	puts("return");

	return ret;
}

#endif
