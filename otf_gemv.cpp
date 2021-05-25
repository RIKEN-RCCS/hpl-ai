#include <stdint.h>
#include "hpl_rand.hpp"
#include "svesim.hpp"

extern "C"
void otf_gemv_kernel(int64_t n, int mb, int nb, double alpha,
	double const* __restrict__ x, double* __restrict__ y, uint64_t seed)
{
	// on-the-fly GEMV computes y = y + alpha * A * x
	// see hpl_rand.hpp for details of the matrix generation. it is LCG.
	// n is the dimension of the whole matrix.
	// mb \times nb is the dimension of the sub-matrix to compute.
	// Note that the sub-matrix cannot have diagonals. Use serial code for the part includes diagonals instead.
	const int vlen = svcntd();
	const int nn = svcntd() * 4;

	RandCoeff c0 = RandCoeff::default_vals();
	RandStat s0; s0.x = seed;
	int64_t rinit[vlen];
	for(int i=0; i<vlen; ++i){
		rinit[i] = s0.x;
		s0 = c0 * s0;
	}
	RandCoeff c8 = c0.pow(vlen);
	RandCoeff c32 = c8.pow(4);
	RandCoeff cn = c0.pow(n);
	RandCoeff cn2 = cn.pow(2);


        auto t64 = svptrue_b64();
        svint64_t stat00 = svld1_s64(t64, rinit);
	alpha *= 0x1.fffffffffffffP-65;
	svfloat64_t sv0 = svdup_f64(0.);

	int64_t jend = nb-2;
        for(int64_t i=0; i<mb; i+=nn){
		__builtin_prefetch(&y[i]);
		svfloat64_t y00 = sv0;
		svfloat64_t y10 = sv0;
		svfloat64_t y20 = sv0;
		svfloat64_t y30 = sv0;
		svfloat64_t y01 = sv0;
		svfloat64_t y11 = sv0;
		svfloat64_t y21 = sv0;
		svfloat64_t y31 = sv0;
		svint64_t sc00, sc10, sc20, sc30, 
			sc01, sc11, sc21, sc31;
		{
			svint64_t a, c;
			a = svdup_s64(c8.a);
			c = svdup_s64(c8.c);
			sc00 = stat00;
			sc10 = svmad_s64_x(t64, sc00, a, c);
			sc20 = svmad_s64_x(t64, sc10, a, c);
			sc30 = svmad_s64_x(t64, sc20, a, c);
			a = svdup_s64(cn.a);
			c = svdup_s64(cn.c);
			sc01 = svmad_s64_x(t64, sc00, a, c);
			sc11 = svmad_s64_x(t64, sc10, a, c);
			sc21 = svmad_s64_x(t64, sc20, a, c);
			sc31 = svmad_s64_x(t64, sc30, a, c);
		}
		svint64_t sva, svc;
		sva = svdup_s64(cn2.a);
		svc = svdup_s64(cn2.c);
		for(int64_t j=0; j<=jend; j+=2){
			svfloat64_t svx1 = svdup_f64(x[j]);
			svfloat64_t r00 = svcvt_f64_s64_x(t64, sc00);
			svfloat64_t r10 = svcvt_f64_s64_x(t64, sc10);
			svfloat64_t r20 = svcvt_f64_s64_x(t64, sc20);
			svfloat64_t r30 = svcvt_f64_s64_x(t64, sc30);
			y00 = svmla_f64_x(t64, y00, r00, svx1);
			y10 = svmla_f64_x(t64, y10, r10, svx1);
			y20 = svmla_f64_x(t64, y20, r20, svx1);
			y30 = svmla_f64_x(t64, y30, r30, svx1);
			sc00 = svmad_s64_x(t64, sc00, sva, svc);
			sc10 = svmad_s64_x(t64, sc10, sva, svc);
			sc20 = svmad_s64_x(t64, sc20, sva, svc);
			sc30 = svmad_s64_x(t64, sc30, sva, svc);

			svfloat64_t svx2 = svdup_f64(x[j+1]);
			svfloat64_t r01 = svcvt_f64_s64_x(t64, sc01);
			svfloat64_t r11 = svcvt_f64_s64_x(t64, sc11);
			svfloat64_t r21 = svcvt_f64_s64_x(t64, sc21);
			svfloat64_t r31 = svcvt_f64_s64_x(t64, sc31);
			y01 = svmla_f64_x(t64, y01, r01, svx2);
			y11 = svmla_f64_x(t64, y11, r11, svx2);
			y21 = svmla_f64_x(t64, y21, r21, svx2);
			y31 = svmla_f64_x(t64, y31, r31, svx2);
			sc01 = svmad_s64_x(t64, sc01, sva, svc);
			sc11 = svmad_s64_x(t64, sc11, sva, svc);
			sc21 = svmad_s64_x(t64, sc21, sva, svc);
			sc31 = svmad_s64_x(t64, sc31, sva, svc);
		}
		if(__builtin_expect(!!(nb&0x1u), 0)){
			svfloat64_t r00 = svcvt_f64_s64_x(t64, sc00);
			svfloat64_t r10 = svcvt_f64_s64_x(t64, sc10);
			svfloat64_t r20 = svcvt_f64_s64_x(t64, sc20);
			svfloat64_t r30 = svcvt_f64_s64_x(t64, sc30);

			svfloat64_t svx = svdup_f64(x[nb-1]);
			y00 = svmla_f64_x(t64, y00, r00, svx);
			y10 = svmla_f64_x(t64, y10, r10, svx);
			y20 = svmla_f64_x(t64, y20, r20, svx);
			y30 = svmla_f64_x(t64, y30, r30, svx);
		}
                auto pg0 = svwhilelt_b64(i, (int64_t)(mb-0*vlen));
                auto pg1 = svwhilelt_b64(i, (int64_t)(mb-1*vlen));
                auto pg2 = svwhilelt_b64(i, (int64_t)(mb-2*vlen));
                auto pg3 = svwhilelt_b64(i, (int64_t)(mb-3*vlen));
		y00 = svadd_f64_x(t64, y00, y01);
		y10 = svadd_f64_x(t64, y10, y11);
		y20 = svadd_f64_x(t64, y20, y21);
		y30 = svadd_f64_x(t64, y30, y31);
		y00 = svmla_n_f64_x(pg0, svld1_vnum_f64(pg0, y+i, 0), y00, alpha);
		y10 = svmla_n_f64_x(pg1, svld1_vnum_f64(pg1, y+i, 1), y10, alpha);
		y20 = svmla_n_f64_x(pg2, svld1_vnum_f64(pg2, y+i, 2), y20, alpha);
		y30 = svmla_n_f64_x(pg3, svld1_vnum_f64(pg3, y+i, 3), y30, alpha);

		svst1_vnum_f64(pg0, y+i, 0, y00);
		svst1_vnum_f64(pg1, y+i, 1, y10);
		svst1_vnum_f64(pg2, y+i, 2, y20);
		svst1_vnum_f64(pg3, y+i, 3, y30);


		svint64_t sva32, svc32;
		sva32 = svdup_s64(c32.a);
		svc32 = svdup_s64(c32.c);
                stat00 = svmad_s64_x(pg0, stat00, sva32, svc32);
        }
}

extern "C"
void hmg_gemv_up(int istart, double a, double b, int mb, int nb, double alpha,
	double const* __restrict__ x, double* __restrict__ y)
{
	// same as abobe, but for the upper-triangular part of the Higham's hpl-ai matrix.
	// istart is the row position of the sub-matrix. jstart is not needed because of the structure of the matrix.
	// a and b are the paramter of the Higham's matrix.
	const int vlen = svcntd();
	const int nn = svcntd() * 4;
        auto t64 = svptrue_b64();
	double ab = a * b;

	svint64_t iindex0, iindex1, iindex2, iindex3;
	iindex0 = svindex_s64(istart, 1);
	iindex1 = svindex_s64(istart+vlen, 1);
	iindex2 = svindex_s64(istart+2*vlen, 1);
	iindex3 = svindex_s64(istart+3*vlen, 1);
	svfloat64_t findex0 = svcvt_f64_s64_x(t64, iindex0);
	svfloat64_t findex1 = svcvt_f64_s64_x(t64, iindex1);
	svfloat64_t findex2 = svcvt_f64_s64_x(t64, iindex2);
	svfloat64_t findex3 = svcvt_f64_s64_x(t64, iindex3);
	svfloat64_t svincr = svdup_f64((double)nn);
	svfloat64_t sva = svdup_f64(b);
	svfloat64_t svab = svdup_f64(ab);
	svfloat64_t sv0 = svdup_f64(0.);

	int64_t jend = nb-2;
        for(int64_t i=0; i<mb; i+=nn){
		__builtin_prefetch(&y[i]);
		svfloat64_t y00 = sv0;
		svfloat64_t y10 = sv0;
		svfloat64_t y20 = sv0;
		svfloat64_t y30 = sv0;
		svfloat64_t y01 = sv0;
		svfloat64_t y11 = sv0;
		svfloat64_t y21 = sv0;
		svfloat64_t y31 = sv0;
		// a+ab*(j+nn) = (a+ab*j) + ab*nn  is better for performance, but we recompute them for accuracy
		svfloat64_t ai0 = svmla_f64_x(t64, sva, svab, findex0);
		svfloat64_t ai1 = svmla_f64_x(t64, sva, svab, findex1);
		svfloat64_t ai2 = svmla_f64_x(t64, sva, svab, findex2);
		svfloat64_t ai3 = svmla_f64_x(t64, sva, svab, findex3);
		for(int64_t j=0; j<=jend; j+=2){
			svfloat64_t svx1 = svdup_f64(x[j]);
			y00 = svmla_f64_x(t64, y00, ai0, svx1);
			y10 = svmla_f64_x(t64, y10, ai1, svx1);
			y20 = svmla_f64_x(t64, y20, ai2, svx1);
			y30 = svmla_f64_x(t64, y30, ai3, svx1);

			svfloat64_t svx2 = svdup_f64(x[j+1]);
			y01 = svmla_f64_x(t64, y01, ai0, svx2);
			y11 = svmla_f64_x(t64, y11, ai1, svx2);
			y21 = svmla_f64_x(t64, y21, ai2, svx2);
			y31 = svmla_f64_x(t64, y31, ai3, svx2);
		}
		if(__builtin_expect(!!(nb&0x1u), 0)){
			svfloat64_t svx = svdup_f64(x[nb-1]);
			y00 = svmla_f64_x(t64, y00, ai0, svx);
			y10 = svmla_f64_x(t64, y10, ai1, svx);
			y20 = svmla_f64_x(t64, y20, ai2, svx);
			y30 = svmla_f64_x(t64, y30, ai3, svx);
		}
                auto pg0 = svwhilelt_b64(i, (int64_t)(mb-0*vlen));
                auto pg1 = svwhilelt_b64(i, (int64_t)(mb-1*vlen));
                auto pg2 = svwhilelt_b64(i, (int64_t)(mb-2*vlen));
                auto pg3 = svwhilelt_b64(i, (int64_t)(mb-3*vlen));
		y00 = svadd_f64_x(t64, y00, y01);
		y10 = svadd_f64_x(t64, y10, y11);
		y20 = svadd_f64_x(t64, y20, y21);
		y30 = svadd_f64_x(t64, y30, y31);
		y00 = svmla_n_f64_x(pg0, svld1_vnum_f64(pg0, y+i, 0), y00, alpha);
		y10 = svmla_n_f64_x(pg1, svld1_vnum_f64(pg1, y+i, 1), y10, alpha);
		y20 = svmla_n_f64_x(pg2, svld1_vnum_f64(pg2, y+i, 2), y20, alpha);
		y30 = svmla_n_f64_x(pg3, svld1_vnum_f64(pg3, y+i, 3), y30, alpha);

		svst1_vnum_f64(pg0, y+i, 0, y00);
		svst1_vnum_f64(pg1, y+i, 1, y10);
		svst1_vnum_f64(pg2, y+i, 2, y20);
		svst1_vnum_f64(pg3, y+i, 3, y30);


		findex0 = svadd_f64_x(t64, findex0, svincr);
		findex1 = svadd_f64_x(t64, findex1, svincr);
		findex2 = svadd_f64_x(t64, findex2, svincr);
		findex3 = svadd_f64_x(t64, findex3, svincr);
        }
}

extern "C"
void hmg_gemv_low(int istart, double a, double b, int mb, int nb, double alpha,
	double const* __restrict__ x, double* __restrict__ y)
{
	// same as above, but for lower-triangular part.
	const int vlen = svcntd();
	const int nn = svcntd() * 4;
        auto t64 = svptrue_b64();
	double ab = a * b;

	svfloat64_t svincr = svdup_f64((double)2);
	svfloat64_t sva = svdup_f64(a);
	svfloat64_t svab = svdup_f64(ab);
	svfloat64_t sv0 = svdup_f64(0.);

	int64_t jend = nb-2;
        for(int64_t i=0; i<mb; i+=nn){
		svfloat64_t findex0 = svdup_f64((double)istart);
		svfloat64_t findex1 = svdup_f64((double)(istart+1));
		__builtin_prefetch(&y[i]);
		svfloat64_t y00 = sv0;
		svfloat64_t y10 = sv0;
		svfloat64_t y20 = sv0;
		svfloat64_t y30 = sv0;
		svfloat64_t y01 = sv0;
		svfloat64_t y11 = sv0;
		svfloat64_t y21 = sv0;
		svfloat64_t y31 = sv0;
		for(int64_t j=0; j<=jend; j+=2){
			// a+ab*(j+nn) = (a+ab*j) + ab*nn  is better for performance, but we recompute them for accuracy
			svfloat64_t svx1 = svdup_f64(x[j]);
			svfloat64_t aj0 = svmla_f64_x(t64, sva, svab, findex0);
			svfloat64_t aj1 = svmla_f64_x(t64, sva, svab, findex1);
			y00 = svmla_f64_x(t64, y00, aj0, svx1);
			y10 = svmla_f64_x(t64, y10, aj0, svx1);
			y20 = svmla_f64_x(t64, y20, aj0, svx1);
			y30 = svmla_f64_x(t64, y30, aj0, svx1);

			svfloat64_t svx2 = svdup_f64(x[j+1]);
			y01 = svmla_f64_x(t64, y01, aj1, svx2);
			y11 = svmla_f64_x(t64, y11, aj1, svx2);
			y21 = svmla_f64_x(t64, y21, aj1, svx2);
			y31 = svmla_f64_x(t64, y31, aj1, svx2);
			findex0 = svadd_f64_x(t64, findex0, svincr);
			findex1 = svadd_f64_x(t64, findex1, svincr);
		}
		if(__builtin_expect(!!(nb&0x1u), 0)){
			svfloat64_t svx = svdup_f64(x[nb-1]);
			svfloat64_t aj0 = svmla_f64_x(t64, sva, svab, findex0);
			y00 = svmla_f64_x(t64, y00, aj0, svx);
			y10 = svmla_f64_x(t64, y10, aj0, svx);
			y20 = svmla_f64_x(t64, y20, aj0, svx);
			y30 = svmla_f64_x(t64, y30, aj0, svx);
		}
                auto pg0 = svwhilelt_b64(i, (int64_t)(mb-0*vlen));
                auto pg1 = svwhilelt_b64(i, (int64_t)(mb-1*vlen));
                auto pg2 = svwhilelt_b64(i, (int64_t)(mb-2*vlen));
                auto pg3 = svwhilelt_b64(i, (int64_t)(mb-3*vlen));
		y00 = svadd_f64_x(t64, y00, y01);
		y10 = svadd_f64_x(t64, y10, y11);
		y20 = svadd_f64_x(t64, y20, y21);
		y30 = svadd_f64_x(t64, y30, y31);
		y00 = svmla_n_f64_x(pg0, svld1_vnum_f64(pg0, y+i, 0), y00, alpha);
		y10 = svmla_n_f64_x(pg1, svld1_vnum_f64(pg1, y+i, 1), y10, alpha);
		y20 = svmla_n_f64_x(pg2, svld1_vnum_f64(pg2, y+i, 2), y20, alpha);
		y30 = svmla_n_f64_x(pg3, svld1_vnum_f64(pg3, y+i, 3), y30, alpha);

		svst1_vnum_f64(pg0, y+i, 0, y00);
		svst1_vnum_f64(pg1, y+i, 1, y10);
		svst1_vnum_f64(pg2, y+i, 2, y20);
		svst1_vnum_f64(pg3, y+i, 3, y30);
        }
}

extern "C" 
void hmg_gemv_diag(int istart, int jstart, double a, double b, int mb, int nb, double alpha,
	double const* __restrict__ x, double* __restrict__ y)
{
	// same as above, but for the sub-matrix whic includes diagonals.
	double ab = a * b;
	for(int i=0; i<mb; ++i){
		double d = 0.;
		for(int j=0; j<i; ++j){
			double aj = b + ab * (jstart + j);
			d += aj * x[j];
		}
		d += (1. + ab*(istart + i)) * x[i];
		double ai = a + ab * (istart + i);
		for(int j=i+1; j<nb; ++j){
			d += ai * x[j];
		}
		y[i] += alpha * d;
	}
}
