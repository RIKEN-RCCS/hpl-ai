#ifndef GETRF_NOPIV
#define GETRF_NOPIV
#include <cstdlib>
#include <omp.h>
#include "schur_updator.hpp"

extern "C" void dtrsm_(...);
extern "C" void strsm_(...);
inline void trsmR(int m, int n, const double *a, int lda, double *b, int ldb)
{
	double one = 1.;
	dtrsm_("R", "U", "N", "N", &m, &n, &one, a, &lda, b, &ldb);
}
inline void trsmR(int m, int n, const float *a, int lda, float *b, int ldb)
{
	float one = 1.f;
	strsm_("R", "U", "N", "N", &m, &n, &one, a, &lda, b, &ldb);
}
inline void trsmL(int m, int n, double const*a, int lda, double *b, int ldb)
{
	double one = 1.;
	dtrsm_("L", "L", "N", "U", &m, &n, &one, a, &lda, b, &ldb);
}
inline void trsmL(int m, int n, float const*a, int lda, float *b, int ldb)
{
	float one = 1.f;
	strsm_("L", "L", "N", "U", &m, &n, &one, a, &lda, b, &ldb);
}

#define NSMALL 16
template<typename F>
void getrf_nopiv_small(int n, F* a, size_t lda) {
	for(int k=0; k<n; k++){
		F inv = static_cast<F>(1) / a[lda*k + k];
		for(int i=k+1; i<n; ++i) a[k*lda + i] *= inv;
		#pragma loop novrec
		#pragma loop unroll(2)
		for(int j=k+1; j<n; ++j){
			F akj = a[j*lda + k];
			#pragma loop novrec
			for(int i=k+1; i<n; ++i){
				a[j*lda + i] -= a[k*lda + i] * akj;
			}
		}
	}
}

#if 0
template<typename F>
void gemm_small(int n, int m, int lda, F const*__restrict__ a, F const*__restrict__ b, F*__restrict__ c)
{
	for(int j=0; j<m; ++j){
		for(int k=0; k<m; ++k){
			for(int i=0; i<n; ++i) c[j*lda+i] -= a[k*lda+i] * b[j*lda+k];
		}
	}
}
void gemm_small(int n, int m, int lda, float const*__restrict__ a, float const*__restrict__ b, float*__restrict__ c)
{
	float one = 1.f, mone=-1.f;
	sgemm_("N", "N", &n, &m, &m, &mone, a, &lda, b, &lda, &one, c, &lda);
}

template<typename F>
void getrf_nopiv(int n, F* a, size_t lda) {
	int const b = NSMALL;
	#pragma omp parallel
	{
		int nt = omp_get_num_threads();
		int id = omp_get_thread_num();
		int who = 0;
		int jstart = b*id;
		int jstride = b*nt;
		for(int k=0; k<n; k+=b, ++who){
			if(who==nt) who = 0;
			if(jstart <= k) jstart += jstride;

			F* apiv = a+lda*k+k;
			if(n-k<b){
				if(id==who){
					int nn = n-k<b? n-k: b;
					getrf_nopiv_small(nn, apiv, lda);
				}
				break;
			}

			F* lpanel = apiv+b;
			int lsize = n-k-b;
			if(id==who){
				getrf_nopiv_small(b, apiv, lda);
				trsmR(lsize, b, apiv, lda, lpanel, lda);
			}
			#pragma omp barrier

			for(int j=jstart; j<n; j+=jstride){
				F* rpanel = a+lda*j+k;
				F* ablock  = rpanel + b;
				int nn = n-j<b ? n-j: b;
				trsmL(b, nn, apiv, lda, rpanel, lda);
				gemm_small(lsize, nn, lda, lpanel, rpanel, ablock);
			}

		}
	}
}
#else
template<typename F>
void getrf_nopiv(int n, F* a, size_t lda, bool warmup=false) {
	for(int k=0; k<n; k+=NSMALL){
		int nn = n-k<NSMALL? n-k: NSMALL;
		getrf_nopiv_small(nn, a+lda*k+k, lda);
		if(n-k>nn){
			trsmL(nn, n-k-nn, a+lda*k+k, lda, a+lda*(k+nn)+k, lda);
			trsmR(n-k-nn, nn, a+lda*k+k, lda, a+lda*k+k+nn, lda);
			gemmschur(n-k-nn, n-k-nn, nn, a+lda*k+k+nn, lda, a+lda*(k+nn)+k, lda, a+lda*(k+nn)+(k+nn), lda);
		}
	}
}
#endif

#if defined(__FUJITSU) || defined(__CLANG_FUJITSU)
extern void sgetrf_nopiv_tuned(int n, float *a, size_t lda);
#endif
template<>
void getrf_nopiv<float>(int n, float* a, size_t lda, bool warmup) {
#if defined(__FUJITSU) || defined(__CLANG_FUJITSU)
	if(!warmup){
		sgetrf_nopiv_tuned(n, a, lda);
		return;
	}
#endif
	for(int k=0; k<n; k+=NSMALL){
		int nn = n-k<NSMALL? n-k: NSMALL;
		getrf_nopiv_small(nn, a+lda*k+k, lda);
		if(n-k>nn){
			trsmL(nn, n-k-nn, a+lda*k+k, lda, a+lda*(k+nn)+k, lda);
			trsmR(n-k-nn, nn, a+lda*k+k, lda, a+lda*k+k+nn, lda);
			gemmschur(n-k-nn, n-k-nn, nn, a+lda*k+k+nn, lda, a+lda*(k+nn)+k, lda, a+lda*(k+nn)+(k+nn), lda);
		}
	}
}

// copy-first version
template<typename F>
void getrf_nopiv(int n, F* a, size_t lda, F* piv, int ldpiv) {
		#pragma omp parallel for
		for(int j=0; j<n; ++j)
#if 0
			for(int i=0; i<n; ++i)
				piv[j*ldpiv + i] = a[j*lda + i];
#else
		{
			const F * __restrict__ src = &a[j*lda];
			F * __restrict__ dst = &piv[j*ldpiv];
			for(int i=0; i<n; ++i) dst[i] = src[i];
		}
#endif

		getrf_nopiv(n, piv, ldpiv);
}
template<typename F>
void warmup_trf(int n, F* a, size_t lda){
	for(int k=0; k<n; k++){
		a[k + lda * k] = 1.0;
	}
	getrf_nopiv(n, a, lda, true);
}
#endif

