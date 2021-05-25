#include <x86intrin.h>
#include "kernel.h"
#include "../fp16sim.hpp"
#include <stdio.h>

#define RI 16
#define RJ 4
extern "C"
void hgemmpp_kernel(int64_t m, int64_t nb, int64_t k, fp16 const* __restrict__ a, fp16 const* __restrict__ b, fp16* __restrict__ c, int64_t ldc)
{
	// very slow. only for test
	for(int j=0; j<nb; ++j, b+=RJ*k, c+=RJ*ldc){
		for(int i=0; i<m; i+=RI){
			fp16 t[RJ][RI];
			for(int jj=0; jj<RJ; ++jj) for(int ii=0; ii<RI; ++ii) t[jj][ii] = 0.f;
			for(int kk=0; kk<k; ++kk)
				for(int jj=0; jj<RJ; ++jj)
					for(int ii=0; ii<RI; ++ii)
						t[jj][ii] = fp16_fma(a[i*k+kk*RI+ii], b[kk*RJ+jj], t[jj][ii]);
			for(int jj=0; jj<RJ; ++jj)
				for(int ii=0; ii<RI && ii<m-i; ++ii)
					c[jj*ldc+i+ii] = (c[jj*ldc+i+ii].convert_to_float() - t[jj][ii].convert_to_float());
		}
	}
}

extern "C"
void hgemmpp_mnend(int64_t m, int64_t n, int64_t k, fp16 const* __restrict__ a, fp16 const* __restrict__ b, fp16* __restrict__ c, int64_t ldc)
{
	for(int j=0; j<n; ++j){
		fp16 t[RI];
		for(int i=0; i<RI; ++i) t[i] = 0.f;
		for(int kk=0; kk<k; ++kk)
			for(int i=0; i<RI; ++i) t[i] = fp16_fma(a[kk*RI+i], b[kk*RJ+j], t[i]);
		for(int i=0; i<RI && i < m; ++i) c[j*ldc+i] = (c[j*ldc+i].convert_to_float() - t[i].convert_to_float());
	}
}

extern "C"
void pack_convert_a_opt(int64_t m, int64_t k, float alpha, float const* a, int64_t lda, fp16* to)
{
	for(int64_t i=0; i<m; i+=RI){
		if(m-i>=RI){
			for(int64_t j=0; j<k; ++j){
				for(int64_t ii=0; ii<RI; ++ii){
					to[i*k+RI*j+ii] = alpha*a[j*lda+i+ii];
				}
			}
		}
		else {
			for(int64_t j=0; j<k; ++j){
				for(int64_t ii=0; ii<RI; ++ii){
					to[i*k+RI*j+ii] = (i+ii<m?alpha*a[j*lda+i+ii]: 0.f);
				}
			}
		}
	}

}

extern "C"
void pack_convert_b_opt(int64_t n, int64_t k, float alpha, float const* b, int64_t ldb, fp16* to)
{
	for(int64_t j=0; j<n; j+=RJ){
		for(int64_t i=0; i<k; ++i)
			for(int64_t jj=0; jj<RJ; ++jj)
				to[j*k+RJ*i+jj] = alpha*b[(j+jj)*ldb+i];
	}
}
extern "C"
void pack_convert_b_small(int64_t n, int64_t k, float alpha, float const* b, int64_t ldb, fp16* to)
{
	for(int64_t i=0; i<k; ++i)
		for(int64_t jj=0; jj<RJ; ++jj)
			to[RJ*i+jj] = (jj<n? alpha*b[jj*ldb+i]: 0.f);
}
#undef RI
#undef RJ
#define RI 16
#define RJ 4

static void hgemm_opt_kernel(int ksize, float alpha, float const* __restrict__ a, float const* __restrict__ b, fp16* __restrict__ c, int ldc)
{
	__m128i c00 = _mm_set1_epi32(0);
	__m128i c10 = _mm_set1_epi32(0);
	__m128i c01 = _mm_set1_epi32(0);
	__m128i c11 = _mm_set1_epi32(0);
	__m128i c02 = _mm_set1_epi32(0);
	__m128i c12 = _mm_set1_epi32(0);
	__m128i c03 = _mm_set1_epi32(0);
	__m128i c13 = _mm_set1_epi32(0);
	for(int kk=0; kk<ksize; ++kk){
		__m256 a0 = _mm256_loadu_ps(a+RI*kk);
		__m256 a1 = _mm256_loadu_ps(a+RI*kk+8);
		__m256 b0 = _mm256_broadcast_ss(b+RJ*kk);
		c00 = _mm256_cvtps_ph(_mm256_fmadd_ps(a0, b0, _mm256_cvtph_ps(c00)), 0);
		c10 = _mm256_cvtps_ph(_mm256_fmadd_ps(a1, b0, _mm256_cvtph_ps(c10)), 0);
		b0 = _mm256_broadcast_ss(b+RJ*kk+1);
		c01 = _mm256_cvtps_ph(_mm256_fmadd_ps(a0, b0, _mm256_cvtph_ps(c01)), 0);
		c11 = _mm256_cvtps_ph(_mm256_fmadd_ps(a1, b0, _mm256_cvtph_ps(c11)), 0);
		b0 = _mm256_broadcast_ss(b+RJ*kk+2);
		c02 = _mm256_cvtps_ph(_mm256_fmadd_ps(a0, b0, _mm256_cvtph_ps(c02)), 0);
		c12 = _mm256_cvtps_ph(_mm256_fmadd_ps(a1, b0, _mm256_cvtph_ps(c12)), 0);
		b0 = _mm256_broadcast_ss(b+RJ*kk+3);
		c03 = _mm256_cvtps_ph(_mm256_fmadd_ps(a0, b0, _mm256_cvtph_ps(c03)), 0);
		c13 = _mm256_cvtps_ph(_mm256_fmadd_ps(a1, b0, _mm256_cvtph_ps(c13)), 0);
	}
	__m256 mmalpha = _mm256_set1_ps(alpha);
	#define cupdate(i, j)\
		_mm_storeu_si128(reinterpret_cast<__m128i*>(c+j*ldc+8*i),\
			_mm256_cvtps_ph(\
				_mm256_fmadd_ps(mmalpha, _mm256_cvtph_ps(c##i##j), \
					_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<__m128i*>(c+j*ldc+8*i)))), 0))
	cupdate(0, 0);
	cupdate(1, 0);
	cupdate(0, 1);
	cupdate(1, 1);
	cupdate(0, 2);
	cupdate(1, 2);
	cupdate(0, 3);
	cupdate(1, 3);
	#undef cupdate

}

void hgemm_opt(int m, int n, int k, float alpha, fp16 const* a, int lda, fp16 const* b, int ldb, float /*beta*/, fp16*c, int ldc)
{
	#if 0
        for(int i=0; i<n; ++i){
                for(int j=0; j<m; ++j){
			fp16 temp(0.f);
                        for(int l=0; l<k; ++l)
                                temp = fp16_fma(a[l*lda+j], b[i*ldb+l], temp);
                        c[ldc*i+j] = (c[ldc*i+j].convert_to_float() * 1.f + temp.convert_to_float()*alpha);
                }
        }
	#else
	// beta == 1.f
	int const ri = RI;
	int const rj = RJ;
	int const bk = 100;
	int const bi = ri * 30;
	int const bj = rj * 30;
	float bufa[ri*bk];
	float bufb[bj*bk];
	for(int l=0; l<k; l+=bk){
		int i;
		for(i=0; i+ri<=m; i+=bi){
			int j;
			for(j=0; j+rj<=n; j+=bj){
				for(int kk=l; kk<k&&kk<l+bk; ++kk){
					for(int jj=j; jj+rj<=n && jj<j+bj; jj+=rj){
						bufb[bk*(jj-j)+rj*(kk-l)] = static_cast<float>(b[jj*ldb+kk]);
						bufb[bk*(jj-j)+rj*(kk-l)+1] = static_cast<float>(b[(jj+1)*ldb+kk]);
						bufb[bk*(jj-j)+rj*(kk-l)+2] = static_cast<float>(b[(jj+2)*ldb+kk]);
						bufb[bk*(jj-j)+rj*(kk-l)+3] = static_cast<float>(b[(jj+3)*ldb+kk]);
					}
				}
				for(int ii=i; ii+ri<=m&&ii<i+bi; ii+=ri){
					for(int kk=l; kk<k&&kk<l+bk; ++kk){
						_mm256_storeu_ps(bufa+ri*(kk-l), _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<__m128i const*>(a+kk*lda+ii))));
						_mm256_storeu_ps(bufa+ri*(kk-l)+8, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<__m128i const*>(a+kk*lda+ii+8))));
					}
					for(int jj=j; jj+rj<=n && jj<j+bj; jj+=rj){
						hgemm_opt_kernel(k-l<bk?k-l:bk, alpha, bufa, bufb+bk*(jj-j), c+jj*ldc+ii, ldc);
					}
				}
			}
			for(int jj=n/rj*rj; jj<n; ++jj){
				for(int ii=i; ii<(m-i)/ri*ri&&ii<i+bi; ++ii){
					fp16 temp(0.f);
					for(int kk=l; kk<l+bk&&kk<k; ++kk)
						temp = fp16_fma(a[kk*lda+ii], b[jj*ldb+kk], temp);
					c[ldc*jj+ii] = (c[ldc*jj+ii].convert_to_float() + temp.convert_to_float()*alpha);
				}

			}
		}
		for(int j=0; j<n; ++j){
			for(int ii=(m/ri*ri); ii<m; ++ii){
				fp16 temp(0.f);
				for(int kk=l; kk<l+bk&&kk<k; ++kk)
					temp = fp16_fma(a[kk*lda+ii], b[j*ldb+kk], temp);
				c[ldc*j+ii] = (c[ldc*j+ii].convert_to_float() + temp.convert_to_float()*alpha);
			}
		}
	}
	#endif
}
#undef RI
#undef RJ

#define RI 24
#define RJ 4
static void sgemm_kernel(int ksize, float alpha, float const* __restrict__ a, float const* __restrict__ b, float * __restrict__ c, int ldc)
{
	__m256 c00 = _mm256_setzero_ps();
	__m256 c10 = _mm256_setzero_ps();
	__m256 c20 = _mm256_setzero_ps();
	__m256 c01 = _mm256_setzero_ps();
	__m256 c11 = _mm256_setzero_ps();
	__m256 c21 = _mm256_setzero_ps();
	__m256 c02 = _mm256_setzero_ps();
	__m256 c12 = _mm256_setzero_ps();
	__m256 c22 = _mm256_setzero_ps();
	__m256 c03 = _mm256_setzero_ps();
	__m256 c13 = _mm256_setzero_ps();
	__m256 c23 = _mm256_setzero_ps();
	for(int kk=0; kk<ksize; ++kk){
		__m256 a0 = _mm256_loadu_ps(a+RI*kk);
		__m256 a1 = _mm256_loadu_ps(a+RI*kk+8);
		__m256 a2 = _mm256_loadu_ps(a+RI*kk+16);
		__m256 b0 = _mm256_broadcast_ss(b+RJ*kk);
		c00 = _mm256_fmadd_ps(a0, b0, c00);
		c10 = _mm256_fmadd_ps(a1, b0, c10);
		c20 = _mm256_fmadd_ps(a2, b0, c20);
		b0 = _mm256_broadcast_ss(b+RJ*kk+1);
		c01 = _mm256_fmadd_ps(a0, b0, c01);
		c11 = _mm256_fmadd_ps(a1, b0, c11);
		c21 = _mm256_fmadd_ps(a2, b0, c21);
		b0 = _mm256_broadcast_ss(b+RJ*kk+2);
		c02 = _mm256_fmadd_ps(a0, b0, c02);
		c12 = _mm256_fmadd_ps(a1, b0, c12);
		c22 = _mm256_fmadd_ps(a2, b0, c22);
		b0 = _mm256_broadcast_ss(b+RJ*kk+3);
		c03 = _mm256_fmadd_ps(a0, b0, c03);
		c13 = _mm256_fmadd_ps(a1, b0, c13);
		c23 = _mm256_fmadd_ps(a2, b0, c23);
	}
	__m256 mmalpha = _mm256_set1_ps(alpha);
	#define cupdate(i, j)\
		_mm256_storeu_ps(c+j*ldc+8*i, _mm256_fmadd_ps(mmalpha, c##i##j,  _mm256_loadu_ps(c+j*ldc+8*i)))
	cupdate(0, 0);
	cupdate(1, 0);
	cupdate(2, 0);
	cupdate(0, 1);
	cupdate(1, 1);
	cupdate(2, 1);
	cupdate(0, 2);
	cupdate(1, 2);
	cupdate(2, 2);
	cupdate(0, 3);
	cupdate(1, 3);
	cupdate(2, 3);
	#undef cupdate

}

void shgemm_opt(int m, int n, int k, float alpha, fp16 const* a, int lda, fp16 const* b, int ldb, float /*beta*/, float*c, int ldc)
{
	// beta == 1.f
	int const ri = RI;
	int const rj = RJ;
	int const bk = 100;
	int const bi = ri * 30;
	int const bj = rj * 30;
	float bufa[ri*bk];
	float bufb[bj*bk];
	for(int l=0; l<k; l+=bk){
		int i;
		for(i=0; i+ri<=m; i+=bi){
			int j;
			for(j=0; j+rj<=n; j+=bj){
				for(int kk=l; kk<k&&kk<l+bk; ++kk){
					for(int jj=j; jj+rj<=n && jj<j+bj; jj+=rj){
						bufb[bk*(jj-j)+rj*(kk-l)] = static_cast<float>(b[jj*ldb+kk]);
						bufb[bk*(jj-j)+rj*(kk-l)+1] = static_cast<float>(b[(jj+1)*ldb+kk]);
						bufb[bk*(jj-j)+rj*(kk-l)+2] = static_cast<float>(b[(jj+2)*ldb+kk]);
						bufb[bk*(jj-j)+rj*(kk-l)+3] = static_cast<float>(b[(jj+3)*ldb+kk]);
					}
				}
				for(int ii=i; ii+ri<=m&&ii<i+bi; ii+=ri){
					for(int kk=l; kk<k&&kk<l+bk; ++kk){
						_mm256_storeu_ps(bufa+ri*(kk-l), _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<__m128i const*>(a+kk*lda+ii))));
						_mm256_storeu_ps(bufa+ri*(kk-l)+8, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<__m128i const*>(a+kk*lda+ii+8))));
						_mm256_storeu_ps(bufa+ri*(kk-l)+16, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<__m128i const*>(a+kk*lda+ii+16))));
					}
					for(int jj=j; jj+rj<=n && jj<j+bj; jj+=rj){
						sgemm_kernel(k-l<bk?k-l:bk, alpha, bufa, bufb+bk*(jj-j), c+ldc*jj+ii, ldc);
					}
				}
			}
			for(int jj=n/rj*rj; jj<n; ++jj){
				for(int ii=i; ii<(m-i)/ri*ri&&ii<i+bi; ++ii){
					float temp = 0.f;
					for(int kk=l; kk<l+bk&&kk<k; ++kk)
						temp = a[kk*lda+ii].convert_to_float() * b[jj*ldb+kk].convert_to_float() + temp;
					c[ldc*jj+ii] = c[ldc*jj+ii] + temp*alpha;
				}

			}
		}
		for(int j=0; j<n; ++j){
			for(int ii=(m/ri*ri); ii<m; ++ii){
				float temp = 0.f;
				for(int kk=l; kk<l+bk&&kk<k; ++kk)
					temp = a[kk*lda+ii].convert_to_float() * b[j*ldb+kk].convert_to_float() + temp;
				c[ldc*j+ii] = c[ldc*j+ii] + temp*alpha;
			}
		}
	}
}
#undef RI
#undef RJ

//#define UNIT_TEST
#ifdef UNIT_TEST
#include <stdlib.h>
#include <math.h>
void pack_convert_a(int m, int k, float alpha, float const* a, int64_t lda, fp16* to)
{
	pack_convert_a_opt(m, k, alpha, a, lda, to);
}
void pack_convert_b(int n, int k, float alpha, float const* b, int64_t ldb, fp16* work)
{
	pack_convert_b_opt(n, k, alpha, b, ldb, work);
	if(n%4){
		int nr = n%4;
		int nlast = n - nr;
		pack_convert_b_small(nr, k, alpha, b+nlast*ldb, ldb, work+nlast*k);

	}
}
void gemm_simple(int m, int n, int b, fp16 * pa, fp16* pb, fp16* pc, int ldc)
{
	int nlast = n/4*4;
	if(nlast) hgemmpp_kernel(m, nlast/4, b, pa, pb, pc, ldc);
	if(n-nlast) {
		for(int i=0; i<m; i+=16){
			int msize2 = m-i>=16?16: m-i;
			hgemmpp_mnend(msize2, n-nlast, b, pa+i*b,
				pb+nlast*b, pc+nlast*ldc+i, ldc);
		}
	}
}
void hgemm_naiive(int m, int n, int k, float alpha, fp16 const* a, int lda, fp16 const* b, int ldb, float beta, fp16*c, int ldc)
{
	if(beta==1.f){ // remove this in the case for test
		hgemm_opt(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
		return;
	}
        for(int i=0; i<n; ++i){
                for(int j=0; j<m; ++j){
			fp16 temp(0.f);
                        for(int l=0; l<k; ++l)
                                temp = fp16_fma(a[l*lda+j], b[i*ldb+l], temp);
                        c[ldc*i+j] = (c[ldc*i+j].convert_to_float() * beta + temp.convert_to_float()*alpha);
                }
        }
}
int main()
{
	int m = 123;
	int n = 111;
	int k = 10;

	float* a = (float*)malloc(sizeof(float)*m*k);
	float* b = (float*)malloc(sizeof(float)*n*k);
	fp16* pa = (fp16*)malloc(sizeof(fp16)*(m+10)*k);
	fp16* pb = (fp16*)malloc(sizeof(fp16)*(n+10)*k);
	fp16* c = (fp16*)malloc(sizeof(fp16)*m*n);
	fp16* c2 = (fp16*)malloc(sizeof(fp16)*m*n);

	for(int i=0; i<m; ++i) for(int j=0; j<k; ++j) pa[j*m+i] = a[j*m+i] = 0.001f*((i*521+j*211) % 1297);
	for(int i=0; i<n; ++i) for(int j=0; j<k; ++j) pb[i*k+j] = b[i*k+j] = 0.001f*((i*227+j*401) % 1193);
	for(int i=0; i<n; ++i) for(int j=0; j<m; ++j) c[i*m+j] = c2[i*m+j] = 1.f;

	hgemm_naiive(m, n, k, -1.f, pa, m, pb, k, 1.f, c, m);
	pack_convert_a(m, k, 1.f, a, m, pa);
	printf("aaaa\n");
	for(int i=0; i<k; ++i) for(int j=0; j<m; ++j){
		int t = (float)pa[j/16*16*k+(j%16)+16*i];
		int s = a[i*m+j];
		if(t!=s) printf("%3d %3d t=%3d, s=%3d\n", j, i, t, s);
	}
	printf("bbbb\n");
	pack_convert_b(n, k, 1.f, b, k, pb);
	for(int i=0; i<n; ++i) for(int j=0; j<k; ++j){
		int t = (float)pb[i/4*4*k+(i%4)+4*j];
		int s = b[i*k+j];
		if(t!=s) printf("%3d %3d t=%3d, s=%3d\n", j, i, t, s);
	}
	gemm_simple(m, n, k, pa, pb, c2, m);
	printf("mmmm\n");
	for(int i=0; i<n; ++i) for(int j=0; j<m; ++j){
		float d = (float)c[i*m+j] - (float)c2[i*m+j];
		d = fabsf(d) / fabsf((float)c[i*m+j]);
		printf("%3d %3d %e %e %e\n", j, i, (float)c[i*m+j], (float)c2[i*m+j], d);
		
	}
	return 0;

}

#endif
