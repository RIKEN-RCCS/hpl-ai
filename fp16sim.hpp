#ifndef FP16SIM_HPP
#define FP16SIM_HPP
// a very small wrapper for fp16.

//#define BF_NMANT 3

#ifdef __aarch64__
#  if !defined(__FUJITSU) && !defined(__CLANG_FUJITSU)
//#    define FP16_NATIVE_SUPPORT
#    define FP16_AUTO_PROMOTION
#  else
#    define FP16_FUJITSU_TRAD_MODE
#  endif
#elif defined(BF_NMANT)
#define FP16_BFLIKE_FLOAT
#if BF_NMANT>7 || BF_NMANT<=1
#error "too large or small mantissa for BFLIKE_FLOAT"
#endif
#elif defined(__AVX2__)
#define FP16_AVX2_EMULATION
#elif defined(__clang__) && __clang_major__ >= 8
#define FP16_AUTO_PROMOTION
#else
#define FP16_IS_NOT_SUPPORTED
#endif

#ifdef FP16_NATIVE_SUPPORT
typedef _Float16 fp16;

inline void hgemm(int m, int n, int k, float alpha, fp16 const* a, int lda, fp16 const* b, int ldb, float beta, fp16*c, int ldc)
{
	// HGEMM
	// replace with native one for performance. 
        for(int i=0; i<n; ++i){
                for(int j=0; j<m; ++j){
			fp16 temp(0.f);
                        for(int l=0; l<k; ++l)
                                temp = a[l*lda+j] * b[i*ldb+l] + temp;
                        c[ldc*i+j] = c[ldc*i+j] * beta + temp * alpha;
                }
        }
}
inline void shgemm(int m, int n, int k, float alpha, fp16 const* a, int lda, fp16 const* b, int ldb, float beta, float *c, int ldc)
{
	// SHGEMM. HGEMM with fp32 accumulator.
	// replace with native one for performance. 
        for(int i=0; i<n; ++i){
                for(int j=0; j<m; ++j){
			float temp = 0.f;
                        for(int l=0; l<k; ++l)
                                temp = a[l*lda+j] *  b[i*ldb+l] + temp;
                        c[ldc*i+j] = c[ldc*i+j] * beta + temp * alpha;
                }
        }
}
#endif

#ifdef FP16_FUJITSU_TRAD_MODE
// and CLANG mode
#include <stdlib.h>
extern "C" void fjblas_gemm_r16_(...);
typedef __fp16 fp16;
inline void hgemm(int m, int n, int k, float alpha, fp16 const* a, int lda, fp16 const* b, int ldb, float beta, fp16*c, int ldc){
	(void)alpha;
	(void)beta;
	short one = 15360; // == 1.
	short mone = -17408; // == -1.
	fjblas_gemm_r16_("N", "N", &m, &n, &k, &mone, a, &lda, b, &ldb, &one, c, &ldc);
}
inline void shgemm(int, int, int, float, fp16 const*, int, fp16 const*, int, float, float *, int){
	abort();
}
#endif

#ifdef FP16_AUTO_PROMOTION
typedef __fp16 fp16;

inline void hgemm(int m, int n, int k, float alpha, fp16 const* a, int lda, fp16 const* b, int ldb, float beta, fp16*c, int ldc)
{
        for(int i=0; i<n; ++i){
                for(int j=0; j<m; ++j){
			fp16 temp(0);
                        for(int l=0; l<k; ++l)
                                temp = a[l*lda+j] * b[i*ldb+l] + temp;
                        c[ldc*i+j] = c[ldc*i+j]* beta + temp*alpha;
                }
        }
}
inline void shgemm(int m, int n, int k, float alpha, fp16 const* a, int lda, fp16 const* b, int ldb, float beta, float *c, int ldc)
{
        for(int i=0; i<n; ++i){
                for(int j=0; j<m; ++j){
			float temp = 0.f;
                        for(int l=0; l<k; ++l)
                                temp = a[l*lda+j] *  b[i*ldb+l] + temp;
                        c[ldc*i+j] = c[ldc*i+j] * beta + temp*alpha;
                }
        }
}
#endif

#if defined(FP16_AVX2_EMULATION) || defined(FP16_BFLIKE_FLOAT)
#ifdef FP16_AVX2_EMULATION
#include <x86intrin.h>
struct fp16 {
	unsigned short x;
	fp16() {}
	fp16(const fp16& rhs): x(rhs.x) {}
	fp16& operator=(fp16 rhs){ x=rhs.x; return *this; }
	fp16(float t) {
		x = _cvtss_sh(t, 0);
	}
	float convert_to_float() const { return _cvtsh_ss(x); }
	explicit operator float() const {
		return convert_to_float();
	}
	explicit operator double() const {
		return static_cast<double>(convert_to_float());
	}

	fp16 operator+(fp16 rhs) const {
		return this->convert_to_float() + rhs.convert_to_float();
	}
	fp16 operator-(fp16 rhs) const {
		return this->convert_to_float() - rhs.convert_to_float();
	}
	fp16 operator*(fp16 rhs) const {
		return this->convert_to_float() * rhs.convert_to_float();
	}
};

#endif

#ifdef FP16_BFLIKE_FLOAT
#include <stdint.h>
#include <stdio.h>
#include <math.h>
struct fp16 {
	uint16_t x;
	fp16() {}
	fp16(const fp16& rhs): x(rhs.x) {}
	fp16& operator=(fp16 rhs){ x=rhs.x; return *this; }
	fp16(float f) {
		uint32_t t = *reinterpret_cast<uint32_t*>(&f);
		uint32_t exp =  t & 0x7f800000u;
		uint32_t mant = t & 0x007fffffu;
		int shift = 16 + 7 - BF_NMANT;
		x = (t>>shift);
		if(mant&(1u<<(shift-1))){
			uint32_t lowmant = mant & ((1u<<shift)-1u);
			uint32_t halfway = 1u << (shift-1);
			if(lowmant > halfway || (x&0x1u))
				++x;
		}
		/*{
			float o = this->convert_to_float();
			float e = (f==0.f? fabs(o-f): fabs(o-f)/fabs(f));
			if(e>1e-1) printf("XX %x %x %.15e -> %.15e :: %f\n", t, (uint32_t)x, f, o, e);
		}*/
	}
	float convert_to_float() const {
		// upcast is easy
		uint32_t t = ((uint32_t)x) << (16+7 - BF_NMANT);
		return *(float*)&t;
	}
	explicit operator float() const {
		return convert_to_float();
	}
	explicit operator double() const {
		return static_cast<double>(convert_to_float());
	}

	fp16 operator+(fp16 rhs) const {
		return this->convert_to_float() + rhs.convert_to_float();
	}
	fp16 operator-(fp16 rhs) const {
		return this->convert_to_float() - rhs.convert_to_float();
	}
	fp16 operator*(fp16 rhs) const {
		return this->convert_to_float() * rhs.convert_to_float();
	}
};
#endif

// double rounding causes larger error in very rare case. we ignore it for performance
inline float fp16_fma(fp16 a, fp16 b, fp16 c)
{
	float fa = a.convert_to_float();
	float fb = b.convert_to_float();
	float fc = c.convert_to_float();
	return fa*fb + fc;

}


void hgemm_opt(int m, int n, int k, float alpha, fp16 const* a, int lda, fp16 const* b, int ldb, float /*beta*/, fp16*c, int ldc);
inline void hgemm(int m, int n, int k, float alpha, fp16 const* a, int lda, fp16 const* b, int ldb, float beta, fp16*c, int ldc)
{
	#ifdef FP16_AVX2_EMULATION
	if(beta==1.f){ // remove this in the case for test
		hgemm_opt(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
		return;
	}
	#endif
	for(int j=0; j<m; ++j){
		for(int i=0; i<n; ++i){
			fp16 temp(0.f);
                        for(int l=0; l<k; ++l)
                                temp = fp16_fma(a[l*lda+j], b[i*ldb+l], temp);
                        c[ldc*i+j] = (c[ldc*i+j].convert_to_float() * beta + temp.convert_to_float()*alpha);
                }
        }
}


void shgemm_opt(int m, int n, int k, float alpha, fp16 const* a, int lda, fp16 const* b, int ldb, float /*beta*/, float*c, int ldc);
inline void shgemm(int m, int n, int k, float alpha, fp16 const* a, int lda, fp16 const* b, int ldb, float beta, float *c, int ldc)
{
	#ifdef FP16_AVX2_EMULATION
	if(beta==1.f){ // remove this in the case of test
		shgemm_opt(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
		return;
	}
	#endif
	for(int j=0; j<m; ++j){
		for(int i=0; i<n; ++i){
			float temp = 0.f;
                        for(int l=0; l<k; ++l)
                                temp = a[l*lda+j].convert_to_float() *  b[i*ldb+l].convert_to_float() + temp;
                        c[ldc*i+j] = c[ldc*i+j] * beta + temp*alpha;
                }
        }
}

#endif


#ifdef FP16_IS_NOT_SUPPORTED
#warning "FP16 IS NOT SUPPORTED"
typedef unsigned short fp16;

// do nothing. 
inline void hgemm(...){}
inline void shgemm(...){}

#endif

#if 0
// test code
#include "fp16sim.hpp"
#include <cstdlib>
#include <cstdio>

int main(){
	int m = 300, n = 210, k=101;
	fp16*a = (fp16*)malloc(sizeof(fp16)*m*k);
	fp16*b = (fp16*)malloc(sizeof(fp16)*k*n);
	fp16*c = (fp16*)malloc(sizeof(fp16)*m*n);
	fp16*c2 = (fp16*)malloc(sizeof(fp16)*m*n);
	for(int j=0; j<k; ++j) for(int i=0; i<m; ++i) a[m*j+i] = (float)std::rand()/RAND_MAX;
	for(int j=0; j<n; ++j) for(int i=0; i<k; ++i) b[k*j+i] = (float)std::rand()/RAND_MAX;
	for(int j=0; j<n; ++j) for(int i=0; i<m; ++i) c[m*j+i] = 0.f;
	for(int j=0; j<n; ++j) for(int i=0; i<m; ++i) c2[m*j+i] = 0.f;
	hgemm(m, n, k, -1.f, a, m, b, k, 1.f, c, m);
	hgemm_opt(m, n, k, -1.f, a, m, b, k, 1.f, c2, m);
	double error = 0.;
	for(int j=0; j<n; ++j) for(int i=0; i<m; ++i) {
		double t = (double)c[m*j+i] - (double)c2[m*j+i];
		t = t < 0. ? -t: t;
		error = t > error ? t: error;
		//std::printf("%d %d %e %e\n", i, j, (float)c[m*j+i], (float)c2[m*j+i]);
	}
	std::printf("hgemm error = %e\n", error);

	float*sc = (float*)malloc(sizeof(float)*m*n);
	float*sc2 = (float*)malloc(sizeof(float)*m*n);
	for(int j=0; j<n; ++j) for(int i=0; i<m; ++i) sc[m*j+i] = 0.f;
	for(int j=0; j<n; ++j) for(int i=0; i<m; ++i) sc2[m*j+i] = 0.f;
	shgemm(m, n, k, -1.f, a, m, b, k, 1.f, sc, m);
	shgemm_opt(m, n, k, -1.f, a, m, b, k, 1.f, sc2, m);
	error = 0.;
	for(int j=0; j<n; ++j) for(int i=0; i<m; ++i) {
		double t = (double)sc[m*j+i] - (double)sc2[m*j+i];
		t = t < 0. ? -t: t;
		error = t > error ? t: error;
		//std::printf("%d %d %e %e\n", i, j, (float)sc[m*j+i], (float)sc2[m*j+i]);
	}
	std::printf("shgemm error = %e\n", error);

	return 0;
}
#endif
#endif
