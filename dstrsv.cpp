#include <cstdio>
#include <cassert>

extern "C" {
	void dsgemv_pack(int n, float alpha, float const* a, int lda, void* buf);
	void dstrsvL_pack(int n, float alpha, float const* a, int lda, void* buf);
	void dstrsvU_pack(int n, float alpha, float const* a, int lda, void* buf);

	void dsgemv_compute(int n, void const* buf, double const* x, double* y);
	void dstrsvL_compute(int n, void const* buf, double* x);
	void dstrsvU_compute(int n, void const* buf, double* x);
}

#define UNIT_DIAG_LOWER
// #define UNIT_DIAG_UPPER

namespace {
void dstrsvL(int n, const float * __restrict aptr, int lda, double * __restrict x){
	auto a = [aptr, lda](int i, int j) -> const float &
	{ return aptr[i + lda*j]; };

	for(int j=0; j<n; j++){
#ifndef UNIT_DIAG_LOWER
		double xj = (x[j] /= (double)a(j,j));
#else
		double xj = x[j];
#endif
		for(int i=j+1; i<n; i++){
			x[i] -= (double)a(i,j) * xj;
		}
	}
}

void dstrsvU(int n, const float * __restrict aptr, int lda, double * __restrict x){
	auto a = [aptr, lda](int i, int j) -> const float &
	{ return aptr[i + lda*j]; };

	for(int j=n-1; j>=0; --j){
#ifndef UNIT_DIAG_UPPER
		double xj = (x[j] /= (double)a(j,j));
#else
		double xj = x[j];
#endif
		for(int i=0; i<j; i++){
			x[i] -= (double)a(i,j) * xj;
		}
	}
}

#ifdef UNIT_TEST
void dstrmvL(int n, const float *aptr, int lda, double *x){
	auto a = [aptr, lda](int i, int j) -> const float &
	{ return aptr[i + lda*j]; };

	for(int i=n-1; i>=0; --i){
#ifndef UNIT_DIAG_LOWER
		double sum = 0.0;
		for(int j=0; j<=i; j++){
			sum += (double)a(i,j) * x[j];
		}
#else
		double sum = x[i];
		for(int j=0; j<i; j++){
			sum += (double)a(i,j) * x[j];
		}
#endif
		x[i] = sum;
	}
}

void dstrmvU(int n, const float *aptr, int lda, double *x){
	auto a = [aptr, lda](int i, int j) -> const float &
	{ return aptr[i + lda*j]; };

	for(int i=0; i<n; i++){
#ifndef UNIT_DIAG_UPPER
		double sum = 0.0;
		for(int j=i; j<n; j++){
			sum += (double)a(i,j) * x[j];
		}
#else
		double sum = x[i];
		for(int j=i+1; j<n; j++){
			sum += (double)a(i,j) * x[j];
		}
#endif
		x[i] = sum;
	}
}
#endif // UNIT_TEST

void bs6_dstrsvL(const float *aptr, const size_t lda, double *x){
	auto a = [aptr, lda](int i, int j) -> const float &
	{ return aptr[i + lda*j]; };

#ifdef UNIT_DIAG_LOWER
	const bool UNIT = true;
#else
	const bool UNIT = false;
#endif
	double x0 = x[0];
	double x1 = x[1];
	double x2 = x[2];
	double x3 = x[3];
	double x4 = x[4];
	double x5 = x[5];

	double a10 = a(1,0);
	double a20 = a(2,0), a21 = a(2,1);
	double a30 = a(3,0), a31 = a(3,1), a32 = a(3,2);
	double a40 = a(4,0), a41 = a(4,1), a42 = a(4,2), a43 = a(4,3);
	double a50 = a(5,0), a51 = a(5,1), a52 = a(5,2), a53 = a(5,3), a54 = a(5,4);

	x0 = UNIT ? x0 : x0 * (1.0 /  a(0,0));
	x1 -= a10 * x0;
	x2 -= a20 * x0;
	x3 -= a30 * x0;
	x4 -= a40 * x0;
	x5 -= a50 * x0;

	x1 = UNIT ? x1 : x1 * (1.0 / a(1,1));
	x2 -= a21 * x1;
	x3 -= a31 * x1;
	x4 -= a41 * x1;
	x5 -= a51 * x1;

	x2 = UNIT ? x2 : x2 * (1.0 / a(2,2));
	x3 -= a32 * x2;
	x4 -= a42 * x2;
	x5 -= a52 * x2;

	x3 = UNIT ? x3 : x3 * (1.0 / a(3,3));
	x4 -= a43 * x3;
	x5 -= a53 * x3;

	x4 = UNIT ? x4 : x4 * (1.0 / a(4,4));
	x5 -= a54 * x4;

	x5 = UNIT ? x5 : x5 * (1.0 / a(5,5));

	x[0] = x0;
	x[1] = x1;
	x[2] = x2;
	x[3] = x3;
	x[4] = x4;
	x[5] = x5;
}

void bs6_dstrsvU(const float *aptr, const size_t lda, double *x){
	auto a = [aptr, lda](int i, int j) -> const float &
	{ return aptr[i + lda*j]; };

#ifdef UNIT_DIAG_UPPER
	const bool UNIT = true;
#else
	const bool UNIT = false;
#endif
	double x0 = x[0];
	double x1 = x[1];
	double x2 = x[2];
	double x3 = x[3];
	double x4 = x[4];
	double x5 = x[5];

	double a01 = a(0,1), a02 = a(0,2), a03 = a(0,3), a04 = a(0,4), a05 = a(0,5);
	double               a12 = a(1,2), a13 = a(1,3), a14 = a(1,4), a15 = a(1,5);
	double                             a23 = a(2,3), a24 = a(2,4), a25 = a(2,5);
	double                                           a34 = a(3,4), a35 = a(3,5);
	double                                                         a45 = a(4,5);

	x5 = UNIT ? x5 : x5 * (1.0 / a(5,5));
	x4 -= a45 * x5;
	x3 -= a35 * x5;
	x2 -= a25 * x5;
	x1 -= a15 * x5;
	x0 -= a05 * x5;

	x4 = UNIT ? x4 : x4 * (1.0 / a(4,4));
	x3 -= a34 * x4;
	x2 -= a24 * x4;
	x1 -= a14 * x4;
	x0 -= a04 * x4;

	x3 = UNIT ? x3 : x3 * (1.0 / a(3,3));
	x2 -= a23 * x3;
	x1 -= a13 * x3;
	x0 -= a03 * x3;

	x2 = UNIT ? x2 : x2 * (1.0 / a(2,2));
	x1 -= a12 * x2;
	x0 -= a02 * x2;

	x1 = UNIT ? x1 : x1 * (1.0 / a(1,1));
	x0 -= a01 * x1;

	x0 = UNIT ? x0 : x0 * (1.0 /  a(0,0));

	x[5] = x5;
	x[4] = x4;
	x[3] = x3;
	x[2] = x2;
	x[1] = x1;
	x[0] = x0;
}

void bs6_dsgemv(const int m, const float *__restrict aptr, const size_t lda, const double *__restrict x, double *__restrict y){
	enum{ N = 6, };
	auto a = [aptr, lda](int i, int j) -> const float &
	{ return aptr[i + lda*j]; };

	double x0 = x[0];
	double x1 = x[1];
	double x2 = x[2];
	double x3 = x[3];
	double x4 = x[4];
	double x5 = x[5];
	for(int i=0; i<m; i++){
		y[i] -= 
			a(i,0) * x0 + 
			a(i,1) * x1 + 
			a(i,2) * x2 + 
			a(i,3) * x3 + 
			a(i,4) * x4 + 
			a(i,5) * x5;
	}
}

} // namespace (anonymous)

extern "C" {
void dstrsvL_pack(int n, float alpha, float const* __restrict aptr, int lda, void* __restrict buf){
	// auto a = [aptr, lda](int i, int j) -> const float &
	// { return aptr[i + lda*j]; };
#if 0
	auto b = [buf, n](int i, int j) -> float &
	{ return ((float *)buf)[i + n*j]; };
#else
	float * __restrict bptr =  (float *)buf;
	// auto b = [bptr, n](int i, int j) -> float &
	// { return bptr[i + n*j]; };
#endif

#pragma omp parallel for
	for(int j=0; j<n; j++){
#if 1
#ifdef UNIT_DIAG_LOWER
		for(int i=j+1; i<n; i++){
#else
		for(int i=j; i<n; i++){
#endif
#else
#pragma loop novrec
		for(int i=0; i<n; i++){ // brute force
#endif
			// b(i,j) = alpha * a(i,j);
			bptr[i + n*j] = alpha * aptr[i + lda*j];
		}
	}
}

void dstrsvU_pack(int n, float alpha, float const* __restrict aptr, int lda, void* __restrict buf){
	// auto a = [aptr, lda](int i, int j) -> const float &
	// { return aptr[i + lda*j]; };
#if 0
	auto b = [buf, n](int i, int j) -> float &
	{ return ((float *)buf)[i + n*j]; };
#else
	float * __restrict bptr =  (float *)buf;
	// auto b = [bptr, n](int i, int j) -> float &
	// { return bptr[i + n*j]; };
#endif

#pragma omp parallel for
	for(int j=0; j<n; j++){
#if 1
#ifdef UNIT_DIAG_UPPER
		for(int i=0; i<j; i++){
#else
		for(int i=0; i<=j; i++){
#endif
#else
		for(int i=0; i<n; i++){ // brute force
#endif
			// b(i,j) = alpha * a(i,j);
			bptr[i + n*j] = alpha * aptr[i + lda*j];
		}
	}
}

void dstrsvL_compute(int n, void const* buf, double* x){
	enum{ BS = 6, };
	// assert(0 == n%BS);
	auto a = [buf, n](int i, int j) -> const float &
	{ return ((const float *)buf)[i + n*j]; };

	const int n6 = n / 6 * 6;
	for(int j=0; j<n6; j+=BS){
		bs6_dstrsvL(&a(j,j), n, &x[j]);
		bs6_dsgemv(n-j-BS, &a(j+BS,j), n, &x[j], &x[j+BS]);
	}
	const int nrem = n - n6;
	if(nrem){
		dstrsvL(nrem, &a(n6,n6), n, &x[n6]);
	}
}

void dstrsvU_compute(int n, void const* buf, double* x){
	enum{ BS = 6, };
	// assert(0 == n%BS);
	auto a = [buf, n](int i, int j) -> const float &
	{ return ((const float *)buf)[i + n*j]; };

	for(int j=n-BS; j>=0; j-=BS){
		bs6_dstrsvU(&a(j,j), n, &x[j]);
		bs6_dsgemv(j, &a(0,j), n, &x[j], &x[0]);
	}
	const int nrem = n%6;
	if(nrem){
		dstrsvU(nrem, &a(0,0), n, &x[0]);
	}
}
} // extern "C"

// #define UNIT_TEST
#ifdef UNIT_TEST

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>

template <int N>
int test_trsv_trmv(const float *a, const size_t lda, const double *b,
                   double torr = 1.e-13, int maxcnt = 100){
	static double y [N];
	static double bb[N];
	static double x [N];
	static double yy[N];

	memcpy(y, b, sizeof(y));
	dstrsvL(N, a, lda, y);

	memcpy(bb, y, sizeof(y));
	dstrmvL(N, a, lda, bb);

	memcpy(x, y, sizeof(y));
	dstrsvU(N, a, lda, x);

	memcpy(yy, x, sizeof(y));
	dstrmvU(N, a, lda, yy);

	int cnt = 0;
	for(int i=0; i<N; i++){
		double db = bb[i] - b[i];
		double dy = yy[i] - y[i];

		if(fabs(db) > torr || fabs(dy) > torr){
			std::printf("%4d : %+e, %+e\n",i, db, dy);
			if(++cnt >= maxcnt) break;
		}
	}
	if(0 == cnt) std::puts("PASS");

	return cnt;
}

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
#  endif
static double tick2second(uint64_t tick){
	return 1.e-9 * (double)tick;
}
#endif

template <int N>
int test_trsv(const float *a, const size_t lda, const double *b,
                   double torr = 1.e-13, int maxcnt = 100){
	using std::printf;
	using std::memcpy;
	using std::memset;

	static float buf[N*N];
	static double y [N];
	static double y2[N];
	static double x [N];
	static double x2[N];

#pragma omp parallel for
	for(int i=0; i<N*N; i++){
		buf[i] = -0.0f;
	}
	// memset(buf, -1, sizeof(buf)); // chache init

	memcpy(y, b, sizeof(y));
	auto t10 = get_utime();
	dstrsvL(N, a, lda, y);
	auto t20 = get_utime();

	memcpy(y2, b, sizeof(y));
	auto t30 = get_utime();
	dstrsvL_pack(N, 1.0f, a, lda, buf);
	auto t40 = get_utime();
	dstrsvL_compute(N, buf, y2);
	auto t50 = get_utime();

	memcpy(x, y, sizeof(y));
	auto t110 = get_utime();
	dstrsvU(N, a, lda, x);
	auto t120 = get_utime();

	memcpy(x2, y2, sizeof(y));
	auto t130 = get_utime();
	dstrsvU_pack(N, 1.0f, a, lda, buf);
	auto t140 = get_utime();
	dstrsvU_compute(N, buf, x2);
	auto t150 = get_utime();

	double dtL1 = tick2second(t20 - t10);
	double dtL2 = tick2second(t40 - t30);
	double dtL3 = tick2second(t50 - t40);

	double dtU1 = tick2second(t120 - t110);
	double dtU2 = tick2second(t140 - t130);
	double dtU3 = tick2second(t150 - t140);

	printf("naiveL : N=%d, %10.4f usec, %10.4f GB/s\n", N, dtL1*1.e6, 1.e-9*sizeof(float)*N*N/2/dtL1);
	printf("pack_L : N=%d, %10.4f usec, %10.4f GB/s\n", N, dtL2*1.e6, 1.e-9*sizeof(float)*N*N/2/dtL2);
	printf("solveL : N=%d, %10.4f usec, %10.4f GB/s\n", N, dtL3*1.e6, 1.e-9*sizeof(float)*N*N/2/dtL3);

	printf("naiveU : N=%d, %10.4f usec, %10.4f GB/s\n", N, dtU1*1.e6, 1.e-9*sizeof(float)*N*N/2/dtU1);
	printf("pack_U : N=%d, %10.4f usec, %10.4f GB/s\n", N, dtU2*1.e6, 1.e-9*sizeof(float)*N*N/2/dtU2);
	printf("solveU : N=%d, %10.4f usec, %10.4f GB/s\n", N, dtU3*1.e6, 1.e-9*sizeof(float)*N*N/2/dtU3);

	int cnt = 0;
	for(int i=0; i<N; i++){
		double dy = y2[i] - y[i];
		double dx = x2[i] - x[i];

		if(fabs(dy) > torr || fabs(dx) > torr){
			printf("%4d : %+e, %+e\n",i, dy, dx);
			if(++cnt >= maxcnt) break;
		}
	}
	if(0 == cnt) puts("PASS");

	return cnt;
}

int main(){
	enum{
		N   = 1152,
		LDA = N + 16,
		BS  = 6,
		NTH = 12,
	};
	static float a[N][LDA];
	static double b[N];

	srand48(20200220);

	for(int j=0; j<N; j++){
		for(int i=0; i<N; i++){
			a[j][i] = (float)(1.0 + drand48());
			if(j < i) a[j][i] *= 0.25f;
			if(j > i) a[j][i] *= 0.125f;
		}
	}
	for(int i=0; i<N; i++){
		b[i] = drand48() - 0.5;
	}

	test_trsv_trmv<N/4>(a[0], LDA, b);
	test_trsv_trmv<N/2>(a[0], LDA, b);
	test_trsv_trmv<N/1>(a[0], LDA, b);
	test_trsv_trmv<320>(a[0], LDA, b);
	test_trsv_trmv<640>(a[0], LDA, b);

	test_trsv<N/4>(a[0], LDA, b);
	test_trsv<N/2>(a[0], LDA, b);
	test_trsv<N/1>(a[0], LDA, b);
	test_trsv<320>(a[0], LDA, b);
	test_trsv<640>(a[0], LDA, b);
}

#endif // UNIT_TEST
