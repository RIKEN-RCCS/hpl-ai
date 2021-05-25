#include <cstdlib>
#include <cstdio>
#include <cmath>
#define TRSV_DEBUG
#include "../panel_trsv.hpp"
extern "C" void dtrsv_(...);
int main()
{
	for(int n=2; n<3000; n=(3*n+1)/2){
		int lda = n + 11;
		double * a = (double*)std::malloc(sizeof(double)*lda*n);
		double * x = (double*)std::malloc(sizeof(double)*n);
		double * x2 = (double*)std::malloc(sizeof(double)*n);
		for(int j=0; j<n; ++j) for(int i=0; i<n; ++i) a[j*lda+i] = std::rand() * (1./RAND_MAX);
		for(int i=0; i<n; ++i) x[i] = x2[i] = std::rand() * (1./RAND_MAX);
		int incx = 1;
		tttrsvL(n, a, lda, x);
		dtrsv_("L", "N", "U", &n, a, &lda, x2, &incx);
		std::printf("%d\n", n);
		for(int i=0; i<n; ++i)
			if(std::fabs(x[i]-x2[i])/std::fabs(x[i])>1e-5)
				std::printf("%3d, %e :: %f, %f\n", i, std::fabs(x[i]-x2[i]), x[i], x2[i]);

		for(int j=0; j<n; ++j) for(int i=0; i<n; ++i) a[j*lda+i] = std::rand() * (1./RAND_MAX);
		for(int i=0; i<n; ++i) x[i] = x2[i] = std::rand() * (1./RAND_MAX);
		tttrsvU(n, a, lda, x);
		dtrsv_("U", "N", "N", &n, a, &lda, x2, &incx);
		for(int i=0; i<n; ++i)
			if(std::fabs(x[i]-x2[i])/std::fabs(x[i])>1e-5)
				std::printf("%3d, %e :: %f, %f\n", i, std::fabs(x[i]-x2[i]), x[i], x2[i]);
				
		std::free(a);
		std::free(x);
		std::free(x2);
	}
	return 0;
}