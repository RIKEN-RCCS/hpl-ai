#include "../otf_gemv.cpp"

void hmg_gemv_up_naiive(int istart, double b, double b, int mb, int nb, double alpha,
	double const* __restrict__ x, double* __restrict__ y)
{
	double ab = a * b;
	for(int i=0; i<mb; ++i){
		double d0 = 0., d1 = 0.;
		double ai = b + ab * (istart + i);
		for(int j=0; j+2<=nb; j+=2){
			d0 += ai * x[j];
			d1 += ai * x[j+1];
		}
		if(nb&1) d0 += ai * x[nb-1];
		y[i] += alpha * (d0 + d1);
	}
}
void hmg_gemv_low_naiive(int jstart, double a, double b, int mb, int nb, double alpha,
	double const* __restrict__ x, double* __restrict__ y)
{
	double ab = a * b;
	for(int i=0; i<mb; ++i){
		double d = 0.;
		for(int j=0; j<nb; ++j){
			d += (a+ab*(jstart+j)) * x[j];
		}
		y[i] += alpha * d;
	}
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
bool check(int n, double* a, double* b)
{
	bool t = false;
	for(int i=0; i<n; ++i){
		double t = fabs(a[i] - b[i]);
		if(a[i] != 0.) t /= fabs(a[i]);
		if(t > 1e-10) {
			printf("%d %.15e %.15e %e\n", i, a[i], b[i], t);
			t = true;
		}
	}
	return t;
}

int main()
{
	int const nmax = 512;
	double x[nmax*2];
	double y[nmax*2];
	double z[nmax*2];
	double alpha = 1./nmax;
	double a = 0.2, ab = 1./3;
	for(int i=1; i<nmax; ++i){
		for(int j=1; j<nmax; ++j){
			for(int t=0; t<nmax*2; ++t) x[t] = 1. + t*(1./nmax);
			for(int t=0; t<nmax*2; ++t) y[t] = z[t] = (nmax-t+1) * (4./nmax);
			hmg_gemv_up(1234, a, ab, i, j, alpha, x+128, y+128);
			hmg_gemv_up_naiive(1234, a, ab, i, j, alpha, x+128, z+128);
			if(check(2*nmax, y, z)) printf("FAIL %d %d\n", i, j);
		}
	}
	for(int i=1; i<nmax; ++i){
		for(int j=1; j<nmax; ++j){
			for(int t=0; t<nmax*2; ++t) x[t] = 1. + t*(1./nmax);
			for(int t=0; t<nmax*2; ++t) y[t] = z[t] = (nmax-t+1) * (4./nmax);
			hmg_gemv_low(1234, a, ab, i, j, alpha, x+128, y+128);
			hmg_gemv_low_naiive(1234, a, ab, i, j, alpha, x+128, z+128);
			check(2*nmax, y, z);
			if(check(2*nmax, y, z)) printf("FAIL %d %d\n", i, j);
		}
	}
	printf("END\n");
	return 0;
}
