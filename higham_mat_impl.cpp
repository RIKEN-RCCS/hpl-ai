// Copyright (c) 2020, Massimiliano Fasi and Nicholas J. Higham 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// The above copyright notice and the code are from https://github.com/higham/hpl-ai-matrix

// This file is manual translation of the above software.
#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>
#define MAX(A,B) ((A)>(B) ? (A): (B))
#define MIN(A,B) ((A)>(B) ? (B): (A))
double fhpl(int n, double alpha, double beta)
{
	// compute the inf-norm condition number of the matrix with alpha and beta
	// FHPL   Value of cond(A,inf) for matrix A(n,a,b).
	if(isnan(alpha)) alpha = beta / 2;
	double a = alpha, b = beta;
	double lambda_1 = 1 + (n-1)*b;
	int idash = MIN((int)floor(1./a), n);
	int k = MIN((int)floor((1+b)/b), n-1);
	double lambda_idash = 1 + (2*k-idash+1)*a + (n-idash)*b + (-k*k+k+3*idash*(idash-1)/2 - n*idash+n)*a*b;
	double lambda_n = 1 + (2*k-n+1)*a + (-k*k+k+n*(n-1)/2)*a*b;
	double na_est = MAX(MAX(lambda_1, lambda_idash), lambda_n);
	double r = (1+a)*(1+b);
	int i = 1;
	double delta1 = (1+a)*(1./(1+a) + (r==0.?0.:b*(1-pow(r,n-1))/(1-r)));
	double deltan = pow(1+a,n) * (1./(1+a));
	double ninva_est = MAX(delta1, deltan);
	//printf("Z %e %e %e %e %e %e %e %e %e\n", a, b, lambda_1, lambda_idash, lambda_n, delta1, deltan, na_est, ninva_est);
	double ret = na_est * ninva_est;
	if(isinf(ret)) return DBL_MAX;
	else return ret;
}

template<typename F>
double zero_find(F f, double left, double right)
{
	// bisection method
	// the brent method consumes half # of f evaluations, it's not good enought for complication
	double fl = f(left);
	double fr = f(right);
	if(fl > 0. || fr < 0.) return 0./0.; // nan
	while(true){
		double tol1 = (2. * fabs(right) + 0.5) * DBL_EPSILON;
		if(right-left < tol1) break;
		double middle = (left+right)/2;
		if(middle==left || middle==right) break;
		double fm = f(middle);
		//printf("%e %e %e :: %e %e %e\n", left, middle, right, fl, fm, fr);
		if(fm==0.) return middle;
		else if(fm<0.) {
			left = middle;
			fl = fm;
		}
		else {
			right = middle;
			fr = fm;
		}
	}
	return right;
}


extern "C"
double higham_mat_comp_beta(int n, double kappa, double rho)
{
	// % Compute alpha and beta to give cond(A,inf) = kappa.
	double left = DBL_EPSILON;
	double left_val = fhpl(n, rho*left, left) - kappa;
	assert(left_val < 0.);
	double right = 1./rho;
	int k = 1;
	while(true){
		double right_val = fhpl(n, rho*right, right) - kappa;
		if(isfinite(right_val) && right_val > 0.) break;
		// %fprintf('F at right endpoint, right = %9.2e, is %9.2e.\n', right, right_val)
		right *= 0.5;
		++k;
		if(k==100) break;
	}
	double beta = zero_find([=](double x) -> double {return fhpl(n,rho*x,x)-kappa;}, left, right);
	double alpha = rho * beta;
	while(alpha > 1.){
		// fprintf('Initial alpha = %9.2e exceeds 1 so recomputing.\n', alpha)
		right *= 0.5;
		right = right/2;
		beta = zero_find([=](double x) -> double {return fhpl(n,rho*x,x)-kappa;}, left, right);
		alpha = rho * beta;
	}
	return beta;
}

extern "C"
void hplai_matrix_impl(int n, double* a, int lda, double alpha, double beta)
{
	for(int j=0; j<n; ++j){
		for(int i=0; i<j; ++i){
			a[j*lda+i] = -beta + alpha*beta*i;
		}
		a[j*lda+j] = 1. + alpha*beta*j;
		for(int i=j+1; i<n; ++i){
			a[j*lda+i] = -alpha + alpha*beta*j;
		}
	}
}

extern "C"
void hplai_matrix(int n, double* a, int lda, double kappa)
{
	double rho = 0.5;
	double beta = higham_mat_comp_beta(n, kappa, rho);
	double alpha = rho * beta;
	hplai_matrix_impl(n, a, lda, alpha, beta);
}


#if 0
#include <stdio.h>
int main()
{
	int n = 10;
	double kappa = 1000;
	double rho = 0.125;
	for(int n=10; n<100000000; n=(n*3)/2){
		double beta = comp_beta(n, kappa, rho);
		printf("%d %e %e\n", n, beta, rho*beta*beta*n);
	}
	return 0;
}
#endif
