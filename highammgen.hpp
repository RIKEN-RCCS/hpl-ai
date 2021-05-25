#ifndef _HIGHAMMGEN_HPP
#define _HIGHAMMGEN_HPP
extern "C" double higham_mat_comp_beta(int n, double cond, double rho);
template<typename F>
struct HMGen {
	// generator of the Higham's HPL-AI matrix in https://github.com/higham/hpl-ai-matrix
	// It generates the matrix A=LU with L and U have the special structure.
	// L is the lower-triangular matrix with the diagonals = 1 and the strictly lower-triangular part = alpha.
	// U is the upper-triangular matrix with the diagonals = 1 and the strictly upper-triangular part = beta.
	int n; // the matrix size
	double alpha, beta;
	double scalea, scaleb; // scaling for left and right panels. 
	F* diag; // diagonal part of the matrix. 
	HMGen(int n, double cond, double rho, F* diag): n(n), diag(diag) {
		// alpha and beta are automatically computed from the condition number and others
		beta = higham_mat_comp_beta(n, cond, rho);
		alpha = rho * beta; // rhos is the ration of alhpa and beta.
		#if 0
		// this scaling may be lead to too good result
		scalea = 1. / (alpha * 32.);
		scaleb = 1. / (beta * 16.);
		#else
		// we observed that alpha~beta~O(1/n).
		scalea = n / (32.);
		scaleb = n / (16.);
		#endif
	}
};
#endif
