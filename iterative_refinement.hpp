#ifndef ITERATIVE_REFINEMENT_HPP
#define ITERATIVE_REFINEMENT_HPP
#include "panel.hpp"
#include "grid.hpp"
#include "hpl_rand.hpp"
#include "panel_gemv.hpp"
#include "panel_trsv.hpp"
#include "panel_norm.hpp"
#include "timer.hpp"
#include <cstdio>
#include <cfloat>

struct IRErrors {
	double residual;
	double hpl_harness;
};

template<typename FPanel, template<class> class Matgen>
IRErrors iterative_refinement(Panels<FPanel>const&p, Matgen<double>& mg,
	double* x, double* w, size_t ldv, double* rhs, double norma, double normb, int maxit, Grid&grid)
{
	// do IR with approximated LU factors in p and the accurate initial matrix which is generated by mg.
	// x is solution. rhs is the right-hand-side vector.
	// w is working vectors. ldv is the leading dimention of w. set good ldv for better performance.
	// norma is inf-norm of the initial matrix. normb is the inf-norm of rhs.
	int const nb = p.nblocks;
	int const n = nb * p.b;
	double*r = w;
	double*v = w + ldv;

	// initial approximation, x_0 = diag(A)^{-1} b
	// this is nice approximation for the matrix of the HPL-AI bench (2019-11-13).
	copycolv(p, rhs, x);
	divcolv(p, mg.diag, x);
	for(int iter=0; iter<maxit; ++iter){
		copycolv(p, rhs, r);
		double normx = colv_infnorm(p, x, grid);
		colv2rowv(p, x, v);
		// compute residual, r_i = b - A x_i
		panel_gemv(-1., p, mg, false, v, 1., r, grid);
		double normr = colv_infnorm(p, r, grid);
		// hplerror := \|b-Ax\|_\infty / (\|A\|_\infty \|x\|_\infty + \|b\|_\infty) * (n * \epsilon)^{-1}
		double hplerror = normr / (norma*normx + normb) * 1./(n * DBL_EPSILON/2);
		if(grid.row==0 && grid.col==0){
			std::printf("# iterative refinement: step=%3d, residual=%20.16e hpl-harness=%f\n", iter, normr, hplerror);
			fflush(stdout);
		}
		if(hplerror < 16.) return {normr, hplerror};

		// x_1 = x_0 + (LU)^{-1} r
		panel_trsvL(p, r, v, ldv, grid);
		panel_trsvU(p, r, v, ldv, grid);
		addcolv(p, r, x);
	}
	// OMG! 
	return {-1., -1.};
}

/*template<typename FPanel>
IRErrors iterative_refinement(Panels<FPanel>const&p, HMGen<double>& mg,
	double* x, double* w, size_t ldv, double* rhs, double norma, double normb, int maxit, Grid&grid)
{
	int const nb = p.nblocks;
	int const n = nb * p.b;
	double*r = w;
	double*v = w + ldv;
	// initial approximation
	// trsv
	copycolv(p, rhs, x);
	panel_trsvL(p, x, v, ldv, grid);
	panel_trsvU(p, x, v, ldv, grid);

	for(int iter=0; iter<maxit; ++iter){
		copycolv(p, rhs, r);
		double normx = colv_infnorm(p, x, grid);
		colv2rowv(p, x, v);
		// compute residual
		panel_gemv(-1., p, mg, false, v, 1., r, grid);
		double normr = colv_infnorm(p, r, grid);
		double hplerror = normr / (norma*normx + normb) * 1./(n * DBL_EPSILON/2);
		if(grid.row==0 && grid.col==0){
			std::printf("# iterative refinement: step=%3d, residual=%20.16e hpl-harness=%f\n", iter, normr, hplerror);
			fflush(stdout);
		}
		if(hplerror < 16.) return {normr, hplerror};

		panel_trsvL(p, r, v, ldv, grid);
		panel_trsvU(p, r, v, ldv, grid);
		addcolv(p, r, x);
	}
	// OMG! 
	return {-1., -1.};
}*/
#endif
