#ifndef PANEL_NORM_HPP
#define PANEL_NORM_HPP
#include "panel.hpp"
#include "grid.hpp"
#include "highammgen.hpp"
#include <mpi.h>

template<typename F>
F calc_infnorm(int n, F const* x)
{
	F norm = static_cast<F>(0);
	#pragma omp parallel for simd reduction(max:norm)
	for(int i=0; i<n; ++i) {
		F t = x[i];
		t = (t >= static_cast<F>(0) ? t: -t);
		norm = (norm >= t ? norm: t);
	}
	return norm;
}
template<typename F>
double colv_infnorm(Panels<F>const& p, double* dx, Grid& g)
{
	// computes the inf-norm of the distributed column vector dx.
	// descriptros are derived from p
	int nprow = p.nprow;
	int b = p.b;
	int i1 = p.i1;
	int j1 = p.j1;
	int istride = p.istride;
	int jstride = p.jstride;
	double norm = 0.;
	for(int i=0; i<nprow; ++i){
		int ipos = i1 + i*istride;
		if((ipos%jstride) == j1){
			double t = calc_infnorm(b, dx+b*i);
			norm = norm >= t ? norm: t;
		}
	}
	MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_MAX, g.commworld);
	return norm;
}

template<typename FPanel>
double panel_infnorm(Matgen<double> const& mg, Panels<FPanel>const& p, double* w, double* piv, Grid& g)
{
	// compute the inf-norm of the matrix.
	// w and piv are working buffer.

	// matrix inf-norm is the inf-norm of the row 1-norms.
	int b = p.b;
	int i1 = p.i1;
	int j1 = p.j1;
	int istride = p.istride;
	int jstride = p.jstride;
	int nprow = p.nprow;
	int npcol = p.npcol;
	for(int i=0; i<b*nprow; ++i) w[i] = 0.;
	for(int j=0; j<npcol; ++j){
		int jpos = j1 + j*jstride;
		for(int i=0; i<nprow; ++i){
			int ipos = i1 + i*istride;
			fill_one_panel_with_rand(mg.n, b*ipos, b*jpos, b, b, piv, b, mg.seed, true);
			for(int jj=0; jj<b; ++jj)
				for(int ii=0; ii<b; ++ii){
					double t = piv[jj*b+ii];
					w[b*i+ii] += (t<0. ? -t: t);
				}
		}
	}
	MPI_Allreduce(MPI_IN_PLACE, w, b*nprow, MPI_DOUBLE, MPI_SUM, g.hcomm);
	double norm = 0.;
	for(int i=0; i<b*nprow; ++i) norm = (norm >= w[i] ? norm: w[i]);
	MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_MAX, g.vcomm);
	return norm;
}


template<typename F>
double hpl_infnorm(Panels<F>const& p, double* d, Grid& g)
{
	// the diagonal of the hpl-ai matrix is the sum of the absolute values of the off-diagonals on the same row.
	// therefore, twice of the diagonal is the l1-norm of that row.
	return 2. * colv_infnorm(p, d, g);
}

template<typename FPanel>
double higham_infnorm(HMGen<double> const& mg, Panels<FPanel>const& p, double* w, Grid& g)
{
	int b = p.b;
	int i1 = p.i1;
	int j1 = p.j1;
	int istride = p.istride;
	int jstride = p.jstride;
	int nprow = p.nprow;
	int npcol = p.npcol;
	double alpha = mg.alpha;
	double beta = mg.beta;
	double ab = alpha * beta;
	for(int i=0; i<b*nprow; ++i) w[i] = 0.;
	#pragma omp parallel for
	for(int i=0; i<nprow; ++i){
		int ipos = i1 + i*istride;
		for(int j=0; j<npcol; ++j){
			int jpos = j1 + j*jstride;
			if(ipos==jpos){
				for(int jj=0; jj<b; ++jj){
					for(int ii=0; ii<jj; ++ii){
						double aij = beta + ab * (b*ipos + ii);
						w[b*i+ii] += aij;
					}
					w[b*j+jj] += 1. + ab * (b*jpos + jj);
					for(int ii=jj+1; ii<b; ++ii){
						double aij = alpha + ab * (b*jpos + jj);
						w[b*i+ii] += aij;
					}
				}
			}
			else if(ipos<jpos){
				for(int jj=0; jj<b; ++jj){
					for(int ii=0; ii<b; ++ii){
						double aij = beta + ab * (b*ipos + ii);
						w[b*i+ii] += aij;
					}
				}
			}
			else {
				for(int jj=0; jj<b; ++jj){
					for(int ii=0; ii<b; ++ii){
						double aij = alpha + ab * (b*jpos + jj);
						w[b*i+ii] += aij;
					}
				}
			}
		}
	}
	MPI_Allreduce(MPI_IN_PLACE, w, b*nprow, MPI_DOUBLE, MPI_SUM, g.hcomm);
	double norm = 0.;
	for(int i=0; i<b*nprow; ++i) norm = (norm >= w[i] ? norm: w[i]);
	MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_MAX, g.vcomm);
	return norm;
}

#endif
