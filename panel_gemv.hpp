#ifndef PANEL_GEMV_HPP
#define PANEL_GEMV_HPP
#include "panel.hpp"
#include "hpl_rand.hpp"
#include "grid.hpp"
#include "timer.hpp"
#include "highammgen.hpp"
#include <mpi.h>

// on-the-fly computation of gemv.
extern "C" void otf_gemv_kernel(int64_t n, int mb, int nb, double alpha, double const* __restrict__ x, double* __restrict__ y, uint64_t seed);
extern "C" void hmg_gemv_up(int i, double a, double b, int mb, int nb, double alpha, double const* __restrict__ x, double* __restrict__ y);
extern "C" void hmg_gemv_low(int j, double a, double b, int mb, int nb, double alpha, double const* __restrict__ x, double* __restrict__ y);
extern "C" void hmg_gemv_diag(int i, int j, double a, double b, int mb, int nb, double alpha, double const* __restrict__ x, double* __restrict__ y);

static void otf_gemv_small(int b, double alpha, double const* __restrict__ x, double* __restrict__ y, double const * __restrict__ diag,
	RandStat stat_00, RandCoeff incl1, RandCoeff jumpn)
{
	for(int i=0; i<b; ++i){
		RandStat stat_j = stat_00;
		double d = diag[i];
		double t = 0.;
		for(int j=0; j<b; ++j){
			double aij = (i==j ? d: static_cast<double>(stat_j));
			t += aij * x[j];
			stat_j = jumpn * stat_j;
		}
		y[i] += alpha * t;
		stat_00 = incl1 * stat_00;
	}
}
static void otf_gemv_kernel_diag(int64_t n, int b, double alpha, double const* __restrict__ x, double* __restrict__ y, double const * __restrict__ diag,
	RandStat stat_00, RandCoeff incl1, RandCoeff jumpn)
{
	int microb = 64;
	RandCoeff jumpmb = incl1.pow(microb);
	RandCoeff jumpmbn = jumpn.pow(microb);
	RandCoeff jumppi = jumpn.pow(0);
	for(int pi=0; pi<b; pi+=microb){
		RandStat stat_j = stat_00;
		int bi = (b-pi < microb ? b-pi: microb);
		if(pi) {
			otf_gemv_kernel(n, bi, pi, alpha, x, y+pi, stat_j.x);
			stat_j = jumppi * stat_j;
		}
		otf_gemv_small(bi, alpha, x+pi, y+pi, diag+pi, stat_j, incl1, jumpn);
		if(b>pi+microb) {
			stat_j = jumpmbn * stat_j;
			otf_gemv_kernel(n, bi, b-pi-microb, alpha, x+pi+microb, y+pi, stat_j.x);
		}

		stat_00 = jumpmb * stat_00;
		jumppi = jumppi * jumpmbn;
	}
}

template<typename FPanel>
void otf_gemv(Matgen<double> const& mg, Panels<FPanel>const & p,
	int rowstart, int rowend, int colstart, int colend, double alpha, double const* x, double* y)
{
	// assuming x is full row.
	int const b = p.b;
	int const i1 = p.i1;
	int const j1 = p.j1;
	int const istride = p.istride;
	int const jstride = p.jstride;
	RandCoeff incl1 = mg.incl1;
	RandCoeff jumpn = mg.jumpn;
	double const* diag = mg.diag;
	for(int pj=colstart; pj<colend; ++pj){
		int j0 = j1 + pj * jstride;
		#pragma omp parallel for schedule(dynamic,1)
		for(int pi=rowstart; pi<rowend; ++pi){
			int i0 = i1 + pi * istride;
			RandCoeff jump_ij = mg.jump(b * i0, b * j0);
			RandStat stat_00 = jump_ij * RandStat::initialize(mg.seed);
			if(i0 != j0)
				otf_gemv_kernel(mg.n, b, b, alpha, x+b*pj, y+b*pi, stat_00.x);
			else 
				otf_gemv_kernel_diag(mg.n, b, alpha, x+b*pj, y+b*pi, diag+b*pi, stat_00, incl1, jumpn);
		}
	}
}

template<typename FPanel>
void otf_gemv(HMGen<double> const& mg, Panels<FPanel>const & p,
	int rowstart, int rowend, int colstart, int colend, double alpha, double const* x, double* y)
{
	// assuming x is full row.
	int const b = p.b;
	int const i1 = p.i1;
	int const j1 = p.j1;
	int const istride = p.istride;
	int const jstride = p.jstride;
	for(int pj=colstart; pj<colend; ++pj){
		int j0 = j1 + pj * jstride;
		#pragma omp parallel for schedule(dynamic,1)
		for(int pi=rowstart; pi<rowend; ++pi){
			int i0 = i1 + pi * istride;
			if(i0 < j0)
				hmg_gemv_up(b*i0, mg.alpha, mg.beta, b, b, alpha, x+b*pj, y+b*pi);
			else if(i0 > j0)
				hmg_gemv_low(b*j0, mg.alpha, mg.beta, b, b, alpha, x+b*pj, y+b*pi);
			else 
				hmg_gemv_diag(b*i0, b*j0, mg.alpha, mg.beta, b, b, alpha, x+b*pj, y+b*pi);
		}
	}
}

template <typename FPanel, typename FAcc, template<class> class Matgen>
void panel_gemv(FAcc alpha, Panels<FPanel>const &p, Matgen<FAcc>const& mg, bool x_is_full, FAcc* x, FAcc beta, FAcc* y, Grid &grid)
{
	// compute y = beta*y + alpha * p * x
	// it can be p.nprow != p.npcol.

	// in: p
	// inout: x(M), y(M)
	// where M = max(p.nprow, p.npcol) * b

	// x is the (partial) row vector. x_j = X_{j1+(j-1)*jstride}. where 1<=j<=npcol.
	// x_j is valid iff j0+(j-1)*jstride >= i0 and (j0+(j-1)*jstride-i0)%istride == 0.
	// If !!x_is_valid, it is assumed that all the data is valid.
	// x was modified and become full at the end of the process

	// y is the (partial) column vector. x_i = Y_{i1+(i-1)*istride}, where 1<=i<=nprow.
	// y_i is valid iff i0+(i-1)*istride >= j0 and (i0+(i-1)*istride-j0)%jstride == 0.
	// Other part of vector are invalid; they have arbitrary values but MODIFIED after the computation.

	int const b = p.b;
	int const i1 = p.i1;
	int const j1 = p.j1;
	int const istride = p.istride;
	int const jstride = p.jstride;
	int const nprow = p.nprow;
	int const npcol = p.npcol;
	FAcc const dzero = static_cast<FAcc>(0);
	// first: initialize y data
	for(int i=0; i<nprow; ++i){
		int ipos = i1 + i*istride;
		if((ipos%jstride) == j1){
			// it is a valid block
			#pragma omp parallel for simd
			for(int k=i*b; k<i*b+b; ++k) y[k] *= beta;
		}
		else {
			// it is an invalid block
			#pragma omp parallel for simd
			for(int k=i*b; k<i*b+b; ++k) y[k] = dzero;
		}
	}

	if(x_is_full || grid.nrow == 1){
		// easy case. just call GEMV. 
		Timer::beg(Timer::IR_GEMV);
		otf_gemv(mg, p, 0, nprow, 0, npcol, alpha, x, y);
		Timer::end(Timer::IR_GEMV, false, 2ull*nprow*npcol*b*b);
	}
	else {
		// multi-step bcast gemv
		MPI_Request req[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
		int rootrow = (j1) % istride;
		Timer::beg(Timer::IR_GEMV_COMM);
		if(npcol>0) MPI_Ibcast(x, b, T2MPI<FAcc>::type, rootrow, grid.vcomm, req);
		Timer::end(Timer::IR_GEMV_COMM);
		for(int j=0; j<npcol; ++j){
			int nextrootrow = (j1+(j+1)*jstride) % istride;
			Timer::beg(Timer::IR_GEMV_COMM, true);
			if(j!=npcol-1) MPI_Ibcast(x+b*(j+1), b, T2MPI<FAcc>::type, nextrootrow, grid.vcomm, req+(j+1)%2);
			MPI_Wait(req+j%2, MPI_STATUS_IGNORE);
			Timer::end(Timer::IR_GEMV_COMM);
			Timer::beg(Timer::IR_GEMV, true);
			otf_gemv(mg, p, 0, nprow, j, j+1, alpha, x, y);
			Timer::end(Timer::IR_GEMV, false, 2ull*nprow*b*b);
		}
		MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
	}
	Timer::beg(Timer::IR_GEMV_COMM, true);
	MPI_Allreduce(MPI_IN_PLACE, y, b*nprow, T2MPI<FAcc>::type, MPI_SUM, grid.hcomm);
	Timer::end(Timer::IR_GEMV_COMM);
}

#endif
