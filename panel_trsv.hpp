#ifndef PANEL_TRSV_HPP
#define PANEL_TRSV_HPP
#include "panel.hpp"
#include "grid.hpp"
#include "timer.hpp"
#include <mpi.h>

#include <omp.h>
#define TRSV_SPARE_THREAD
template <typename FAcc, typename FPanel>
void ttgemv(int m, int n, FAcc alpha, FPanel const* a, int lda, FAcc const * x, FAcc beta, FAcc * y, bool spare=false)
{
	if(alpha==static_cast<FAcc>(0)) return;
	int id = omp_get_thread_num();
	int nt = omp_get_num_threads();
	int ibegin, iend;
	if(!spare || nt==1){
		int nn = (m+31)/32;
		int nb = (nn+nt-1) / nt;
		ibegin = 32*nb*id;
		iend = 32*nb*(id+1);
		iend = iend>m ? m: iend;
	}
	else {
		if(id==0) return;
		int nn = (m+31)/32;
		int nb = (nn+nt-2) / (nt-1);
		ibegin = 32*nb*(id-1);
		iend = 32*nb*id;
		iend = iend>m ? m: iend;

	}
	if(beta!=static_cast<FAcc>(1)){
		if(beta==static_cast<FAcc>(0)) {
			for(int i=ibegin; i<iend; ++i) y[i] = 0.;
		}
		else {
			for(int i=ibegin; i<iend; ++i) y[i] *= beta;
		}
	}
	for(int j=0; j<n; ++j){
		for(int i=ibegin; i<iend; ++i)
			y[i] += alpha * static_cast<FAcc>(a[j*lda+i]) * x[j];
	}
}

#if (defined __FUJITSU) || (defined __CLANG_FUJITSU)
#include <arm_sve.h>
void ttgemv(int m, int n, double alpha, float const * __restrict__ a, int lda, double const * __restrict__ x, double beta, double * __restrict__ y, bool spare=false)
{
	// designed for large vector
	assert(0 == n%4);
	int id = omp_get_thread_num();
	int nt = omp_get_num_threads();
	int ibegin, iend;
	if(!spare || nt==1){
		int nn = (m+31)/32;
		int nb = (nn+nt-1) / nt;
		ibegin = 32*nb*id;
		iend = 32*nb*(id+1);
		iend = iend>m ? m: iend;
	}
	else {
		if(id==0) return;
		int nn = (m+31)/32;
		int nb = (nn+nt-2) / (nt-1);
		ibegin = 32*nb*(id-1);
		iend = 32*nb*id;
		iend = iend>m ? m: iend;

	}
	if(beta == 0.0){
		for(int i=ibegin; i<iend; ++i){
			y[i] = 0.0;
		}
	}else if(beta!=1.){
		for(int i=ibegin; i<iend; ++i){
			y[i] *= beta;
		}
	}

	// svfloat64_t valpha = svdup_f64(alpha);
	size_t ldaa = lda;
	for(int j=0; j<n; j+=4, a+=4*lda){
		const double x0 = x[j+0] * alpha;
		const double x1 = x[j+1] * alpha;
		const double x2 = x[j+2] * alpha;
		const double x3 = x[j+3] * alpha;
		#pragma loop novrec
		#pragma loop simd
		for(int i=ibegin; i<iend; ++i){
			double a0 = a[i+ldaa*0];
			double a1 = a[i+ldaa*1];
			double a2 = a[i+ldaa*2];
			double a3 = a[i+ldaa*3];
			y[i] += a0*x0 + a1*x1 + a2*x2 + a3*x3;
		}
	}
}
#endif

template <typename FAcc, typename FPanel>
void ttgemv_range(int istart, int iend, int pj, FAcc alpha, Panels<FPanel> const& p, FAcc const * x, FAcc beta, FAcc * y, bool spare=false)
{
	if(istart>=iend) return;
	int const b = p.b;
	size_t const lda = p.lda;
	if(p.is_tile)
		for(int pi=istart; pi<iend; ++pi)
			ttgemv(b, b, alpha, p(pi, pj), p.lda, x, beta, y, spare);
	else
		ttgemv(b*(iend-istart), b, alpha, p(istart, pj), lda, x, beta, y, spare);
}

template <typename FAcc, typename FPanel>
void tttrsvL(int n, FPanel const* __restrict__ a, int lda, FAcc* __restrict__ x)
{
	for(int j=0; j+1<n; ++j){
		FAcc alpha = x[j];
		for(int i=j+1; i<n; ++i){
			x[i] -= alpha * static_cast<FAcc>(a[i + j*lda]);
		}
	}
}


#ifndef TRSV_DEBUG
extern "C" void strsv_(...);
extern "C" void dtrsv_(...);
inline void tttrsvL(int n, float const* __restrict__ a, int lda, float* __restrict__ x)
{
	int incx = 1;
	strsv_("L", "N", "U", &n, a, &lda, x, &incx);
}
inline void tttrsvL(int n, double const* __restrict__ a, int lda, double* __restrict__ x)
{
	int incx = 1;
	dtrsv_("L", "N", "U", &n, a, &lda, x, &incx);
}
#endif

template <typename FAcc, typename FPanel>
void tttrsvU(int n, FPanel const* __restrict__ a, int lda, FAcc* __restrict__ x)
{
	for(int jj=n-1; jj>=0; --jj){
		x[jj] /= static_cast<FAcc>(a[(jj)*lda + (jj)]);
		FAcc alpha = x[jj];
		#pragma loop novrec
		for(int ii=0; ii<jj; ++ii){
			x[ii] -= alpha * static_cast<FAcc>(a[(jj)*lda + (ii)]);
		}
	}
}

#ifndef TRSV_DEBUG
inline void tttrsvU(int n, float const* __restrict__ a, int lda, float* __restrict__ x)
{
	int incx = 1;
	strsv_("U", "N", "N", &n, a, &lda, x, &incx);
}
inline void tttrsvU(int n, double const* __restrict__ a, int lda, double* __restrict__ x)
{
	int incx = 1;
	dtrsv_("U", "N", "N", &n, a, &lda, x, &incx);
}
#endif

template<typename FPanel>
void ttgemv_pack(int n, FPanel alpha, FPanel const*__restrict__ a, int lda, FPanel*__restrict__ buf)
{
	size_t nn = n;
	size_t ldaa = lda;
	size_t const bi = 32;
	FPanel*__restrict__ to = reinterpret_cast<FPanel*>(buf);
	#pragma omp parallel for
	for(size_t ii=0; ii<nn; ii+=bi){
		if(nn-ii>=bi){
			for(size_t j=0; j<nn; ++j){
				#if defined(__FUJITSU) || defined(__CLANG_FUJITSU)
				#pragma loop novrec
				#pragma loop unroll
				#pragma loop simd
				#else
				#pragma omp simd 
				#endif
				for(size_t i=0; i<bi; ++i){
					to[nn*ii+bi*j+i] = alpha*a[j*ldaa+ii+i];
				}
			}
		}
		else{
			for(size_t j=0; j<nn; ++j){
				for(size_t i=0; i+ii<nn; ++i){
					to[nn*ii+bi*j+i] = alpha*a[j*ldaa+ii+i];
				}
			}
		}
	}
}
template<typename FAcc, typename FPanel>
void ttgemv_compute(int n, FPanel const*__restrict__ buf, FAcc const*__restrict__ x, FAcc*__restrict__ y)
{
	size_t const bi = 32;
	size_t const nn = n;
	FPanel const*__restrict__ a = reinterpret_cast<FPanel const*>(buf);
	#pragma omp parallel for
	for(size_t ii=0; ii<nn; ii+=bi){
		FPanel const*__restrict__ aii = a + nn*ii;
		if(nn-ii>=bi){
			FAcc yii[bi];
			for(size_t i=0; i<bi; ++i) yii[i] = 0;
			for(size_t j=0; j<nn; ++j){
				FAcc xj = x[j];
				#if defined(__FUJITSU) || defined(__CLANG_FUJITSU)
				#pragma loop novrec
				#pragma loop unroll
				#pragma loop simd
				#else
				#pragma omp simd
				#endif
				for(size_t i=0; i<bi; ++i){
					yii[i] += aii[bi*j+i] * xj;
				}
			}
			for(size_t i=0; i<bi; ++i) y[ii+i] = yii[i];

		}
		else {
			// slow path
			FAcc yii[bi];
			for(size_t i=0; i+ii<nn; ++i) yii[i] = 0;
			for(size_t j=0; j<nn; ++j){
				FAcc xj = x[j];
				for(size_t i=0; i+ii<nn; ++i){
					yii[i] += aii[bi*j+i] * xj;
				}
			}
			for(size_t i=0; i+ii<nn; ++i) y[ii+i] = yii[i];
		}
	}
}
#if 0
extern "C" dsgemv_pack(int n, float alpha, int lda, float* buf);
void ttgemv_pack(int n, float alpha, float const* a, int lda, float* buf)
{
	dsgemv_pack(n, alpha, a, lda, buf);
}
extern "C" dsgemv_compute(int n, float const* buf, double const* x, double* y);
void ttgemv_compute(int n, float const* buf, double const* x, double* y) 
{
	dsgemv_compute(n, buf, x, y);
}
#endif

template<typename FPanel>
void tttrsvL_pack(int n, FPanel alpha, FPanel const* a, int lda, FPanel* buf)
{
	// only for test
	FPanel* to = reinterpret_cast<FPanel*>(buf);
	for(int j=0; j<n; ++j){
		for(int i=j; i<n; ++i)
			to[j*n + i] = alpha * a[j*lda + i];
	}
}
template<typename FAcc, typename FPanel>
void tttrsvL_compute(int n, FPanel const* buf, FAcc* x)
{
	tttrsvL(n, buf, n, x);
}

template<typename FPanel>
void tttrsvU_pack(int n, FPanel alpha, FPanel const* a, int lda, FPanel* buf)
{
	// only for test
	FPanel* to = reinterpret_cast<FPanel*>(buf);
	for(int j=0; j<n; ++j){
		for(int i=0; i<=j; ++i)
			to[j*n + i] = alpha * a[j*lda + i];
	}
}
template<typename FAcc, typename FPanel>
void tttrsvU_compute(int n, FPanel const* buf, FAcc* x)
{
	tttrsvU(n, buf, n, x);
}

#if (defined(__FUJITSU) || defined(__CLANG_FUJITSU)) && defined(__ARM_FEATURE_SVE)
extern "C" void dstrsvL_pack(int n, float alpha, float const* a, int lda, void* buf);
extern "C" void dstrsvL_compute(int n, void const* buf, double* x);
void tttrsvL_pack(int n, float alpha, float const* a, int lda, float * buf)
{
	dstrsvL_pack(n, alpha, a, lda, reinterpret_cast<void*>(buf));
}
void tttrsvL_compute(int n, float const* buf, double * x)
{
	dstrsvL_compute(n, reinterpret_cast<void const*>(buf), x);
}
extern "C" void dstrsvU_pack(int n, float alpha, float const* a, int lda, void* buf);
extern "C" void dstrsvU_compute(int n, void const* buf, double* x);
void tttrsvU_pack(int n, float alpha, float const* a, int lda, float * buf)
{
	dstrsvU_pack(n, alpha, a, lda, reinterpret_cast<void*>(buf));
}
void tttrsvU_compute(int n, float const* buf, double * x)
{
	dstrsvU_compute(n, reinterpret_cast<void const*>(buf), x);
}
#endif

template <typename FAcc, typename FPanel>
void panel_trsvL(Panels<FPanel>const &p,  FAcc* x, FAcc* w, size_t ldv, Grid &grid)
{
	// x is a column vector
	// only the diagonal part need to be valid
	// all the values will be modified after the computation
	// w1 and w2 are working space which has same size with x.
	// w3 has length b
	int const nb = p.nblocks;
	int const b = p.b;
	int const i1 = p.i1;
	int const j1 = p.j1;
	int const istride = p.istride;
	int const jstride = p.jstride;
	int const nprow = p.nprow;
	int const npcol = p.npcol;

	int const lda = p.lda;

	bool const single_col = grid.ncol == 1;
	int const left = single_col ? MPI_PROC_NULL: (grid.col == 0 ? grid.ncol - 1: grid.col - 1);
	int const right = single_col ? MPI_PROC_NULL: (grid.col == grid.ncol - 1 ? 0: grid.col + 1);
	int const top = (grid.row == 0 ? grid.nrow - 1: grid.row - 1);
	int const bottom = (grid.row == grid.nrow - 1? 0: grid.row + 1);

	FAcc *wr = w, *ws = w+ldv, *wt = w+2*ldv, *w3 = w+3*ldv;
	FPanel *wp1 = reinterpret_cast<FPanel*>(w3 + b);
	FPanel *wp2 = wp1 + b*b;
	FAcc const dzero = 0, done = 1;
	FPanel const sone = 1;

	MPI_Request reqs[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
	int offset = 0;

	#pragma omp simd
	for(int i=0; i<b*nprow; ++i) wr[i] = dzero;

	for(int pj=0; pj<npcol; ++pj){
		MPI_Request req_recv_pivv=MPI_REQUEST_NULL, req_recv_v=MPI_REQUEST_NULL;
		int gj = j1 + pj*jstride;
		int cleft = gj==0 ? MPI_PROC_NULL: left;
		int pivot = gj%istride;
		int istart = (gj < i1 ? 0: (gj-i1+istride-1)/istride);
		int ii = i1 + istart * istride;
		if(ii>=nb) break;
		bool impivot = pivot == i1;
		bool no_bottom = grid.nrow==1 || ii+1>=nb;
		if(impivot){
			MPI_Request req_recv_pivv, req_recv_v=MPI_REQUEST_NULL;
			Timer::beg(Timer::IR_TRSV_COMM);
			MPI_Irecv(wr+istart*b, b, T2MPI<FAcc>::type, cleft, 200, grid.hcomm, &req_recv_pivv);
			if(istart+1<nprow)
				MPI_Irecv(wr+b*(istart+1), b*(nprow-istart-1), T2MPI<FAcc>::type, cleft, 200, grid.hcomm, &req_recv_v);
			Timer::end(Timer::IR_TRSV_COMM);

			Timer::beg(Timer::IR_TRSV, true);
			tttrsvL_pack(b, sone, p(istart, pj), lda, wp1);
			Timer::end(Timer::IR_TRSV);
			if(istart+1<nprow){
				Timer::beg(Timer::IR_TRSV_MV, true);
				ttgemv_pack(b, sone, p(istart+1, pj), lda, wp2);
				Timer::end(Timer::IR_TRSV_MV);
			}


			Timer::beg(Timer::IR_TRSV_COMM, true);
			MPI_Wait(&req_recv_pivv, MPI_STATUS_IGNORE);
			Timer::end(Timer::IR_TRSV_COMM);
			
			// compute the pivot first
			#pragma omp simd
			for(int i=b*istart; i<b*istart+b; ++i) x[i] -= wr[i];
			Timer::beg(Timer::IR_TRSV);
			tttrsvL_compute(b, wp1, x+b*istart);
			Timer::end(Timer::IR_TRSV, false, 1ull*b*b);
			if(!no_bottom){
				Timer::beg(Timer::IR_TRSV_COMM, true);
				MPI_Send(x+b*istart, b, T2MPI<FAcc>::type, bottom, 100, grid.vcomm);
				Timer::end(Timer::IR_TRSV_COMM);
			}
			if(istart+1<nprow){
				Timer::beg(Timer::IR_TRSV_MV, true);
				ttgemv_compute(b, wp2, x+b*istart, wt+b*(istart+1));
				//ttgemv(b, b, done, p(istart+1, pj), lda, x+b*istart, dzero, wt+b*(istart+1));
				Timer::end(Timer::IR_TRSV_MV, false, 2ull*b*b);
				// compute others last
				if(istart+2<nprow){
					#ifdef TRSV_SPARE_THREAD
					#pragma omp parallel
					{
						if(omp_get_thread_num() == 0) {
							Timer::beg(Timer::IR_TRSV_COMM);
							MPI_Wait(&req_recv_v, MPI_STATUS_IGNORE);
							#pragma omp simd
							for(int i=b*(istart+1); i<b*(istart+2); ++i) wr[i] += wt[i];
							MPI_Isend(wr+b*(istart+1), b, T2MPI<FAcc>::type, right, 200, grid.hcomm, reqs+offset);
							MPI_Waitall(2, reqs+2-offset, MPI_STATUSES_IGNORE);
							Timer::end(Timer::IR_TRSV_COMM);
							Timer::beg(Timer::IR_TRSV_MV, true);
						}
						ttgemv_range(istart+2, nprow, pj, done, p, x+b*istart, dzero, wt+b*(istart+2), true);
					}
					Timer::end(Timer::IR_TRSV_MV);

					#else
					Timer::beg(Timer::IR_TRSV_COMM, true);
					MPI_Wait(&req_recv_v, MPI_STATUS_IGNORE);
					#pragma omp simd
					for(int i=b*(istart+1); i<b*(istart+2); ++i) wr[i] += wt[i];
					MPI_Isend(wr+b*(istart+1), b, T2MPI<FAcc>::type, right, 200, grid.hcomm, reqs+offset);
					Timer::end(Timer::IR_TRSV_COMM);

					Timer::beg(Timer::IR_TRSV_MV, true);
					ttgemv_range(istart+2, nprow, pj, done, p, x+b*istart, dzero, wt+b*(istart+2));
					Timer::end(Timer::IR_TRSV_MV, false, 2ull*b*(nprow-istart-2));
					#endif

					#pragma omp simd
					for(int i=b*(istart+2); i<b*nprow; ++i) wr[i] += wt[i];
					Timer::beg(Timer::IR_TRSV_COMM, true);
					MPI_Isend(wr+b*(istart+2), b*(nprow-istart-2), T2MPI<FAcc>::type, right, 200, grid.hcomm, reqs+offset+1);
					Timer::end(Timer::IR_TRSV_COMM);
				}
				else {
					Timer::beg(Timer::IR_TRSV_COMM, true);
					MPI_Wait(&req_recv_v, MPI_STATUS_IGNORE);
					Timer::end(Timer::IR_TRSV_COMM);

					#pragma omp simd
					for(int i=b*(istart+1); i<b*(istart+2); ++i) wr[i] += wt[i];
					
					Timer::beg(Timer::IR_TRSV_COMM, true);
					MPI_Isend(wr+b*(istart+1), b, T2MPI<FAcc>::type, right, 200, grid.hcomm, reqs+offset);
					Timer::end(Timer::IR_TRSV_COMM);
				}
			}	
		}
		else {
			bool bottom_is_pivot = pivot == bottom;
			Timer::beg(Timer::IR_TRSV_COMM);
			MPI_Irecv(wr+istart*b, b, T2MPI<FAcc>::type, cleft, 200, grid.hcomm, &req_recv_pivv);
			if(istart+1<nprow)
				MPI_Irecv(wr+(istart+1)*b, b*(nprow-istart-1), T2MPI<FAcc>::type, cleft, 200, grid.hcomm, &req_recv_v);
			Timer::end(Timer::IR_TRSV_COMM);

			Timer::beg(Timer::IR_TRSV_MV, true);
			ttgemv_pack(b, sone, p(istart, pj), lda, wp2);
			Timer::end(Timer::IR_TRSV_MV);
			
			
			Timer::beg(Timer::IR_TRSV_COMM, true);
			MPI_Recv(w3, b, T2MPI<FAcc>::type, top, 100, grid.vcomm, MPI_STATUS_IGNORE);
			if(!bottom_is_pivot && !no_bottom)
				MPI_Send(w3, b, T2MPI<FAcc>::type, bottom, 100, grid.vcomm);
			Timer::end(Timer::IR_TRSV_COMM);
			
			// compute the critical-path first
			Timer::beg(Timer::IR_TRSV_MV, true);
			ttgemv_compute(b, wp2, w3, wt+b*istart);
			//ttgemv(b, b, done, p(istart, pj), lda, w3, dzero, wt+b*istart);
			Timer::end(Timer::IR_TRSV_MV, false, 2ull*b*b);

			// compute others
			if(istart+1 < nprow){
				#ifdef TRSV_SPARE_THREAD
				#pragma omp parallel
				{
					if(omp_get_thread_num() == 0){
						Timer::beg(Timer::IR_TRSV_COMM);
						MPI_Wait(&req_recv_pivv, MPI_STATUS_IGNORE);
						#pragma omp simd
						for(int i=b*istart; i<b*(istart+1); ++i) wr[i] += wt[i];
						MPI_Isend(wr+b*istart, b, T2MPI<FAcc>::type, right, 200, grid.hcomm, reqs+offset);
						MPI_Wait(&req_recv_v, MPI_STATUS_IGNORE);
						MPI_Waitall(2, reqs+2-offset, MPI_STATUSES_IGNORE);
						Timer::end(Timer::IR_TRSV_COMM);
						Timer::beg(Timer::IR_TRSV_MV, true);
					}
					ttgemv_range(istart+1, nprow, pj, done, p, w3, dzero, wt+b*(istart+1), true);
				}
				Timer::end(Timer::IR_TRSV_MV);

				#else
				Timer::beg(Timer::IR_TRSV_COMM, true);
				MPI_Wait(&req_recv_pivv, MPI_STATUS_IGNORE);
				#pragma omp simd
				for(int i=b*istart; i<b*(istart+1); ++i) wr[i] += wt[i];
				MPI_Isend(wr+b*istart, b, T2MPI<FAcc>::type, right, 200, grid.hcomm, reqs+offset);
				Timer::end(Timer::IR_TRSV_COMM);

				Timer::beg(Timer::IR_TRSV_MV, true);
				ttgemv_range(istart+1, nprow, pj, done, p, w3, dzero, wt+b*(istart+1));
				Timer::end(Timer::IR_TRSV_MV, false, 2ull*b*(nprow-istart-1));

				Timer::beg(Timer::IR_TRSV_COMM, true);
				MPI_Wait(&req_recv_v, MPI_STATUS_IGNORE);
				Timer::end(Timer::IR_TRSV_COMM);
				#endif

				#pragma omp simd
				for(int i=b*(istart+1); i<b*nprow; ++i) wr[i] += wt[i];
				Timer::beg(Timer::IR_TRSV_COMM);
				MPI_Isend(wr+b*(istart+1), b*(nprow-istart-1), T2MPI<FAcc>::type, right, 200, grid.hcomm, reqs+offset+1);
				Timer::end(Timer::IR_TRSV_COMM);
			}
			else {
				Timer::beg(Timer::IR_TRSV_COMM, true);
				MPI_Wait(&req_recv_pivv, MPI_STATUS_IGNORE);
				Timer::end(Timer::IR_TRSV_COMM);

				#pragma omp simd
				for(int i=b*istart; i<b*(istart+1); ++i) wr[i] += wt[i];
				Timer::beg(Timer::IR_TRSV_COMM);
				MPI_Isend(wr+b*istart, b, T2MPI<FAcc>::type, right, 200, grid.hcomm, reqs+offset);
				Timer::end(Timer::IR_TRSV_COMM);
			}

		}
		Timer::beg(Timer::IR_TRSV_COMM);
		MPI_Waitall(2, reqs+2-offset, MPI_STATUSES_IGNORE);
		Timer::end(Timer::IR_TRSV_COMM);
		offset = 2-offset;
		FAcc* t = wr; wr=ws; ws=t;
	}
	MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
}

template <typename FAcc, typename FPanel>
void panel_trsvU(Panels<FPanel>const &p,  FAcc* x, FAcc* w, size_t ldv, Grid &grid)
{
	// x is a column vector
	// only the diagonal part need to be valid
	// all the values will be modified after the computation
	// w1 and w2 are working space which has same size with x.
	// w3 has length b
	int const nb = p.nblocks;
	int const b = p.b;
	int const i1 = p.i1;
	int const j1 = p.j1;
	int const istride = p.istride;
	int const jstride = p.jstride;
	int const nprow = p.nprow;
	int const npcol = p.npcol;
	int const lda = p.lda;

	bool const single_col = grid.ncol == 1;
	int const left = single_col ? MPI_PROC_NULL : (grid.col == 0 ? grid.ncol - 1: grid.col - 1);
	int const right = single_col ? MPI_PROC_NULL : (grid.col == grid.ncol - 1 ? 0: grid.col + 1);
	int const top = (grid.row == 0 ? grid.nrow - 1: grid.row - 1);
	int const bottom = (grid.row == grid.nrow - 1? 0: grid.row + 1);

	FAcc done = 1, dzero = 0;
	FPanel sone = 1;
	FAcc *wr = w, *ws = w+ldv, *wt = w+2*ldv, *w3 = w+3*ldv;
	FPanel* wp1 = reinterpret_cast<FPanel*>(w3+b);
	FPanel* wp2 = wp1 + b*b;
	MPI_Request reqs[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
	int offset = 2;

	#pragma omp parallel for simd
	for(int i=0; i<b*nprow; ++i) wr[i] = dzero;

	for(int pj=npcol-1; pj>=0; --pj){
		MPI_Request req_recv_pivv=MPI_REQUEST_NULL, req_recv_v=MPI_REQUEST_NULL;
		int gj = j1 + pj*jstride;
		if(gj < i1) break;
		int pivot = gj%istride;
		bool impivot = pivot == i1;
		int iend = gj/istride + (i1 <= pivot ? 1: 0);
		if(impivot){
			bool no_top = (gj == 0) || (grid.nrow==1);
			if(gj != nb-1){
				Timer::beg(Timer::IR_TRSV_COMM);
				MPI_Irecv(wr+b*(iend-1), b, T2MPI<FAcc>::type, right, 200, grid.hcomm, &req_recv_pivv);
				if(iend>1)
					MPI_Irecv(wr, b*(iend-1), T2MPI<FAcc>::type, right, 200, grid.hcomm, &req_recv_v);
				Timer::end(Timer::IR_TRSV_COMM);
			}
			Timer::beg(Timer::IR_TRSV);
			tttrsvU_pack(b, sone, p(iend-1, pj), lda, wp1);
			Timer::end(Timer::IR_TRSV);
			if(iend>1){
				Timer::beg(Timer::IR_TRSV_MV, true);
				ttgemv_pack(b, sone, p(iend-2, pj), lda, wp2);
				Timer::end(Timer::IR_TRSV_MV);
			}
			Timer::beg(Timer::IR_TRSV_COMM);
			MPI_Wait(&req_recv_pivv, MPI_STATUS_IGNORE);
			Timer::end(Timer::IR_TRSV_COMM);

			#pragma omp simd
			for(int i=b*(iend-1); i<b*iend; ++i) x[i] -= wr[i];
			// compute the pivot first
			Timer::beg(Timer::IR_TRSV, true);
			tttrsvU_compute(b, wp1, x+b*(iend-1));
			Timer::end(Timer::IR_TRSV, false, 1ull*b*b);
			if(!no_top){
				Timer::beg(Timer::IR_TRSV_COMM, true);
				MPI_Send(x+b*(iend-1), b, T2MPI<FAcc>::type, top, 100, grid.vcomm);
				Timer::end(Timer::IR_TRSV_COMM);
			}
			if(iend>1){
				Timer::beg(Timer::IR_TRSV_MV, true);
				ttgemv_compute(b, wp2, x+b*(iend-1), wt+b*(iend-2));
				Timer::end(Timer::IR_TRSV_MV, false, 2ull*b*b);
				if(iend>2){
					#ifdef TRSV_SPARE_THREAD
					#pragma omp parallel
					{
						if(omp_get_thread_num()==0){
							Timer::beg(Timer::IR_TRSV_COMM);
							MPI_Wait(&req_recv_v, MPI_STATUS_IGNORE);
							#pragma omp simd
							for(int i=b*(iend-2); i<b*(iend-1); ++i) wr[i] += wt[i];
							MPI_Isend(wr+b*(iend-2), b, T2MPI<FAcc>::type, left, 200, grid.hcomm, reqs+offset);
							MPI_Waitall(2, reqs+(2-offset), MPI_STATUSES_IGNORE);
							Timer::end(Timer::IR_TRSV_COMM);
							Timer::beg(Timer::IR_TRSV_MV);
						}
						ttgemv_range(0, iend-2, pj, done, p, x+b*(iend-1), dzero, wt, true);
					}
					Timer::end(Timer::IR_TRSV_MV);

					#else
					Timer::beg(Timer::IR_TRSV_COMM);
					MPI_Wait(&req_recv_v, MPI_STATUS_IGNORE);
					#pragma omp simd
					for(int i=b*(iend-2); i<b*(iend-1); ++i) wr[i] += wt[i];
					MPI_Isend(wr+b*(iend-2), b, T2MPI<FAcc>::type, left, 200, grid.hcomm, reqs+offset);
					Timer::end(Timer::IR_TRSV_COMM);

					Timer::beg(Timer::IR_TRSV_MV);
					ttgemv_range(0, iend-2, pj, done, p, x+b*(iend-1), dzero, wt);
					Timer::end(Timer::IR_TRSV_MV, false, 2ull*b*(iend-2));

					#endif

					#pragma omp simd
					for(int i=0; i<b*(iend-2); ++i) wr[i] += wt[i];
					Timer::beg(Timer::IR_TRSV_COMM, true);
					MPI_Isend(wr, b*(iend-2), T2MPI<FAcc>::type, left, 200, grid.hcomm, reqs+offset+1);
					Timer::end(Timer::IR_TRSV_COMM);
				}
				else {
					Timer::beg(Timer::IR_TRSV_COMM, true);
					MPI_Wait(&req_recv_v, MPI_STATUS_IGNORE);
					Timer::end(Timer::IR_TRSV_COMM);

					#pragma omp simd
					for(int i=b*(iend-2); i<b*(iend-1); ++i) wr[i] += wt[i];
					Timer::beg(Timer::IR_TRSV_COMM, true);
					MPI_Isend(wr+b*(iend-2), b, T2MPI<FAcc>::type, left, 200, grid.hcomm, reqs+offset);
					Timer::end(Timer::IR_TRSV_COMM);
				}
			}
			Timer::beg(Timer::IR_TRSV_COMM, true);
			MPI_Waitall(2, reqs+(2-offset), MPI_STATUSES_IGNORE);
			Timer::end(Timer::IR_TRSV_COMM);
			FAcc* t=wr; wr=ws; ws=t;
			offset = 2-offset;
		}
		else {
			bool stop_bcast = pivot == top || (gj < istride && i1 == 0);
			if(gj != nb-1){
				Timer::beg(Timer::IR_TRSV_COMM);
				MPI_Irecv(wr+b*(iend-1), b, T2MPI<FAcc>::type, right, 200, grid.hcomm, &req_recv_pivv);
				if(iend>1)
					MPI_Irecv(wr, b*(iend-1), T2MPI<FAcc>::type, right, 200, grid.hcomm, &req_recv_v);
				Timer::end(Timer::IR_TRSV_COMM);
			}
			Timer::beg(Timer::IR_TRSV_MV);
			ttgemv_pack(b, sone, p(iend-1, pj), lda, wp2);
			Timer::end(Timer::IR_TRSV_MV);

			Timer::beg(Timer::IR_TRSV_COMM, true);
			MPI_Recv(w3, b, T2MPI<FAcc>::type, bottom, 100, grid.vcomm, MPI_STATUS_IGNORE);
			if(!stop_bcast)
				MPI_Send(w3, b, T2MPI<FAcc>::type, top, 100, grid.vcomm);
			Timer::end(Timer::IR_TRSV_COMM);
			// compute the critical-path first
			Timer::beg(Timer::IR_TRSV_MV, true);
			ttgemv_compute(b, wp2, w3, wt+b*(iend-1));
			Timer::end(Timer::IR_TRSV_MV, false, 2ull*b*b);

			// compute others last
			if(iend>1){
				#ifdef TRSV_SPARE_THREAD
				#pragma omp parallel
				{
					if(omp_get_thread_num()==0){
						Timer::beg(Timer::IR_TRSV_COMM);
						MPI_Wait(&req_recv_pivv, MPI_STATUS_IGNORE);

						#pragma omp simd
						for(int i=b*(iend-1); i<b*iend; ++i) wr[i] += wt[i];
						MPI_Isend(wr+b*(iend-1), b, T2MPI<FAcc>::type, left, 200, grid.hcomm, reqs+offset);
						MPI_Wait(&req_recv_v, MPI_STATUS_IGNORE);
						MPI_Waitall(2, reqs+(2-offset), MPI_STATUSES_IGNORE);
						Timer::end(Timer::IR_TRSV_COMM);
						Timer::beg(Timer::IR_TRSV_MV);
					}
					ttgemv_range(0, iend-1, pj, done, p, w3, dzero, wt, true);
				}
				Timer::end(Timer::IR_TRSV_MV);

				#else
				Timer::beg(Timer::IR_TRSV_COMM, true);
				MPI_Wait(&req_recv_pivv, MPI_STATUS_IGNORE);
				#pragma omp simd
				for(int i=b*(iend-1); i<b*iend; ++i) wr[i] += wt[i];
				MPI_Isend(wr+b*(iend-1), b, T2MPI<FAcc>::type, left, 200, grid.hcomm, reqs+offset);
				Timer::end(Timer::IR_TRSV_COMM);

				Timer::beg(Timer::IR_TRSV_MV, true);
				ttgemv_range(0, iend-1, pj, done, p, w3, dzero, wt);
				Timer::end(Timer::IR_TRSV_MV, false, 2ull*b*(iend-1));

				Timer::beg(Timer::IR_TRSV_COMM, true);
				MPI_Wait(&req_recv_v, MPI_STATUS_IGNORE);
				Timer::end(Timer::IR_TRSV_COMM);
				#endif

				#pragma omp simd
				for(int i=0; i<b*(iend-1); ++i) wr[i] += wt[i];
				Timer::beg(Timer::IR_TRSV_COMM);
				MPI_Isend(wr, b*(iend-1), T2MPI<FAcc>::type, left, 200, grid.hcomm, reqs+offset+1);
				Timer::end(Timer::IR_TRSV_COMM);
			}
			else{
				Timer::beg(Timer::IR_TRSV_COMM, true);
				MPI_Wait(&req_recv_pivv, MPI_STATUS_IGNORE);
				Timer::end(Timer::IR_TRSV_COMM);

				#pragma omp simd
				for(int i=b*(iend-1); i<b*iend; ++i) wr[i] += wt[i];
				Timer::beg(Timer::IR_TRSV_COMM);
				MPI_Isend(wr+b*(iend-1), b, T2MPI<FAcc>::type, left, 200, grid.hcomm, reqs+offset);
				Timer::end(Timer::IR_TRSV_COMM);
			}
		}
		Timer::beg(Timer::IR_TRSV_COMM, true);
		MPI_Waitall(2, reqs+(2-offset), MPI_STATUSES_IGNORE);
		Timer::end(Timer::IR_TRSV_COMM);

		FAcc* t=wr; wr=ws; ws=t;
		offset = 2-offset;
	}
	MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
}

#endif
