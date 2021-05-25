#ifndef PANEL_TRF_HPP
#define PANEL_TRF_HPP
#include "panel.hpp"
#include "grid.hpp"
#include "hpl_rand.hpp"
#include "lazy_init.hpp"
#include "kernels/kernel.h"
#include "schur_updator.hpp"
#include "getrf_nopiv.hpp"
#include "back_buffer.hpp"
#include <mpi.h>
#include <cmath>

struct RequestStack {
	int nreq;
	int maxnreq;
	MPI_Request* reqs;
	RequestStack(int num): nreq(0), maxnreq(num) {
		reqs = (MPI_Request*)malloc(sizeof(MPI_Request)*maxnreq);
	}
	~RequestStack(){
		free(reqs);
	}
	RequestStack(RequestStack const&) = delete;
	RequestStack& operator=(RequestStack const&) = delete;
	MPI_Request* get_request() {
		if(nreq==maxnreq) std::abort(); // Die imidiately
		return reqs + nreq++;
	}
	void wait_all() {
		if(!nreq) return;
		MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
		nreq = 0;
	}
	bool test_all() {
		if(nreq==0) return true;
		int flags = 0;
		MPI_Testall(nreq, reqs, &flags, MPI_STATUSES_IGNORE);
		if(flags){
			nreq = 0;
			return true;
		}
		else return false;
	}
	void wait_all(Timer::Items item) {
		Timer::beg(item);
		wait_all();
		Timer::end(item);
	}
	bool test_all(Timer::Items item) {
		Timer::beg(item);
		bool ret = test_all();
		Timer::end(item);
		return ret;
	}
};

// communications
template <typename F>
void broadcast_pivlu(bool is_tile, int b, F *lu, size_t lda, F* piv, size_t ldpiv, Grid &grid, RequestStack& req)
{
	// broadcast the pivot block to row and column

	Timer::beg(Timer::DIAG_BCAST);
	req.wait_all();
	if(is_tile){
		MPI_Ibcast(const_cast<F *>(lu), lda * b, T2MPI<F>::type, grid.row, grid.vcomm, req.get_request());
		MPI_Ibcast(const_cast<F *>(lu), lda * b, T2MPI<F>::type, grid.col, grid.hcomm, req.get_request());
	}
	else {
		MPI_Ibcast(piv, ldpiv* b, T2MPI<F>::type, grid.row, grid.vcomm, req.get_request());
		MPI_Ibcast(piv, ldpiv* b, T2MPI<F>::type, grid.col, grid.hcomm, req.get_request());
		#pragma omp parallel for
		for(int j=0; j<b; ++j)
			for(int i=0; i<b; ++i)
				lu[j*lda + i] = piv[j*ldpiv + i];
	}
	Timer::end(Timer::DIAG_BCAST);
}
template <typename F>
void receive_pivu(int b, F *piv, int ldpiv, int root, Grid &grid, RequestStack& req)
{
	Timer::beg(Timer::DIAG_BCAST);
	req.wait_all();
	MPI_Ibcast(piv, ldpiv * b, T2MPI<F>::type, root, grid.vcomm, req.get_request());
	Timer::end(Timer::DIAG_BCAST);
}
template <typename F>
void receive_pivl(int b, F *piv, int ldpiv, int root, Grid &grid, RequestStack& req)
{
	Timer::beg(Timer::DIAG_BCAST);
	req.wait_all();
	MPI_Ibcast(piv, ldpiv * b, T2MPI<F>::type, root, grid.hcomm, req.get_request());
	Timer::end(Timer::DIAG_BCAST);
}

template <typename F>
void broadcast_left_panels(int b, const LRPanels<F> &lp, int rowstart, int nrow, Grid &grid, RequestStack& req)
{
	Timer::beg(Timer::LCOL_BCAST);
	// broadcast the left panels to row
	if(lp.is_tile)
		MPI_Ibcast(const_cast<F *>(lp(rowstart)), lp.ldp * (nrow-rowstart),
			T2MPI<F>::type, grid.col, grid.hcomm, req.get_request());
	else
		MPI_Ibcast(const_cast<F *>(lp(rowstart)), lp.get_lda() * b,
			T2MPI<F>::type, grid.col, grid.hcomm, req.get_request());
	Timer::end(Timer::LCOL_BCAST);
}
template <typename F>
void receive_left_panels(int b, LRPanels<F> &lp, int rowstart, int nrow, int root, Grid &grid, RequestStack& req)
{
	Timer::beg(Timer::LCOL_BCAST);
	lp.set_start(rowstart);
	if(lp.is_tile)
		MPI_Ibcast(const_cast<F *>(lp(rowstart)), lp.ldp * (nrow-rowstart),
			T2MPI<F>::type, root, grid.hcomm, req.get_request());
	else
		MPI_Ibcast(const_cast<F *>(lp(rowstart)), lp.get_lda() * b,
			T2MPI<F>::type, root, grid.hcomm, req.get_request());
	Timer::end(Timer::LCOL_BCAST);
}
template <typename F>
void broadcast_right_panels(int /*b*/, LRPanels<F> &rp, int colstart, int ncol, Grid &grid, RequestStack& req)
{
	assert(rp.is_tile);
	Timer::beg(Timer::RROW_BCAST);
	MPI_Ibcast(const_cast<F *>(rp(colstart)), rp.ldp * (ncol-colstart),
		T2MPI<F>::type, grid.row, grid.vcomm, req.get_request());
	Timer::end(Timer::RROW_BCAST);
}
template <typename F>
void receive_right_panels(int /*b*/, LRPanels<F> &rp, int colstart, int ncol, int root, Grid &grid, RequestStack& req)
{
	assert(rp.is_tile);
	Timer::beg(Timer::RROW_BCAST);
	rp.set_start(colstart);
	MPI_Ibcast(const_cast<F *>(rp(colstart)), rp.ldp * (ncol-colstart),
		T2MPI<F>::type, root, grid.vcomm, req.get_request());
	Timer::end(Timer::RROW_BCAST);
}

template <typename F>
void update_left_panels(const F *u, int ldu, Panels<F> &p, int rowstart, int col)
{
	if(rowstart == p.nprow) return;
	Timer::beg(Timer::TRSM_L);
	if(p.is_tile){
		for (int i = rowstart; i < p.nprow; ++i) {
			trsmR(p.b, p.b, u, ldu, p(i, col), p.lda);
		}
	}
	else {
		trsmR(p.b*(p.nprow-rowstart), p.b, u, ldu, p(rowstart, col), p.lda);
	}
	Timer::end(Timer::TRSM_L, false, 1ll*p.b*(p.nprow-rowstart)*p.b*p.b);
}
template <typename F>
void update_right_panels(F const *l, int ldl, Panels<F> &p, int row, int colstart)
{
	if(colstart == p.npcol) return;
	Timer::beg(Timer::TRSM_R);
	if(p.is_tile){
		for (int j = colstart; j < p.npcol; ++j) {
			trsmL(p.b, p.b, l, ldl, p(row, j), p.lda);
		}
	}
	else
		trsmL(p.b, p.b*(p.npcol-colstart), l, ldl, p(row, colstart), p.lda);
	Timer::end(Timer::TRSM_R, false, 1ll*p.b*p.b*p.b*(p.npcol-colstart));
}

template <typename FHigh, typename FLow>
void convert_panel_impl(int b, FHigh scale, FHigh const *__restrict__ a, size_t lda, FLow *__restrict__ to, size_t ldb)
{
	// we may change the type of a and b
	#pragma omp parallel for
	for (int j = 0; j < b; ++j)
		for (int i = 0; i < b; ++i)
			to[j * ldb + i] = static_cast<FLow>(scale * a[j * lda + i]);
}
template <typename FHigh, typename FLow>
void convert_left_panels(Panels<FHigh> const &p, FHigh scale, int rowstart, int col, LRPanels<FLow> &lp)
{
	Timer::beg(Timer::CONV_L);
	lp.set_start(rowstart);
	if(p.is_tile || lp.is_tile){
		for (int i = rowstart; i < p.nprow; ++i) {
			convert_panel_impl(p.b, scale, p(i, col), p.lda, lp(i), lp.get_lda());
		}
	}
	else {
		int b = p.b;
		int nprow = p.nprow;
		size_t plda = p.lda;
		size_t lplda = lp.get_lda();
		FHigh const * pdata = p(rowstart, col);
		FLow * lpdata = lp(rowstart);
		#pragma omp parallel for
		for(int j=0; j<b; ++j){
			for(int i=0; i<b*(nprow-rowstart); ++i){
				lpdata[j*lplda + i] = static_cast<FLow>(scale * pdata[j*plda + i]);
			}
		}
	}
	Timer::end(Timer::CONV_L, false, 1ll*p.b*p.b*(p.nprow-rowstart));
}
template <typename FHigh, typename FLow>
void convert_right_panels(Panels<FHigh> const &p, FHigh scale, int row, int colstart, LRPanels<FLow> &rp)
{
	Timer::beg(Timer::CONV_R);
	rp.set_start(colstart);
	if(p.is_tile){
		for (int j = colstart; j < p.npcol; ++j) {
			convert_panel_impl(p.b, scale, p(row, j), p.lda, rp(j), rp.get_lda());
		}
	}
	else{
		int b = p.b;
		int npcol = p.npcol;
		size_t plda = p.lda;
		size_t rplda = rp.get_lda();
		FHigh const * pdata = p(row, colstart);
		FLow * rpdata = rp(colstart);
		#pragma omp parallel for
		for(int j=0; j<b*(npcol-colstart); ++j){
			for(int i=0; i<b; ++i){
				rpdata[j*rplda + i] = static_cast<FLow>(scale * pdata[j*plda + i]);
			}
		}
	}
	Timer::end(Timer::CONV_R, false, 1ll*p.b*(p.npcol-colstart)*p.b);
}

#ifdef __aarch64__
extern "C" void conv_scale_copy(fp16 *, const float *, int, const float);
void convert_left_panels(Panels<float> const &p, float scale, int rowstart, int col, LRPanels<fp16> &lp)
{
	Timer::beg(Timer::CONV_L);

	lp.set_start(rowstart);
	if(p.is_tile || lp.is_tile){
		std::abort();
	}
	else {
		int b = p.b;
		int nprow = p.nprow;
		size_t plda = p.lda;
		size_t lplda = lp.get_lda();
		float const * pdata = p(rowstart, col);
		fp16 * lpdata = lp(rowstart);
		#pragma omp parallel for
		for(int j=0; j<b; ++j){
			//for(int i=0; i<b*(nprow-rowstart); ++i){
			//	lpdata[j*lplda + i] = static_cast<FLow>(scale * pdata[j*plda + i]);
			//}
			conv_scale_copy(&lpdata[j*lplda], &pdata[j*plda], b*(nprow-rowstart), scale);
		}
	}
	Timer::end(Timer::CONV_L, false, 1ll*p.b*p.b*(p.nprow-rowstart));
}

void convert_right_panels(Panels<float> const &p, float scale, int row, int colstart, LRPanels<fp16> &rp)
{
	Timer::beg(Timer::CONV_R);

	rp.set_start(colstart);
	if(p.is_tile){
		std::abort();
	}
	else{
		int b = p.b;
		int npcol = p.npcol;
		size_t plda = p.lda;
		size_t rplda = rp.get_lda();
		float const * pdata = p(row, colstart);
		fp16 * rpdata = rp(colstart);
		#pragma omp parallel for
		for(int j=0; j<b*(npcol-colstart); ++j){
			// for(int i=0; i<b; ++i){
			// 	rpdata[j*rplda + i] = static_cast<FLow>(pdata[j*plda + i]);
			// }
			conv_scale_copy(&rpdata[j*rplda], &pdata[j*plda], b, scale);
		}
	}
	Timer::end(Timer::CONV_R, false, 1ll*p.b*(p.npcol-colstart)*p.b);
}
#endif

#ifdef HGEMM_PACK
static void convert_left_panels_pack(Panels<float> const &p, float scale, int rowstart, int col, LRPanels<fp16> &lp)
{
	Timer::beg(Timer::CONV_L);
	lp.set_start(rowstart);
	pack_convert_a(p.b*(p.nprow-rowstart), p.b, scale, p(rowstart, col), p.lda, lp.data());
	Timer::end(Timer::CONV_L, false, 1ll*p.b*p.b*(p.nprow-rowstart));
}

static void convert_right_panels_pack(Panels<float> const &p, float scale, int row, int colstart, LRPanels<fp16> &rp)
{
	Timer::beg(Timer::CONV_R);
	rp.set_start(colstart);
	pack_convert_b(p.b*(p.npcol-colstart), p.b, scale, p(row,colstart), p.lda, rp.data());
	Timer::end(Timer::CONV_R, false, 1ll*p.b*(p.npcol-colstart)*p.b);
}
#else
static void convert_left_panels_pack(Panels<float> const &p, float scale, int rowstart, int col, LRPanels<fp16> &lp)
{
	std::abort();
}

static void convert_right_panels_pack(Panels<float> const &p, float scale, int row, int colstart, LRPanels<fp16> &rp)
{
	std::abort();
}
#endif
template<typename F, typename H>
void convert_left_panels_pack(Panels<F> const &p, F scale, int rowstart, int col, LRPanels<H> &lp)
{
	std::abort();
}

template<typename F, typename H>
void convert_right_panels_pack(Panels<F> const &p, F scale, int row, int colstart, LRPanels<H> &rp)
{
	std::abort();
}

template <typename FHigh, typename FLow, template<class> class Matgen, int du, bool dd>
void panel_lu(Panels<FHigh> &p, LRPanels<FLow> lrpanels[4], Matgen<FHigh>& mg, FHigh *piv, size_t ldpiv, Grid &grid, bool warmup=false)
{
	// easy implemnetation of LU decomp. for description purpose
	typedef LazyInitializer<FLow, du, dd> LI;
	typedef SchurUpdator<FHigh, FLow, dd> SU;
	int const nb = p.nblocks;
	int const b = p.b;
	size_t const lda = p.lda;
	int const nprow = p.nprow;
	int const npcol = p.npcol;
	LRPanels<FLow>& lp = lrpanels[0];
	LRPanels<FLow>& rp = lrpanels[1];
	RequestStack req(8);
	FHigh scalea = static_cast<FHigh>(1);
	FHigh scaleb = static_cast<FHigh>(1);
	FHigh downscale = static_cast<FHigh>(1./((double)scalea*scaleb));
	FHigh* buf = piv + ldpiv * b;
	int kend = warmup? 20 : nb;
	if(kend > nb) kend = nb;

	for (int k = 0; k < kend; ++k) {
		// position of the panels to decomp in process grid
		int const rootrow = k % grid.nrow;
		int const rootcol = k % grid.ncol;
		// position of the panels to decomp in local matrix
		int i = k / grid.nrow + (rootrow > grid.row ? 1 : 0);
		int j = k / grid.ncol + (rootcol > grid.col ? 1 : 0);
		if (rootrow == grid.row && rootcol == grid.col) {
			// I have a pivot panel.
			// 1) add the initial values to the partial sum (lazy init)
			FHigh *lu = p(i, j);
			LI::update_diag(mg, p, downscale, i, j, buf, ldpiv);
			// 2) lu decomp of the diagonal panel
			if(p.is_tile){
				getrf_nopiv(b, p(i, j), lda);
			}else{
				getrf_nopiv(b, p(i, j), lda, piv, ldpiv);
			}
			if (k == nb - 1) return;
			// 3) broadcast it
			broadcast_pivlu(p.is_tile, b, lu, lda, piv, ldpiv, grid, req);

			// 4) trsm row and column panels
			// 4.1) lazy init
			LI::update(mg, p, downscale, i+1, nprow, j, j+1, buf, ldpiv);
			// 4.2) trsm
			update_left_panels(lu, p.lda, p, i + 1, j);
			// 4.3) downcast from FHigh to FLow
			convert_left_panels(p, scalea, i + 1, j, lp);
			// 4.4) broadcast
			broadcast_left_panels(b, lp, i + 1, nprow, grid, req);

			// 5) same as 4
			LI::update(mg, p, downscale, i, i+1, j+1, npcol, buf, ldpiv);
			update_right_panels(lu, p.lda, p, i, j + 1);
			convert_right_panels(p, scaleb, i, j + 1, rp);
			broadcast_right_panels(b, rp, j + 1, npcol, grid, req);
			++i;
			++j;
		}
		else if (rootrow == grid.row) {
			if (k == nb - 1) return;
			// I have a right panel.
			// 1) lazy-init
			LI::update(mg, p, downscale, i, i+1, j, npcol, buf, ldpiv);
			// 2) get the LU factors of the diagonal panel
			receive_pivl(b, piv, ldpiv, rootcol, grid, req);
			req.wait_all();

			// 3) trsm U
			update_right_panels(piv, ldpiv, p, i, j);
			// 4) downcast from FHigh to FLow
			convert_right_panels(p, scaleb, i, j, rp);
			// 5) broadcast (send) U
			broadcast_right_panels(b, rp, j, npcol, grid, req);
			++i;
			// 6) broadast (receive) L
			receive_left_panels(b, lp, i, nprow, rootcol, grid, req);
		}
		else if (rootcol == grid.col) {
			if (k == nb - 1) return;
			// I have a left panel.
			LI::update(mg, p, downscale, i, nprow, j, j+1, buf, ldpiv);
			receive_pivu(b, piv, ldpiv, rootrow, grid, req);
			req.wait_all();

			update_left_panels(piv, ldpiv, p, i, j);
			convert_left_panels(p, scalea, i, j, lp);
			broadcast_left_panels(b, lp, i, nprow, grid, req);
			++j;
			receive_right_panels(b, rp, j, npcol, rootrow, grid, req);
		}
		else {
			if (k == nb - 1) return;
			// broadcast (receive) L and U panels
			receive_left_panels(b, lp, i, nprow, rootcol, grid, req);
			receive_right_panels(b, rp, j, npcol, rootrow, grid, req);
		}
		req.wait_all();
		// GEMM
		SU::update(p, lp, rp, i, p.nprow, j, p.npcol);
	}
}

template <typename FHigh, typename FLow, template<class> class Matgen, int du, bool dd>
void panel_lu_async(Panels<FHigh> &p, LRPanels<FLow> lrpanels[4], Matgen<FHigh>&mg, FHigh *piv, size_t ldpiv, Grid &grid, bool warmup=false)
{
	// same as panel_lu but with look-ahead computation for better communication hiding
	typedef LazyInitializer<FLow, du, dd> LI;
	typedef SchurUpdator<FHigh, FLow, dd> SU;
	int const nb = p.nblocks;
	int const b = p.b;
	int const n = nb * b;
	size_t const lda = p.lda;
	int const nprow = p.nprow;
	int const npcol = p.npcol;
	FHigh scalea = du ? static_cast<FHigh>(mg.scalea): static_cast<FHigh>(1);
	FHigh scaleb = du ? static_cast<FHigh>(mg.scaleb): static_cast<FHigh>(1);
	FHigh downscale = static_cast<FHigh>(1./((double)scalea*scaleb));
	FHigh* buf = piv + ldpiv * b;

	LRPanels<FLow> *lprev = &lrpanels[0], *rprev = &lrpanels[1], *lnext = &lrpanels[2], *rnext = &lrpanels[3];
	RequestStack lrreq(8);
	RequestStack pivreq(8);

	int schur_row = 0;
	int schur_col = 0;
	int64_t diag_lu_comp = b*(b*(4ll*b-3ll)+5ll)/6ll;
	int kend = warmup? 20 : nb;
	if(kend > nb) kend = nb;

	{
		// look ahead
		// same as above, but leaves gemm to next iteration
		if (0 == grid.row && 0 == grid.col) {
			// I have the pivot panel.

			FHigh *lu = p(0, 0);
			LI::update_diag(mg, p, downscale, 0, 0, buf, ldpiv);
			Timer::beg(Timer::DIAG_LU);
			if(p.is_tile){ 
				getrf_nopiv(b, p(0, 0), lda);
			}else{
				getrf_nopiv(b, p(0, 0), lda, piv, ldpiv);
			}
			Timer::end(Timer::DIAG_LU, false, diag_lu_comp);

			broadcast_pivlu(p.is_tile, b, lu, lda, piv, ldpiv, grid, pivreq);

			LI::update(mg, p, downscale, 1, nprow, 0, 1, buf, ldpiv);
			update_left_panels(lu, lda, p, 1, 0);
			convert_left_panels(p, scalea, 1, 0, *lprev);
			broadcast_left_panels(b, *lprev, 1, nprow, grid, lrreq);

			LI::update(mg, p, downscale, 0, 1, 1, npcol, buf, ldpiv);
			update_right_panels(lu, lda, p, 0, 1);
			convert_right_panels(p, scaleb, 0, 1, *rprev);
			broadcast_right_panels(b, *rprev, 1, npcol, grid, lrreq);
			schur_row = 1;
			schur_col = 1;
		}
		else if (0 == grid.row) {
			// I have the right panel.
			receive_pivl(b, piv, ldpiv, 0, grid, pivreq);
			LI::update(mg, p, downscale, 0, 1, 0, npcol, buf, ldpiv);
			pivreq.wait_all(Timer::DIAG_BCAST);

			receive_left_panels(b, *lprev, 1, nprow, 0, grid, lrreq);

			update_right_panels(piv, ldpiv, p, 0, 0);
			convert_right_panels(p, scaleb, 0, 0, *rprev);
			broadcast_right_panels(b, *rprev, 0, npcol, grid, lrreq);
			schur_row = 1;
		}
		else if (0 == grid.col) {
			// I have the left panel.
			receive_pivu(b, piv, ldpiv, 0, grid, pivreq);
			LI::update(mg, p, downscale, 0, nprow, 0, 1, buf, ldpiv);
			pivreq.wait_all(Timer::DIAG_BCAST);

			receive_right_panels(b, *rprev, 1, npcol, 0, grid, lrreq);

			update_left_panels(piv, ldpiv, p, 0, 0);
			convert_left_panels(p, scalea, 0, 0, *lprev);
			broadcast_left_panels(b, *lprev, 0, nprow, grid, lrreq);
			schur_col = 1;
		}
		else {
			receive_left_panels(b, *lprev, 0, nprow, 0, grid, lrreq);
			receive_right_panels(b, *rprev, 0, npcol, 0, grid, lrreq);
		}
		if(0==grid.row && 0==grid.col) pivreq.wait_all(Timer::DIAG_BCAST);
		lrreq.wait_all(Timer::WAIT);
	}

	for (int k = 1; k < kend; ++k) {
		// printf("k=%d, wtime=%f\n", k, Timer::put(Timer::MISC));
		// GEMM of the last iteration are leaving
		int const rootrow = k % grid.nrow;
		int const rootcol = k % grid.ncol;
		int i = k / grid.nrow + (rootrow > grid.row ? 1 : 0);
		int j = k / grid.ncol + (rootcol > grid.col ? 1 : 0);
		if (rootrow == grid.row && rootcol == grid.col) {
			// I have the pivot panel.
			// do GEMM for the diagonal block
			SU::update(p, *lprev, *rprev, i, i+1, j, j+1);
			LI::update_diag(mg, p, downscale, i, j, buf, ldpiv);
			FHigh *lu = p(i, j);

			Timer::beg(Timer::DIAG_LU);
			if(p.is_tile){
				getrf_nopiv(b, p(i, j), lda);
			}else{
				getrf_nopiv(b, p(i, j), lda, piv, ldpiv);
			}
			Timer::end(Timer::DIAG_LU, false, diag_lu_comp);

			broadcast_pivlu(p.is_tile, b, lu, lda, piv, ldpiv, grid, pivreq);

			// do GEMM of L
			SU::update(p, *lprev, *rprev, i+1, nprow, j, j+1);
			LI::update(mg, p, downscale, i+1, nprow, j, j+1, buf, ldpiv);
			update_left_panels(lu, lda, p, i + 1, j);
			convert_left_panels(p, scalea, i + 1, j, *lnext);
			broadcast_left_panels(b, *lnext, i + 1, nprow, grid, lrreq);

			// do GEMM of U
			SU::update(p, *lprev, *rprev, i, i+1, j+1, npcol);
			LI::update(mg, p, downscale, i, i+1, j+1, npcol, buf, ldpiv);
			update_right_panels(lu, lda, p, i, j + 1);
			convert_right_panels(p, scaleb, i, j + 1, *rnext);
			broadcast_right_panels(b, *rnext, j + 1, npcol, grid, lrreq);

			++schur_row;
			++schur_col;
			++i;
			++j;
		}
		else if (rootrow == grid.row) {
			// receive LU factors of the diagonal block
			receive_pivl(b, piv, ldpiv, rootcol, grid, pivreq);
			// GEMM
			SU::update(p, *lprev, *rprev, i, i+1, j, npcol);
			++schur_row;
			LI::update(mg, p, downscale, i, i+1, j, npcol, buf, ldpiv);
			while(!pivreq.test_all(Timer::DIAG_BCAST)){
				// progressive communication
				if(schur_row == nprow) {
					pivreq.wait_all(Timer::DIAG_BCAST);
					break;
				}
				SU::update(p, *lprev, *rprev, schur_row, schur_row+1, schur_col, npcol);
				++schur_row;
			}

			receive_left_panels(b, *lnext, i+1, nprow, rootcol, grid, lrreq);

			update_right_panels(piv, ldpiv, p, i, j);
			convert_right_panels(p, scaleb, i, j, *rnext);
			broadcast_right_panels(b, *rnext, j, npcol, grid, lrreq);
			++i;
		}
		else if (rootcol == grid.col) {
			receive_pivu(b, piv, ldpiv, rootrow, grid, pivreq);
			SU::update(p, *lprev, *rprev, i, nprow, j, j+1);
			++schur_col;
			LI::update(mg, p, downscale, i, nprow, j, j+1, buf, ldpiv);
			while(!pivreq.test_all(Timer::DIAG_BCAST)){
				if(schur_row == nprow) {
					pivreq.wait_all(Timer::DIAG_BCAST);
					break;
				}
				SU::update(p, *lprev, *rprev, schur_row, schur_row+1, schur_col, npcol);
				++schur_row;
			}

			receive_right_panels(b, *rnext, j+1, npcol, rootrow, grid, lrreq);

			update_left_panels(piv, ldpiv, p, i, j);
			convert_left_panels(p, scalea, i, j, *lnext);
			broadcast_left_panels(b, *lnext, i, nprow, grid, lrreq);
			++j;
		}
		else {
			receive_left_panels(b, *lnext, i, nprow, rootcol, grid, lrreq);
			receive_right_panels(b, *rnext, j, npcol, rootrow, grid, lrreq);
		}

		// GEMM for last iteration
		SU::update(p, *lprev, *rprev, schur_row, nprow, schur_col, npcol);
		LRPanels<FLow>* t;
		t = lprev; lprev = lnext; lnext = t;
		t = rprev; rprev = rnext; rnext = t;
		schur_row = i;
		schur_col = j;

		if(rootrow == grid.row && rootcol == grid.col) pivreq.wait_all(Timer::DIAG_BCAST);
		lrreq.wait_all(Timer::WAIT);
	}
	if(warmup){
		pivreq.wait_all();
		lrreq.wait_all();
	}
}

template<typename FHigh, typename FLow, typename RDMACom>
struct RDMAPanelLU {
	// Tofu RDMA based implementation
	// Algorithm is same as panel_lu_async. We need misc. works to do rdma.
	Panels<FHigh>& p;
	LRPanels<FLow>* lrpanels;
	FHigh* piv;
	size_t ldpiv;
	Grid& grid;
	RDMACom& lcom;
	RDMACom& rcom;
	int nbuf;
	int lphandle, rphandle;
	int* hndls;

	double start_time;

	RDMAPanelLU(Panels<FHigh>& p, LRPanels<FLow>* lrpanels, FHigh* piv, size_t ldpiv,
		Grid& grid, RDMACom& lcom, RDMACom& rcom, int nbuf=2):
		p(p), lrpanels(lrpanels), piv(piv), ldpiv(ldpiv), grid(grid), lcom(lcom), rcom(rcom), nbuf(nbuf)
	{
		// piv(ldpiv, 3*b) is a working space, ldpiv >= b

		// register addresses for rdma
		assert(nbuf>=2);
		assert(lrpanels[1].is_tile || lrpanels[1].is_pack);
		assert(lrpanels[3].is_tile || lrpanels[3].is_pack);
		lphandle = lcom.get_handle(reinterpret_cast<char*>(piv + ldpiv*p.b), sizeof(FHigh) * ldpiv * p.b);
		rphandle = rcom.get_handle(reinterpret_cast<char*>(piv), sizeof(FHigh) * ldpiv * p.b);
		hndls = (int*)malloc(sizeof(int)*2*nbuf);
		for(int i=0; i<2*nbuf; i+=2){
			hndls[i] = lcom.get_handle(lrpanels[i].p, lsize(0,p.nprow,p.b,&lrpanels[i]));
			hndls[i+1] = rcom.get_handle(lrpanels[i+1].p, rsize(0,p.npcol,p.b,&lrpanels[i+1]));
		}
	}
	~RDMAPanelLU(){
		free(hndls);
	}

	template<typename F>
	static inline size_t lsize(int rowstart, int nprow, int b, LRPanels<F> const* lp) {
		if(lp->is_tile) return sizeof(F) * lp->ldp * (nprow-rowstart>0?nprow-rowstart:0);
		else return sizeof(F) * lp->get_lda() * b;
	}
	template<typename F>
	static inline size_t rsize(int colstart, int npcol, int b, LRPanels<F> const* rp) {
		return sizeof(F) * rp->ldp * (npcol-colstart>0?npcol-colstart:0);
	}
	#ifdef HGEMM_PACK
	static inline size_t lsize(int rowstart, int nprow, int b, LRPanels<fp16> const* lp) {
		if(lp->is_pack) {
			int m = (nprow-rowstart>0?nprow-rowstart:0) * b;
			m = (m+HGEMM_PACK_MUNIT-1)/HGEMM_PACK_MUNIT*HGEMM_PACK_MUNIT;
			return sizeof(fp16) * m * b;
		}
		else if(lp->is_tile) return sizeof(fp16) * lp->ldp * (nprow-rowstart>0?nprow-rowstart:0);
		else return sizeof(fp16) * lp->get_lda() * b;
	}
	static inline size_t rsize(int colstart, int npcol, int b, LRPanels<fp16> const* rp) {
		if(rp->is_pack) {
			int n = (npcol-colstart>0?npcol-colstart:0) * b;
			n = (n+HGEMM_PACK_NUNIT-1)/HGEMM_PACK_NUNIT*HGEMM_PACK_NUNIT;
			return sizeof(fp16) * n * b;
		}
		else return sizeof(fp16) * rp->ldp * (npcol-colstart>0?npcol-colstart:0);
	}
	#endif

	template<typename Matgen, int du, bool dd, bool pack>
	void run(Matgen& mg, bool warmup=false){
		typedef LazyInitializer<FLow, du, dd> LI; // for static-dispatching method for lazy-init
		BackBuffer<FHigh, FLow, du> bb(p.nprow);
		GemmControl<FHigh, FLow, dd, pack> mmcon(p.b, p.nprow, p.npcol);
		int const nb = p.nblocks;
		int const b = p.b;
		int const n = nb * b;
		size_t const lda = p.lda;
		int const nprow = p.nprow;
		int const npcol = p.npcol;

		int const epoch_size = p.epoch_size;

		FHigh scalea = static_cast<FHigh>(du ? mg.scalea: 1.); // scaling for L panel
		FHigh scaleb = static_cast<FHigh>(du ? mg.scaleb: 1.); // scaling for R panel
		FHigh downscale = static_cast<FHigh>(1./(scalea*scaleb));
		FHigh* buf = piv + 2 * ldpiv * b;


		int i=0, j=0;
		int64_t diag_lu_comp = b*(b*(4ll*b-3ll)+5ll)/6ll;

		int kend = warmup ? grid.nrow+2: nb;
		if(kend > nb) kend = nb;

		for (int k = 0; k < kend; ++k) {
			if(epoch_size){ // please set it 0 to disable this output
				if(0 == (k*b) % epoch_size){
					if(grid.row==0 && grid.col==0){
						int cur = (k*b) / epoch_size;
						int tot = n     / epoch_size;
						double elapsed = MPI_Wtime() - start_time;
						auto pow3 = [](double x){ return x*x*x; };
						int nrem = n - k*b;
						double flop = 2./3. * (pow3(n) - pow3(nrem));
						double Pflops = 1.e-15 * flop / elapsed;
						std::printf("!epoch %d/%d: elapsed=%f, %f Pflops (estimate)\n", 
								cur, tot, elapsed, Pflops);
						std::fflush(stdout);
					}
				}
			}

			int prev = k%nbuf;
			int next = (k+1)%nbuf;
			LRPanels<FLow> *lprev = &lrpanels[prev*2], *rprev = &lrpanels[prev*2+1],
				*lnext = &lrpanels[next*2], *rnext = &lrpanels[next*2+1];
			int lnexthndl = hndls[next*2], rnexthndl = hndls[next*2+1];

			// send_sync: tell nighbors how much I progressed
			lcom.send_sync(k);
			rcom.send_sync(k);

			int const rootrow = k % grid.nrow;
			int const rootcol = k % grid.ncol;

			// gemm control
			mmcon.set(k!=0, i, j, p, *lprev, *rprev);
			if (rootrow == grid.row && rootcol == grid.col) {
				// I have the pivot panel.
				lnext->set_start(i+1);
				rnext->set_start(j+1);

				mmcon.update_11();
				LI::update_diag(mg, p, downscale, i, j, buf, ldpiv);

				Timer::beg(Timer::DIAG_LU);
				getrf_nopiv(b, p(i, j), lda, piv, ldpiv);
				FHigh* lu = p(i, j);
				#pragma omp parallel for
				for(int y=0; y<b; ++y)
					for(int x=0; x<b; ++x)
						lu[y*lda + x] = piv[(y+b)*ldpiv + x] = piv[y*ldpiv + x];
				Timer::end(Timer::DIAG_LU, false, diag_lu_comp);
				
				// invoke start the broadcast communication
				// it wait for neighbors to progress at (k-grid.ncol+1) or (k-grid.nrow+1) step before communication
				rcom.invoke(k-grid.ncol+1, rphandle, 0, sizeof(FHigh)*ldpiv*b, rootrow);
				lcom.invoke(k-grid.nrow+1, lphandle, 0, sizeof(FHigh)*ldpiv*b, rootcol);

				mmcon.update_col2_n();
				LI::update(mg, p, downscale, i+1, nprow, j, j+1, buf, ldpiv);
				rcom.progress();
				lcom.progress();
				update_left_panels(piv, ldpiv, p, i + 1, j);
				if(pack) convert_left_panels_pack(p, scalea, i+1, j, *lnext);
				else convert_left_panels(p, scalea, i + 1, j, *lnext);
				lcom.invoke(k-nbuf+2, lnexthndl, 0, lsize(i+1,nprow,b,lnext), rootcol);

				mmcon.update_row2_n();
				LI::update(mg, p, downscale, i, i+1, j+1, npcol, buf, ldpiv);
				bb.pop(i);
				rcom.progress();
				lcom.progress();
				update_right_panels(piv, ldpiv, p, i, j + 1);
				if(pack) convert_right_panels_pack(p, scaleb, i, j + 1, *rnext);
				else convert_right_panels(p, scaleb, i, j + 1, *rnext);
				rcom.invoke(k-nbuf+2, rnexthndl, 0, rsize(j+1,npcol,b,rnext), rootrow);

				mmcon.skip_11();
				++i;
				++j;
			}
			else if (rootrow == grid.row) {
				lnext->set_start(i+1);
				rnext->set_start(j);
				lcom.invoke(k-grid.nrow+1, 0, lphandle, sizeof(FHigh)*ldpiv*b, rootcol);
				mmcon.update_row1_n();
				LI::update(mg, p, downscale, i, i+1, j, npcol, buf, ldpiv);
				bb.pop(i);

				mmcon.update_until(lcom);
				lcom.wait();
				lcom.invoke(k-nbuf+2, lnexthndl, 0, lsize(i+1,nprow,b,lnext), rootcol);

				update_right_panels(piv+ldpiv*b, ldpiv, p, i, j);
				if(pack) convert_right_panels_pack(p, scaleb, i, j, *rnext);
				else convert_right_panels(p, scaleb, i, j, *rnext);
				rcom.invoke(k-nbuf+2, rnexthndl, 0, rsize(j,npcol,b,rnext), rootrow);
				++i;
			}
			else if (rootcol == grid.col) {
				lnext->set_start(i);
				rnext->set_start(j+1);
				rcom.invoke(k-grid.ncol+1, rphandle, 0, sizeof(FHigh)*ldpiv*b, rootrow);
				mmcon.update_col1_n();
				LI::update(mg, p, downscale, i, nprow, j, j+1, buf, ldpiv);

				mmcon.update_until(rcom);
				rcom.wait();
				rcom.invoke(k-nbuf+2, rnexthndl, 0, rsize(j+1,npcol,b,rnext), rootrow);

				update_left_panels(piv, ldpiv, p, i, j);
				if(pack) convert_left_panels_pack(p, scalea, i, j, *lnext);
				else convert_left_panels(p, scalea, i, j, *lnext);
				lcom.invoke(k-nbuf+2, lnexthndl, 0, lsize(i,nprow,b,lnext), rootcol);
				++j;
			}
			else {
				lnext->set_start(i);
				rnext->set_start(j);
				lcom.invoke(k-nbuf+2, lnexthndl, 0, lsize(i,nprow,b,lnext), rootcol);
				rcom.invoke(k-nbuf+2, rnexthndl, 0, rsize(j,npcol,b,rnext), rootrow);
			}

			mmcon.update_rest(lcom, rcom);
			bb.write_back(k, p, i, j);
			wait_all(lcom, rcom);
		}
		if(warmup){
			MPI_Barrier(grid.commworld);
		}
	}

};

#endif
