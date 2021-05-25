#ifndef SCHUR_UPDATOR_HPP
#define SCHUR_UPDATOR_HPP
#include "panel.hpp"
#include "kernels/kernel.h"
#include "fp16sim.hpp"
#include "timer.hpp"
#include "chain_schedule.hpp"

// gemm for alpha=-1, beta=1
extern "C" void dgemm_(...);
extern "C" void sgemm_(...);
inline void gemmschur(int n, int m, int k, double const *a, int lda, double const *b, int ldb, double *c, int ldc)
{
	double one = 1.;
	double mone = -1.;
	dgemm_("N", "N", &n, &m, &k, &mone, a, &lda, b, &ldb, &one, c, &ldc);
}
inline void gemmschur(int n, int m, int k, float const *a, int lda, float const *b, int ldb, float *c, int ldc)
{
	float one = 1.;
	float mone = -1.;
	sgemm_("N", "N", &n, &m, &k, &mone, a, &lda, b, &ldb, &one, c, &ldc);
}
inline void gemmschur(int n, int m, int k, fp16 const *a, int lda, fp16 const *b, int ldb, fp16 *c, int ldc)
{
	hgemm(n, m, k, -1.f, a, lda, b, ldb, 1.f, c, ldc);
}
inline void gemmschur(int n, int m, int k, fp16 const *a, int lda, fp16 const *b, int ldb, float *c, int ldc)
{
	shgemm(n, m, k, -1.f, a, lda, b, ldb, 1.f, c, ldc);
}

inline void gemmschur(int n, int m, int k, float const *a, int lda, float const *b, int ldb, double *c, int ldc)
{
	// XXX dsgemm, only for test
	for(int j=0; j<m; ++j){
		for(int i=0; i<n; ++i){
			double t = 0.;
			for(int l=0; l<k; ++l){
				double da = static_cast<double>(a[l*lda + i]);
				double db = static_cast<double>(b[j*ldb + l]);
				t += da * db;
			}
			c[j*ldc + i] -= t;
		}
	}
}

// wrapper for gemm. 
template<typename FHigh, typename FLow, bool double_decker>
struct SchurUpdator {
	struct FakeComm {
		void progress() const{} // do nothing
		bool detached() const { return true; } // fake
	};
	typedef DDAdaptor<FHigh, FLow, double_decker> DDA;
	static void update( Panels<FHigh> &p, LRPanels<FLow> const &lp, LRPanels<FLow> const &rp, int rowstart, int rowend, int colstart, int colend) {
		FakeComm lcom, rcom;
		SchurUpdator::update(p, lp, rp, rowstart, rowend, colstart, colend, lcom, rcom);
	}

	template<typename Comm>
	static void update( Panels<FHigh> &p, LRPanels<FLow> const &lp, LRPanels<FLow> const &rp, int rowstart, int rowend, int colstart, int colend, Comm& lcs, Comm& rcs) {
		// do gemm and progress communication
		// p is matrix , lp is left pane, rp is right panel
		// rowstart, rowend, colstart, colend: region to update
		// lcs, rcs: communicators
		if(rowend==rowstart || colend==colstart) return;
		if(lp.is_tile || p.is_tile){
			Timer::beg(Timer::GEMM_UPDATE);
			for (int j = colstart; j < colend; ++j) {
				for (int i = rowstart; i < rowend; ++i) {
					gemmschur(p.b, p.b, p.b, lp(i), lp.get_lda(), rp(j), rp.get_lda(), DDA::get_deck(p, i, j), DDA::get_ldl(p));
					lcs.progress();
					rcs.progress();
				}
			}
			Timer::end(Timer::GEMM_UPDATE, false, 2ull*p.b*p.b*p.b*(rowend-rowstart)*(colend-colstart));
		}
		else {
			int stride = p.b<1000 && rowend-rowstart>10 ? 2: 1;
			int cstride = p.b<500 ? 50 : (p.b<1000 ? 20: 10);
			while(rowstart + stride <= rowend && (!lcs.detached() || !rcs.detached())){
				for(int c=colstart; c<colend; c+=cstride){
					int csize = c+cstride < colend ? cstride: colend - c;
					Timer::beg(Timer::GEMM_UPDATE);
					gemmschur(p.b*stride, p.b*csize, p.b,
						lp(rowstart), lp.get_lda(), rp(c), rp.get_lda(),
						DDA::get_deck(p, rowstart, c), DDA::get_ldl(p));
					Timer::end(Timer::GEMM_UPDATE, false, 2ull*p.b*p.b*p.b*stride*csize);
					lcs.progress();
					rcs.progress();
				}
				rowstart += stride;
			}
			if(rowstart < rowend){
				Timer::beg(Timer::GEMM_UPDATE);
				gemmschur(p.b*(rowend-rowstart), p.b*(colend-colstart), p.b,
					lp(rowstart), lp.get_lda(), rp(colstart), rp.get_lda(),
					DDA::get_deck(p, rowstart, colstart), DDA::get_ldl(p));
				Timer::end(Timer::GEMM_UPDATE, false, 2ull*p.b*p.b*p.b*(rowend-rowstart)*(colend-colstart));
			}
		}
	}

};

template<typename FHigh, typename FLow, bool double_decker, bool pack>
struct GemmControl {
	// GemmControl is a manager for gemm region.
	// this hides complication of the progress communication behide gemm
	// this class is stateful. call methods in appropriate order.
	// ex)
	// GemmControl<...> gc(block_size, nprow, npcol);
	// for(int k=0; k<nblock; ++k){
	//   rowstart = calc_myrow(k);
	//   colstart = calc_mycol(k);
	//   gc.set(true, rowstart, colstart, p, lp, rp);
	//   if(is_diagonal){
	//     // gemm diagonal part
	//     gc.updatc_11();
	//     // do something
	//     // gemm left panel
	//     gc.update_col2_n();
	//     // do somethiing
	//     // gemm right panel
	//     gc.update_row2_n();
	//     // do something
	//     gc.skip_11()
	//   }
	//   else if(is_left_panel){
	//     gc.update_col1_n();
	//     // do something
	//     gc.update_untill(comm); // progress
	//     // do something
	//   }
	//   else if(is_right_panel){
	//     gc.update_row1_n();
	//     // do something
	//     gc.update_untill(comm); // progress
	//     // do something
	//   }
	//   else {
	//     // do something
	//   }
	//   gc.update_rest();
	// }
	GemmControl(int b, int nprow, int npcol){}
	void set(bool _do_gemm, int rowstart, int colstart, Panels<FHigh> &p, LRPanels<FLow> const& lp, LRPanels<FLow> const &rp) {
		assert(false);
	}
	// gemm part of panels. Call these 4 methods in this order
	//      c
	//      |
	//    |*########|
	// r--|$++++++++|
	//    |$++++++++|
	//    |$++++++++|
	// update_11()  update *
	// update_col2_n() update $
	// update_row2_n() update #
	// skip_11() move r+=1 and c+=1
	// update_col1_n() update * and $, and move c += 1
	// update_row1_n() update * and 3, and move r += 1
	// update_until(com) repeat update_row1_n() until com is ready
	// update_rest(com) update all the part simultaneously progressing the communication
	void update_11(){}
	void update_col2_n(){}
	void update_row2_n(){}
	void skip_11(){}
	void update_col1_n(){}
	void update_row1_n(){}
	template<typename Comm>
	void update_until(Comm& com) {}
	template<typename Comm>
	void update_rest(Comm& lcom, Comm& rcom) {}
};
template<typename FHigh, typename FLow, bool double_decker>
struct GemmControl<FHigh, FLow, double_decker, false> {
	typedef DDAdaptor<FHigh, FLow, double_decker> DDA;
	typedef typename DDA::FDeck FType;

	size_t r, c;
	size_t rend, cend;
	int b;
	bool do_gemm;
	FLow const* pa;
	FLow const* pb;
	FType * pc;
	size_t lda, ldb, ldc;
	GemmControl(int b, int nprow, int npcol): b(b), rend(b*nprow), cend(b*npcol) {
		r=0;
		c=0;
	}

	void set(bool _do_gemm, int rowstart, int colstart, Panels<FHigh> &p, LRPanels<FLow> const& lp, LRPanels<FLow> const &rp) {
		assert(!p.is_tile && !lp.is_tile);
		do_gemm = _do_gemm;
		if(!do_gemm) return;
		r = rowstart * b;
		c = colstart * b;
		lda = lp.get_lda();
		ldb = rp.get_lda();
		ldc = DDA::get_ldl(p);
		pa = lp.data();
		pb = rp.data();
		pc = DDA::get_deck(p, 0, 0);
	}
	// gemm part of panels. Call these 4 methods in this order
	void update_11() {
		if(!do_gemm) return;
		if(rend>r&&cend>c) gemmschur(b, b, b, pa, lda, pb, ldb, pc+r+c*ldc, ldc);
	}
	void update_col2_n() {
		if(!do_gemm) return;
		if(rend>r+b&&cend>c) gemmschur(rend-r-b, b, b, pa+b, lda, pb, ldb, pc+r+b+c*ldc, ldc);
	}
	void update_row2_n() {
		if(!do_gemm) return;
		if(rend>r&&cend>c+b) gemmschur(b, cend-c-b, b, pa, lda, pb+b*ldb, ldb, pc+r+(c+b)*ldc, ldc);
	}
	void skip_11() {
		r += b;
		c += b;
		pa += b;
		pb += b*ldb;
	}

	// gemm 1 column
	void update_col1_n() {
		if(!do_gemm) return;
		if(rend>r&&cend>c) gemmschur(rend-r, b, b, pa, lda, pb, ldb, pc+r+c*ldc, ldc);
		c += b;
		pb += b*ldb;
	}
	// gemm 1 row
	void update_row1_n() {
		if(!do_gemm) return;
		if(cend>c) gemmschur(b, cend-c, b, pa, lda, pb, ldb, pc+r+c*ldc, ldc);
		r += b;
		pa += b;
	}

	template<typename Comm>
	void update_until(Comm& com) {
		// gemm until communications are completed
		return;
		if(!do_gemm) return;
		if(cend<=c) return;
		while(rend>r && !com.test())
			update_row1_n();
	}
	template<typename Comm>
	void update_rest(Comm& lcom, Comm& rcom) {
		// gemm until communications are detached
		if(!do_gemm) return;
		if(cend<=c) return;
		while(rend>r && (!lcom.detached() || !rcom.detached())){
			update_row1_n();
			lcom.progress();
			rcom.progress();
		}
		if(rend > r){
			Timer::beg(Timer::GEMM_UPDATE);
			gemmschur(rend-r, cend-c, b, pa, lda, pb, ldb, pc+r+c*ldc, ldc);
			Timer::end(Timer::GEMM_UPDATE, false, 2ull*(rend-r)*(cend-c)*b);
		}
	}
};

#ifdef HGEMM_PACK
template<>
struct GemmControl <float, fp16, true, true>{
	// specialization of GemmControl for packed left/right panels
	// the interfaces are same, but it internally handles the alightment.
	typedef DDAdaptor<float, fp16, true> DDA;
	typedef typename DDA::FDeck FType;

	int r, c, deltarow, deltacol;
	int rend, cend;
	int b;
	bool do_gemm;
	fp16 const* pa;
	fp16 const* pb;
	FType * pc;
	size_t ldc;
	GemmControl(int b, int nprow, int npcol): b(b), rend(b*nprow), cend(b*npcol) {
		r=0;
		c=0;
		// we need to care about the alighnment for the packed strage.
		// If deltarow/col are not the multiple of the blocksize, we need to compicated addressing in the gemm kernel.
		deltarow = (b+HGEMM_PACK_MUNIT-1)/HGEMM_PACK_MUNIT*HGEMM_PACK_MUNIT;
		deltacol = (b+HGEMM_PACK_NUNIT-1)/HGEMM_PACK_NUNIT*HGEMM_PACK_NUNIT;
	}

	void set(bool _do_gemm, int rowstart, int colstart, Panels<float> &p, LRPanels<fp16> const& lp, LRPanels<fp16> const &rp) {
		assert(lp.is_pack && rp.is_pack);
		do_gemm = _do_gemm;
		if(!do_gemm) return;
		r = rowstart * b;
		c = colstart * b;
		ldc = DDA::get_ldl(p);
		pa = lp.data();
		pb = rp.data();
		pc = DDA::get_deck(p, 0, 0);
	}
	// call update_11 -> update_col2_n and update_row2_n -> skip_11 at diagonal proc
	void update_11() {
		if(!do_gemm) return;
		int rsize = r+deltarow < rend ? deltarow: rend-r;
		int csize = c+deltacol < cend ? deltacol: cend-c;
		gemmsimple(rsize, csize, pa, pb, pc+r+c*ldc);
	}
	void update_col2_n() {
		if(!do_gemm) return;
		int rsize = r+deltarow < rend ? rend-r-deltarow: 0;
		int csize = c+deltacol < cend ? deltacol: cend-c;
		gemmsimple(rsize, csize, pa+deltarow*b, pb, pc+(r+deltarow)+c*ldc);
	}
	void update_row2_n() {
		if(!do_gemm) return;
		int rsize = r+deltarow < rend ? deltarow: rend-r;
		int csize = c+deltacol < cend ? cend-c-deltacol: 0ull;
		gemmsimple(rsize, csize, pa, pb+deltacol*b, pc+r+(c+deltacol)*ldc);
	}
	void skip_11() {
		r += deltarow;
		c += deltacol;
		pa += deltarow*b;
		pb += deltacol*b;
	}

	void update_col1_n() {
		if(!do_gemm) return;
		int rsize = rend-r;
		int csize = c+deltacol < cend ? deltacol: cend-c;
		gemmsimple(rsize, csize, pa, pb, pc+r+c*ldc);
		c += deltacol;
		pb += deltacol*b;
	}
	void update_row1_n() {
		if(!do_gemm) return;
		int rsize = r+deltarow < rend ? deltarow: rend - r;
		int csize = cend - c;
		gemmsimple(rsize, csize, pa, pb, pc+r+c*ldc);
		r += deltarow;
		pa += deltarow*b;
	}
	void gemmsimple(int m, int n, fp16 const* _pa, fp16 const* _pb, fp16* _pc)
	{
		if(m<=0 || n<=0) return; // safe guard

		#if (defined(__FUJITSU) || defined(__CLANG_FUJITSU)) && defined(__ARM_FEATURE_SVE)
		// XXX use pragma (or internal interface) to config sector cache.
		// 2 ways for a, 1 way for b, and another 1 way for c
		// contact with Fujitsu and Riken to know the actual parameters
		fp16 const* pa = _pa;
		fp16 const* pb = reinterpret_cast<fp16*>(reinterpret_cast<size_t>(_pb) | 0x0200000000000000ull);
		fp16 * pc = reinterpret_cast<fp16*>(reinterpret_cast<size_t>(_pc) | 0x4100000000000000ull);
		#else
		fp16 const* pa = _pa;
		fp16 const* pb = _pb;
		fp16 * pc = _pc;
		#endif

		// hpctag
		// sector cache
		Timer::beg(Timer::GEMM_UPDATE);
		#pragma omp parallel
		{
			int id = omp_get_thread_num();
			int nt = omp_get_num_threads();
			int nr = n%HGEMM_PACK_NUNIT;
			int nb = n/HGEMM_PACK_NUNIT;
			int nn = (nb+nt-1)/nt;
			int nbegin = nn * id;
			int nend = nbegin + nn < nb ? nbegin + nn: nb;
			for(int i=0; i<m; i+=HGEMM_PACK_MB){
				fp16 const * ca = pa + i * b;
				fp16* cc = pc + i;
				int msize = m-i>=HGEMM_PACK_MB ? HGEMM_PACK_MB: m-i;
				if(nend>nbegin) hgemmpp_kernel(msize, nend-nbegin, b, ca, pb+nbegin*HGEMM_PACK_NUNIT*b, cc+nbegin*HGEMM_PACK_NUNIT*ldc, ldc);
				if(nr){
					for(int ii=HGEMM_PACK_MUNIT*id; ii<msize; ii+=HGEMM_PACK_MUNIT*nt){
						int msize2 = msize-ii>=HGEMM_PACK_MUNIT?HGEMM_PACK_MUNIT: msize-ii;
						hgemmpp_mnend(msize2, nr, b, ca+ii*b,
							pb+nb*HGEMM_PACK_NUNIT*b, cc+nb*HGEMM_PACK_NUNIT*ldc+ii, ldc);
					}
				}
			}

		}

		Timer::end(Timer::GEMM_UPDATE, false, 2ull*m*n*b);
		/*uint64_t t = Timer::end(Timer::GEMM_UPDATE, false, 2ull*m*n*b);
		double time = tick2second(t);
		printf("mm:: %5d %5d %e %.9f\n", m, n, time, 2ull*m*n*b/time*1e-12);*/
	}

	template<typename Comm>
	void update_until(Comm& com) {
		if(!do_gemm) return;
		if(cend<=c) return;
		if(r>=rend || com.test()) return;

		int n = cend - c;
		int firstr = r;
		bool flag = false;

		#if (defined(__FUJITSU) || defined(__CLANG_FUJITSU)) && defined(__ARM_FEATURE_SVE)
		// XXX use pragma (or internal interface) to config sector cache.
		// 2 ways for a, 1 way for b, and another 1 way for c
		// contact with Fujitsu and Riken to know the actual parameters
		fp16 const* ca = pa;
		fp16 const* cb = reinterpret_cast<fp16*>(reinterpret_cast<size_t>(pb) | 0x0200000000000000ull);
		fp16 * cc = reinterpret_cast<fp16*>(reinterpret_cast<size_t>(pc) | 0x4100000000000000ull);
		#else
		fp16 const* ca = pa;
		fp16 const* cb = pb;
		fp16 * cc = pc;
		#endif
		cc = cc + c*ldc + r;

		Timer::beg(Timer::GEMM_PROGRESS);
		#pragma omp parallel firstprivate(ca, cb, cc)
		{
			int id = omp_get_thread_num();
			int nt = omp_get_num_threads();
			int nr = n%HGEMM_PACK_NUNIT;
			int nb = n/HGEMM_PACK_NUNIT;
			int nn = (nb+nt-1)/nt;
			int nbegin = nn * id;
			int nend = nbegin + nn < nb ? nbegin + nn: nb;
			int myr = r;
			#pragma omp barrier
			while(myr<rend){
				int msize = myr+deltarow < rend ? deltarow: rend - myr;
				if(nend>nbegin) hgemmpp_kernel(msize, nend-nbegin, b, ca, cb+nbegin*HGEMM_PACK_NUNIT*b, cc+nbegin*HGEMM_PACK_NUNIT*ldc, ldc);
				if(nt>1 && nr && id>0){
					for(int ii=HGEMM_PACK_MUNIT*(id-1); ii<msize; ii+=HGEMM_PACK_MUNIT*(nt-1)){
						int msize2 = msize-ii>=HGEMM_PACK_MUNIT?HGEMM_PACK_MUNIT: msize-ii;
						hgemmpp_mnend(msize2, nr, b, ca+ii*b,
							cb+nb*HGEMM_PACK_NUNIT*b, cc+nb*HGEMM_PACK_NUNIT*ldc+ii, ldc);
					}
				}
				if(nt==1 && nr){
					for(int ii=0; ii<msize; ii+=HGEMM_PACK_MUNIT){
						int msize2 = msize-ii>=HGEMM_PACK_MUNIT?HGEMM_PACK_MUNIT: msize-ii;
						hgemmpp_mnend(msize2, nr, b, ca+ii*b,
							cb+nb*HGEMM_PACK_NUNIT*b, cc+nb*HGEMM_PACK_NUNIT*ldc+ii, ldc);
					}

				}
				ca += msize * b;
				cc += msize;
				myr += msize;
				if(id==0){
					r = myr;
					pa += msize * b;
					if(com.test(false)) flag = true;
				}
				#pragma omp barrier
				if(flag) break;

			}
		}

		Timer::end(Timer::GEMM_PROGRESS, false, 2ull*(r-firstr)*(cend-c)*b);
		/*while(rend>r && !com.test()){
			// optimize later
			int rsize = r+deltarow < rend ? deltarow: rend - r;
			gemmsimple(rsize, csize, pa, pb, pc+r+c*ldc);
			r += deltarow;
			pa += deltarow*b;
		}*/
	}
	template<typename Comm>
	void update_rest(Comm& lcom, Comm& rcom) {
		if(!do_gemm) return;
		if(cend<=c) return;
		int csize = cend - c;
		while(rend>r && (!lcom.detached() || !rcom.detached())){
			// optimize later
			int rsize = r+deltarow < rend ? deltarow: rend - r;
			gemmsimple(rsize, csize, pa, pb, pc+r+c*ldc);
			r += deltarow;
			pa += deltarow*b;
			lcom.progress();
			rcom.progress();
		}
		if(rend > r){
			//Timer::beg(Timer::GEMM_UPDATE);
			gemmsimple(rend-r, csize, pa, pb, pc+r+c*ldc);
			//Timer::end(Timer::GEMM_UPDATE, false, 2ull*(rend-r)*(cend-c)*b);
		}
	}
};
#endif


#endif
