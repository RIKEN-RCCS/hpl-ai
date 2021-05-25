#ifndef DELAYED_UPDATOR_HPP
#define DELAYED_UPDATOR_HPP
// Lazy initializer adds the initial matrix to the partial sums.
// It changes the order of summantion from C+AB to AB+C.
// This is helpful if the magnitude of valued in C is much larger than that of AB. 
// Partial sums are separeted in two in case of back_buffer.

#ifdef XXX__CLANG_FUJITSU
#include <stdio.h>
#else
#include <cstdio>
#endif
#include "panel.hpp"
#include "hpl_rand.hpp"
#include "timer.hpp"
#include "fp16sim.hpp"
#include "highammgen.hpp"
#include "back_buffer.hpp"

#define LAZY_INIT_OPTIMIZED


template<typename FLow, int lazy_init, bool double_decker>
struct LazyInitializer {
	// lazy-init a diagonal block
	// scale: scales up the partial sum before lazy init
	// row, col: local position of the block
	// buf, ldbuf: working buffer
	template<typename F>
	static void update_diag(Matgen<F>const& mg, Panels<F>& p, F scale, int row, int col, F* buf, size_t ldbuf);
	// lazy-init blocks without diagonal
	// scale: scales up the partial sum before lazy init
	// rowstart, rowend, colstart, colend: local region to lazy-init
	// buf, ldbuf: working buffer
	template<typename F>
	static void update(Matgen<F>const& mg, Panels<F>& p, F scale, int rowstart, int rowend, int colstart, int colend, F* buf, size_t ldbuf);
	template<typename F>
	static void update_diag(HMGen<F>const& mg, Panels<F>& p, F scale, int row, int col, F* buf, size_t ldbuf);
	template<typename F>
	static void update(HMGen<F>const& mg, Panels<F>& p, F scale, int rowstart, int rowend, int colstart, int colend, F* buf, size_t ldbuf);
};

template<typename FLow>
struct LazyInitializer<FLow,0,false> {
	template<typename F>
	static void update_diag(Matgen<F>const& mg, Panels<F>& p, F scale, int row, int col, F* buf, size_t ldbuf);
	template<typename F>
	static void update(Matgen<F>const& mg, Panels<F>& p, F scale, int rowstart, int rowend, int colstart, int colend, F* buf, size_t ldbuf);
	template<typename F>
	static void update_diag(HMGen<F>const& mg, Panels<F>& p, F scale, int row, int col, F* buf, size_t ldbuf);
	template<typename F>
	static void update(HMGen<F>const& mg, Panels<F>& p, F scale, int rowstart, int rowend, int colstart, int colend, F* buf, size_t ldbuf);
};

template<typename FLow, bool dd>
struct LazyInitializer<FLow,1,dd> {
	template<typename F>
	static void update_diag(Matgen<F>const& mg, Panels<F>& p, F scale, int row, int col, F* buf, size_t ldbuf);
	template<typename F>
	static void update(Matgen<F>const& mg, Panels<F>& p, F scale, int rowstart, int rowend, int colstart, int colend, F* buf, size_t ldbuf);
	template<typename F>
	static void update_diag(HMGen<F>const& mg, Panels<F>& p, F scale, int row, int col, F* buf, size_t ldbuf);
	template<typename F>
	static void update(HMGen<F>const& mg, Panels<F>& p, F scale, int rowstart, int rowend, int colstart, int colend, F* buf, size_t ldbuf);
};

template<typename FLow>
struct LazyInitializer<FLow,2,true> {
	template<typename F>
	static void update_diag(Matgen<F>const& mg, Panels<F>& p, F scale, int row, int col, F* buf, size_t ldbuf);
	template<typename F>
	static void update(Matgen<F>const& mg, Panels<F>& p, F scale, int rowstart, int rowend, int colstart, int colend, F* buf, size_t ldbuf);
	template<typename F>
	static void update_diag(HMGen<F>const& mg, Panels<F>& p, F scale, int row, int col, F* buf, size_t ldbuf);
	template<typename F>
	static void update(HMGen<F>const& mg, Panels<F>& p, F scale, int rowstart, int rowend, int colstart, int colend, F* buf, size_t ldbuf);
};


// no lazy init
template<typename FLow> template<typename F>
void LazyInitializer<FLow,0,false>::update_diag(Matgen<F>const&, Panels<F>&,F,int,int,F*,size_t) {} // nop
template<typename FLow> template<typename F>
void LazyInitializer<FLow,0,false>::update(Matgen<F>const&, Panels<F>&,F,int,int,int,int,F*,size_t) {} // nop
template<typename FLow> template<typename F>
void LazyInitializer<FLow,0,false>::update_diag(HMGen<F>const&, Panels<F>&,F,int,int,F*,size_t) {} // nop
template<typename FLow> template<typename F>
void LazyInitializer<FLow,0,false>::update(HMGen<F>const&, Panels<F>&,F,int,int,int,int,F*,size_t) {} // nop


// Matgen
template<typename FLow, bool dd> template<typename F>
void LazyInitializer<FLow, 1, dd>::update_diag(
	Matgen<F>const& mg, Panels<F>& p, F scale, int row, int col, F*, size_t)
{
	Timer::beg(Timer::LAZY_INIT);
	#ifdef LAZY_INIT_OPTIMIZED
	typedef DDAdaptor<F, FLow, dd> DDA;
	typedef typename DDA::FDeck FDeck;
	F* to = p(row, col);
	FDeck* from = DDA::get_deck(p, row, col);
	size_t const lda = p.lda;
	size_t const ldl = DDA::get_ldl(p);
	int const b = p.b;
	int const i0 = p.i1 + row*p.istride; // can be > 32bit
	RandCoeff incl1 = mg.incl1;
	RandCoeff jumpn = mg.jumpn;
	RandCoeff jump_ij = mg.jump(b*i0, b*i0);
	RandStat stat_00 = jump_ij * RandStat::initialize(mg.seed);
	F const* diag = mg.diag;
	for(int j=0; j<b; ++j){
		RandStat stat_i = stat_00;
		F d = diag[b*row + j];
		for(int i=0; i<b; ++i){
			F aij;
			if(i==j)
				aij = d;
			else
				aij = static_cast<F>(stat_i);
			// add "initial" value aij to the partial sum from[j*ldl+i].
			to[j*lda + i] = scale * static_cast<F>(from[j*ldl + i]) + aij;
			stat_i = incl1 * stat_i;
		}
		stat_00 = jumpn * stat_00;
	}
	#else
	typedef DDAdaptor<F, FLow, dd> DDA;
	typedef typename DDA::FDeck FDeck;
	F* to = p(row, col);
	FDeck* from = DDA::get_deck(p, row, col);
	size_t const lda = p.lda;
	size_t const ldl = DDA::get_ldl(p);
	int const b = p.b;
	int const i0 = p.i1 + row*p.istride; 
	fill_one_panel_with_rand(mg.n, b*i0, b*i0, b, b, buf, ldbuf, mg.seed, false);
	F const* diag = mg.diag;
	for(int j=0; j<b; ++j){
		F d = diag[b*row + j];
		for(int i=0; i<b; ++i){
			to[j*lda + i] = scale * static_cast<F>(from[j*ldl + i]) + (i==j? d: buf[j*ldbuf+i]);
		}
	}
	#endif
	Timer::end(Timer::LAZY_INIT, false, 1ull*b*b);
}

template<typename FLow, bool dd> template<typename F>
void LazyInitializer<FLow,1,dd>::update(
	Matgen<F>const& mg, Panels<F>& p, F scale, int rowstart, int rowend, int colstart, int colend, F*, size_t)
{
	Timer::beg(Timer::LAZY_INIT);
	#ifdef LAZY_INIT_OPTIMIZED
	// assuming the range (rowstart:rowend, colstart:colend) doesn't include diagonal panels.
	typedef DDAdaptor<F, FLow, dd> DDA;
	typedef typename DDA::FDeck FDeck;
	size_t const lda = p.lda;
	size_t const ldl = DDA::get_ldl(p);
	int const b = p.b;
	int const i1 = p.i1;
	int const j1 = p.j1;
	int const istride = p.istride;
	int const jstride = p.jstride;
	int const istart = i1 + rowstart*istride;
	int const jstart = j1 + colstart*jstride;
	RandCoeff incl1 = mg.incl1;
	RandCoeff jumpi = mg.jumpi;
	RandCoeff jumpj = mg.jumpj;
	RandCoeff jumpn = mg.jumpn;
	RandCoeff jump_ij = mg.jump(b*istart, b*jstart);
	RandStat stat_00 = jump_ij * RandStat::initialize(mg.seed);
	for(int pj=colstart; pj<colend; ++pj){
		for(int j=0; j<b; ++j){
			RandStat stat_i = stat_00;
			for(int pi=rowstart; pi<rowend; ++pi){
				F* to = p(pi, pj);
				FDeck* from = DDA::get_deck(p, pi, pj);
				for(int i=0; i<b; ++i){
					// assuming no diag.
					F aij = static_cast<F>(stat_i);
					to[j*lda + i] = scale * static_cast<F>(from[j*ldl + i]) + aij;
					stat_i = incl1 * stat_i;
				}
				stat_i = jumpi * stat_i;
			}
			stat_00 = jumpn * stat_00;
		}
		stat_00 = jumpj * stat_00;
	}
	#else
	typedef DDAdaptor<F, FLow, dd> DDA;
	typedef typename DDA::FDeck FDeck;
	size_t const lda = p.lda;
	size_t const ldl = DDA::get_ldl(p);
	int const b = p.b;
	int const i1 = p.i1;
	int const j1 = p.j1;
	int const istride = p.istride;
	int const jstride = p.jstride;
	int const istart = i1 + rowstart*istride;
	int const jstart = j1 + colstart*jstride;
	RandCoeff incl1 = mg.incl1;
	RandCoeff jumpi = mg.jumpi;
	RandCoeff jumpj = mg.jumpj;
	RandCoeff jumpn = mg.jumpn;
	RandCoeff jump_ij = mg.jump(b*istart, b*jstart);
	RandStat stat_00 = jump_ij * RandStat::initialize(mg.seed);
	for(int pj=colstart; pj<colend; ++pj){
		for(int j=0; j<b; ++j){
			RandStat stat_i = stat_00;
			for(int pi=rowstart; pi<rowend; ++pi){
				F* to = p(pi, pj);
				FDeck* from = DDA::get_deck(p, pi, pj);
				for(int i=0; i<b; ++i){
					// assuming no diag.
					F aij = static_cast<F>(stat_i);
					to[j*lda + i] = scale * static_cast<F>(from[j*ldl + i]) + aij;
					stat_i = incl1 * stat_i;
				}
				stat_i = jumpi * stat_i;
			}
			stat_00 = jumpn * stat_00;
		}
		stat_00 = jumpj * stat_00;
	}
	#endif
	Timer::end(Timer::LAZY_INIT, 1ull*(rowend-rowstart)*(colend-colstart)*b*b);
}

// with back_buffer
template<typename FLow> template<typename F>
void LazyInitializer<FLow, 2, true>::update_diag(
	Matgen<F>const& mg, Panels<F>& p, F scale, int row, int col, F*, size_t)
{
	Timer::beg(Timer::LAZY_INIT);
	typedef DDAdaptor<F, FLow, true> DDA;
	typedef typename DDA::FDeck FDeck;
	F* to = p(row, col);
	FDeck* from = DDA::get_deck(p, row, col);
	size_t const lda = p.lda;
	size_t const ldl = DDA::get_ldl(p);
	int const b = p.b;
	int const i0 = p.i1 + row*p.istride; // can be > 32bit
	RandCoeff incl1 = mg.incl1;
	RandCoeff jumpn = mg.jumpn;
	RandCoeff jump_ij = mg.jump(b*i0, b*i0);
	RandStat stat_00 = jump_ij * RandStat::initialize(mg.seed);
	F const* diag = mg.diag;
	F lbufscale = static_cast<F>(BB_NCYCLE);
	F buf[b];
	FLow* lbuf = reinterpret_cast<FLow*>(buf);
	// lbuf contains the part of the partial sum. (back_buffer)
	// By separating the partial sum two, both accuracy and the value-ranges can be better in some case (especially for the Higham's matrix)
	// See back_buffer.hpp, cpp to know how back_buffers are move around during the decomposition.
	for(int j=0; j<b; ++j){
		RandStat stat_i = stat_00;
		F d = diag[b*row + j];
		// 1. backup
		for(int i=0; i<b; ++i) buf[i] = to[j*lda+i];
		// 2. lazy init with bb
		for(int i=0; i<b; ++i){
			F aij;
			if(i==j)
				aij = d;
			else
				aij = static_cast<F>(stat_i);
			F xij = static_cast<F>(from[j*ldl + i]) + lbufscale*static_cast<F>(lbuf[i]);
			to[j*lda + i] = scale * xij +  aij;
			stat_i = incl1 * stat_i;
		}
		stat_00 = jumpn * stat_00;
		// 3. write back bb
		if(row!=p.nprow-1) for(int i=0; i<b; ++i) from[j*ldl+i] = lbuf[b+i];
		// lower bottom part of the lbuf are backed-up to "from". 
	}
	Timer::end(Timer::LAZY_INIT, false, 1ull*b*b);
}

template<typename FLow> template<typename F>
void LazyInitializer<FLow,2,true>::update(
	Matgen<F>const& mg, Panels<F>& p, F scale, int rowstart, int rowend, int colstart, int colend, F*, size_t)
{
	Timer::beg(Timer::LAZY_INIT);
	// assuming the range (rowstart:rowend, colstart:colend) doesn't include diagonal panels.
	typedef DDAdaptor<F, FLow, true> DDA;
	typedef typename DDA::FDeck FDeck;
	size_t const lda = p.lda;
	size_t const ldl = DDA::get_ldl(p);
	int const b = p.b;
	int const i1 = p.i1;
	int const j1 = p.j1;
	int const istride = p.istride;
	int const jstride = p.jstride;
	int const istart = i1 + rowstart*istride;
	int const jstart = j1 + colstart*jstride;
	RandCoeff incl1 = mg.incl1;
	RandCoeff jumpi = mg.jumpi;
	RandCoeff jumpj = mg.jumpj;
	RandCoeff jumpn = mg.jumpn;
	RandCoeff jump_ij = mg.jump(b*istart, b*jstart);
	RandStat stat_00 = jump_ij * RandStat::initialize(mg.seed);
	F lbufscale = static_cast<F>(BB_NCYCLE);
	F buf[b];
	FLow* lbuf = reinterpret_cast<FLow*>(buf);
	for(int pj=colstart; pj<colend; ++pj){
		for(int j=0; j<b; ++j){
			RandStat stat_i = stat_00;
			for(int pi=rowstart; pi<rowend; ++pi){
				F* to = p(pi, pj);
				FDeck* from = DDA::get_deck(p, pi, pj);
				// 1. backup
				for(int i=0; i<b; ++i) buf[i] = to[j*lda+i];
				// 2. lazy init with bb
				for(int i=0; i<b; ++i){
					// assuming no diag.
					F aij = static_cast<F>(stat_i);
					F xij = static_cast<F>(from[j*ldl+i]) + lbufscale*static_cast<F>(lbuf[i]);
					to[j*lda + i] = scale * xij + aij;
					stat_i = incl1 * stat_i;
				}
				stat_i = jumpi * stat_i;
				// 3. write back bb
				if(pi!=p.nprow-1) for(int i=0; i<b; ++i) from[j*ldl+i] = lbuf[b+i];
			}
			stat_00 = jumpn * stat_00;
		}
		stat_00 = jumpj * stat_00;
	}
	Timer::end(Timer::LAZY_INIT, 1ull*(rowend-rowstart)*(colend-colstart)*b*b);
}

// HMGen
template<typename FLow, bool dd> template<typename F>
void LazyInitializer<FLow,1,dd>::update_diag(
	HMGen<F>const& mg, Panels<F>& p, F scale, int row, int col, F* , size_t )
{
	Timer::beg(Timer::LAZY_INIT);
	typedef DDAdaptor<F, FLow, dd> DDA;
	typedef typename DDA::FDeck FDeck;
	size_t const lda = p.lda;
	size_t const ldl = DDA::get_ldl(p);
	int const b = p.b;
	int const i1 = p.i1;
	int const j1 = p.j1;
	int const istride = p.istride;
	int const jstride = p.jstride;
	int istart = b*(i1 + row*istride);
	int jstart = b*(j1 + col*jstride);
	F const alpha = -mg.alpha;
	F const beta = -mg.beta;
	F const ab = alpha*beta;
	F const done = 1;
	F* to = p(row, col);
	FDeck* from = DDA::get_deck(p, row, col);
	#pragma omp parallel for 
	for(int j=0; j<b; ++j){
		F const fpjj = jstart + j;
		for(int i=0; i<j; ++i){
			/*F x0 = -scale * static_cast<F>(from[j*ldl + i]);
			F x1 = ab * istart;
			F e = x1!=(F)0 ? fabs(x0 - x1) / x1: (F)0;
			if(e>1e-7) printf("X %d %d %e %e %.9f\n", istart + i, jstart + j, x0, x1, e);*/
			F aij = beta + ab * (istart + i);
			to[j*lda + i] = scale * static_cast<F>(from[j*ldl + i]) + aij;
		}
		{
			/*F x0 = -scale * static_cast<F>(from[j*ldl + j]);
			F x1 = ab * jstart;
			F e = x1!=(F)0 ? fabs(x0 - x1) / x1: (F)0;
			if(e>1e-7) printf("X %d %d %e %e %.9f\n", jstart + j, jstart + j, x0, x1, e);*/
			to[j*lda + j] = scale * static_cast<F>(from[j*ldl + j]) + (done + ab * fpjj);
		}
		//if(fabs(to[j*lda+j]-1.) > 1e-1) printf("X %d %d %e %e %f\n", jstart + j, jstart + j, to[j*lda+j], done, fabs(to[j*lda+j]-1.));
		for(int i=j+1; i<b; ++i){
			/*F x0 = -scale * static_cast<F>(from[j*ldl + i]);
			F x1 = ab * jstart;
			F e = x1!=(F)0 ? fabs(x0 - x1) / x1: (F)0;
			if(e>1e-7) printf("X %d %d %e %e %.9f\n", istart + i, jstart + j, x0, x1, e);*/
			F aij = alpha + ab * fpjj;
			to[j*lda + i] = scale * static_cast<F>(from[j*ldl + i]) + aij;
		}
		
	}
	Timer::end(Timer::LAZY_INIT, false, 1ull*b*b);
}

template<typename FLow, bool dd> template<typename F>
void LazyInitializer<FLow,1,dd>::update(
	HMGen<F>const& mg, Panels<F>& p, F scale, int rowstart, int rowend, int colstart, int colend, F* , size_t )
{
	Timer::beg(Timer::LAZY_INIT);
	typedef DDAdaptor<F, FLow, dd> DDA;
	typedef typename DDA::FDeck FDeck;
	size_t const lda = p.lda;
	size_t const ldl = DDA::get_ldl(p);
	int const b = p.b;
	int const i1 = p.i1;
	int const j1 = p.j1;
	int const istride = p.istride;
	int const jstride = p.jstride;
	F const alpha = -mg.alpha;
	F const beta = -mg.beta;
	F const ab = alpha*beta;
	#pragma omp parallel for collapse(2)
	for(int pj=colstart; pj<colend; ++pj){
		for(int j=0; j<b; ++j){
			int jstart = b*(j1 + pj*jstride);
			F const fpjj = jstart + j;
			for(int pi=rowstart; pi<rowend; ++pi){
				int istart = b*(i1 + pi*istride);
				F* to = p(pi, pj);
				FDeck* from = DDA::get_deck(p, pi, pj);
				assert(istart!=jstart);
				if(istart<jstart){
					for(int i=0; i<b; ++i){
						// assuming no diag.
						/*F x0 = -scale * static_cast<F>(from[j*ldl + i]);
						F x1 = ab * istart;
						F e = x1!=(F)0 ? fabs(x0 - x1) / x1: (F)0;
						if(e>1e-7) printf("X %d %d %e %e %.9f\n", istart + i, jstart + j, x0, x1, e);*/
						F aij = beta + ab * (istart + i);
						to[j*lda + i] = scale * static_cast<F>(from[j*ldl + i]) + aij;
					}
				}
				else {
					for(int i=0; i<b; ++i){
						// assuming no diag.
						/*F x0 = -scale * static_cast<F>(from[j*ldl + i]);
						F x1 = ab * jstart;
						F e = x1!=(F)0 ? fabs(x0 - x1) / x1: (F)0;
						if(e>1e-7) printf("X %d %d %e %e %.9f\n", istart + i, jstart + j, x0, x1, e);*/
						F aij = alpha + ab * fpjj;
						to[j*lda + i] = scale * static_cast<F>(from[j*ldl + i]) + aij;
					}
				}
			}
		}
	}
	Timer::end(Timer::LAZY_INIT, false, 1ull*(rowend-rowstart)*(colend-colstart)*b*b);
}

// with bb
template<typename FLow> template<typename F>
void LazyInitializer<FLow,2,true>::update_diag(
	HMGen<F>const& mg, Panels<F>& p, F scale, int row, int col, F* , size_t )
{
	Timer::beg(Timer::LAZY_INIT);
	typedef DDAdaptor<F, FLow, true> DDA;
	typedef typename DDA::FDeck FDeck;
	size_t const lda = p.lda;
	size_t const ldl = DDA::get_ldl(p);
	int const b = p.b;
	int const i1 = p.i1;
	int const j1 = p.j1;
	int const istride = p.istride;
	int const jstride = p.jstride;
	int istart = b*(i1 + row*istride);
	int jstart = b*(j1 + col*jstride);
	F const alpha = -mg.alpha;
	F const beta = -mg.beta;
	F const ab = alpha*beta;
	F const done = 1;
	F* to = p(row, col);
	FDeck* from = DDA::get_deck(p, row, col);
	F lbufscale = BB_NCYCLE;
	#pragma omp parallel for 
	for(int j=0; j<b; ++j){
		FLow lbuf[2*b];
		memcpy(lbuf, to+j*lda, sizeof(FLow)*2*b);
		F const fpjj = jstart + j;
		for(int i=0; i<j; ++i){
			#if 0
			F xb = scale * lbufscale * static_cast<F>(lbuf[i]);
			F x0 = scale * static_cast<F>(from[j*ldl + i]);
			F x1 = ab * istart;
			F e = x1!=(F)0 ? fabs(xb + x0 + x1) / x1: (F)0;
			if(e>1e-2) printf("X %d %d %e %e %e %e %.9f\n", istart + i, jstart + j, xb, x0, xb+x0, x1, e);
			#endif
			F aij = beta + ab * (istart + i);
			F xij = static_cast<F>(from[j*ldl+i]) + lbufscale*static_cast<F>(lbuf[i]);
			to[j*lda + i] = scale * xij + aij;
		}
		{
			#if 0
			F xb = scale * lbufscale * static_cast<F>(lbuf[j]);
			F x0 = scale * static_cast<F>(from[j*ldl + j]);
			F x1 = ab * jstart;
			F e = x1!=(F)0 ? fabs(xb + x0 + x1) / x1: (F)0;
			if(e>1e-2) printf("Y %d %d %e %e %e %e %.9f\n", jstart + j, jstart + j, xb, x0, xb+x0, x1, e);
			#endif
			F xij = static_cast<F>(from[j*ldl+j]) + lbufscale*static_cast<F>(lbuf[j]);
			to[j*lda + j] = scale * xij + (done + ab * fpjj);
		}
		for(int i=j+1; i<b; ++i){
			#if 0
			F xb = scale * lbufscale * static_cast<F>(lbuf[i]);
			F x0 = scale * static_cast<F>(from[j*ldl + i]);
			F x1 = ab * jstart;
			F e = x1!=(F)0 ? fabs(xb + x0 + x1) / x1: (F)0;
			if(e>1e-2) printf("Z %d %d %e %e %e %e %.9f\n", istart + i, jstart + j, xb, x0, xb+x0, x1, e);
			#endif
			F aij = alpha + ab * fpjj;
			F xij = static_cast<F>(from[j*ldl+i]) + lbufscale*static_cast<F>(lbuf[i]);
			to[j*lda + i] = scale * xij + aij;
		}
		if(row!=p.nprow-1) memcpy(from+j*ldl, lbuf+b, sizeof(FLow)*b);
		
	}
	Timer::end(Timer::LAZY_INIT, false, 1ull*b*b);
}

template<typename FLow> template<typename F>
void LazyInitializer<FLow,2,true>::update(
	HMGen<F>const& mg, Panels<F>& p, F scale, int rowstart, int rowend, int colstart, int colend, F* , size_t )
{
	Timer::beg(Timer::LAZY_INIT);
	typedef DDAdaptor<F, FLow, true> DDA;
	typedef typename DDA::FDeck FDeck;
	size_t const lda = p.lda;
	size_t const ldl = DDA::get_ldl(p);
	int const b = p.b;
	int const i1 = p.i1;
	int const j1 = p.j1;
	int const istride = p.istride;
	int const jstride = p.jstride;
	F const alpha = -mg.alpha;
	F const beta = -mg.beta;
	F const ab = alpha*beta;
	F lbufscale = BB_NCYCLE;
	#pragma omp parallel for collapse(2)
	for(int pj=colstart; pj<colend; ++pj){
		for(int j=0; j<b; ++j){
			FLow lbuf[2*b];
			int jstart = b*(j1 + pj*jstride);
			F const fpjj = jstart + j;
			for(int pi=rowstart; pi<rowend; ++pi){
				int istart = b*(i1 + pi*istride);
				F* to = p(pi, pj);
				FDeck* from = DDA::get_deck(p, pi, pj);
				assert(istart!=jstart);
				memcpy(lbuf, to+j*lda, sizeof(FLow)*2*b);
				if(istart<jstart){
					for(int i=0; i<b; ++i){
						// assuming no diag.
						#if 0
						F xb = scale * lbufscale * static_cast<F>(lbuf[i]);
						F x0 = scale * static_cast<F>(from[j*ldl+i]);
						F x1 = ab * istart;
						F e = x1!=(F)0 ? fabs(xb + x0 + x1) / x1: (F)0;
						if(e>1e-2) printf("Z %d %d %e %e %e %e %.9f\n", istart + i, jstart + j, xb, x0, xb+x0, x1, e);
						#endif
						F xij = static_cast<F>(from[j*ldl+i]) + lbufscale*static_cast<F>(lbuf[i]);
						F aij = beta + ab * (istart + i);
						to[j*lda + i] = scale * xij + aij;
					}
				}
				else {
					for(int i=0; i<b; ++i){
						// assuming no diag.
						#if 0
						F xb = scale * lbufscale * static_cast<F>(lbuf[i]);
						F x0 = scale * static_cast<F>(from[j*ldl + i]);
						F x1 = ab * jstart;
						F e = x1!=(F)0 ? fabs(xb + x0 + x1) / x1: (F)0;
						if(e>1e-2) printf("W %d %d %e %e %e %e %.9f\n", istart + i, jstart + j, xb, x0, xb+x0, x1, e);
						#endif
						F aij = alpha + ab * fpjj;
						F xij = static_cast<F>(from[j*ldl+i]) + lbufscale*static_cast<F>(lbuf[i]);
						to[j*lda + i] = scale * xij + aij;
					}
				}
				if(pi!=p.nprow-1) memcpy(from+j*ldl, lbuf+b, sizeof(FLow)*b);
			}
		}
	}
	Timer::end(Timer::LAZY_INIT, false, 1ull*(rowend-rowstart)*(colend-colstart)*b*b);
}

#ifdef __aarch64__
// Specializatin of lazy-init.
// We need them because the Fujitsu's compiler does not support fp16 in trad-mode.

#include <omp.h>
#include "panel.hpp"

extern "C"
void lazy_init_f32_f16_in_omp(
		Matgen<float>const& mg, Panels<float>& p, float scale, 
		int rowstart, int rowend, int colstart, int colend,
		int jthreadstart, int jthreadend);
extern "C"
void lazy_init_diag_f32_f16_in_omp(
		Matgen<float>const& mg, Panels<float>& p, float scale, 
		int row, int col, const float *diag,
		int jthreadstart, int jthreadend);

// specialization for OpenMP and SVE
template<> template<>
inline void LazyInitializer<fp16, 1, true>::update(Matgen<float>const& mg, Panels<float>& p, float scale, int rowstart, int rowend, int colstart, int colend, float *, size_t){
	Timer::beg(Timer::LAZY_INIT);
#pragma omp parallel
	{
		int nth = omp_get_num_threads();
		int tid = omp_get_thread_num();
		int js = (p.b * (0+tid)) / nth;
		int je = (p.b * (1+tid)) / nth;
		lazy_init_f32_f16_in_omp(mg, p, scale, rowstart, rowend, colstart, colend, js, je);
	}
	Timer::end(Timer::LAZY_INIT, false, 1ull*(rowend-rowstart)*(colend-colstart)*p.b*p.b);
}
template<> template<>
inline void LazyInitializer<fp16, 1, true>::update_diag(Matgen<float>const& mg, Panels<float>& p, float scale, int row, int col, float *, size_t){
	Timer::beg(Timer::LAZY_INIT);
#pragma omp parallel
	{
		int nth = omp_get_num_threads();
		int tid = omp_get_thread_num();
		int js = (p.b * (0+tid)) / nth;
		int je = (p.b * (1+tid)) / nth;
		lazy_init_diag_f32_f16_in_omp(mg, p, scale, row, col, mg.diag, js, je);
	}
	Timer::end(Timer::LAZY_INIT, false, 1ull*(p.b*p.b));
}

// with bb
extern "C"
void lazy_init_f32_f16_in_omp_bb(
		Matgen<float>const& mg, Panels<float>& p, float scale, 
		int rowstart, int rowend, int colstart, int colend,
		int jthreadstart, int jthreadend, float* buf);
extern "C"
void lazy_init_diag_f32_f16_in_omp_bb(
		Matgen<float>const& mg, Panels<float>& p, float scale, 
		int row, int col, const float *diag,
		int jthreadstart, int jthreadend, float* buf);

template<> template<>
inline void LazyInitializer<fp16, 2, true>::update(Matgen<float>const& mg, Panels<float>& p, float scale, int rowstart, int rowend, int colstart, int colend, float *, size_t){
	Timer::beg(Timer::LAZY_INIT);
#pragma omp parallel
	{
		int nth = omp_get_num_threads();
		int tid = omp_get_thread_num();
		int js = (p.b * (0+tid)) / nth;
		int je = (p.b * (1+tid)) / nth;
		float buf[p.b];
		lazy_init_f32_f16_in_omp_bb(mg, p, scale, rowstart, rowend, colstart, colend, js, je, buf);
	}
	Timer::end(Timer::LAZY_INIT, false, 1ull*(rowend-rowstart)*(colend-colstart)*p.b*p.b);
}
template<> template<>
inline void LazyInitializer<fp16, 2, true>::update_diag(Matgen<float>const& mg, Panels<float>& p, float scale, int row, int col, float *, size_t){
	Timer::beg(Timer::LAZY_INIT);
#pragma omp parallel
	{
		int nth = omp_get_num_threads();
		int tid = omp_get_thread_num();
		int js = (p.b * (0+tid)) / nth;
		int je = (p.b * (1+tid)) / nth;
		float buf[p.b];
		lazy_init_diag_f32_f16_in_omp_bb(mg, p, scale, row, col, mg.diag, js, je, buf);
	}
	Timer::end(Timer::LAZY_INIT, false, 1ull*(p.b*p.b));
}

// for HMGen
extern "C"
void lazy_init_f32_f16_in_omp_hm(
	HMGen<float>const& mg, Panels<float>& p, float scale, 
	int rowstart, int rowend, int colstart, int colend,
	int jthreadstart, int jthreadend);
extern "C"
void lazy_init_diag_f32_f16_in_omp_hm(
	HMGen<float>const& mg, Panels<float>& p, float scale, 
	int row, int col, int jthreadstart, int jthreadend);

template<> template<>
inline void
LazyInitializer<fp16, 1, true>::update(HMGen<float>const& mg, Panels<float>& p, float scale, int rowstart, int rowend, int colstart, int colend, float *, size_t)
{
	Timer::beg(Timer::LAZY_INIT);
#pragma omp parallel
	{
		int nth = omp_get_num_threads();
		int tid = omp_get_thread_num();
		int js = (p.b * (0+tid)) / nth;
		int je = (p.b * (1+tid)) / nth;
		lazy_init_f32_f16_in_omp_hm(mg, p, scale, rowstart, rowend, colstart, colend, js, je);
	}
	Timer::end(Timer::LAZY_INIT, false, 1ull*(rowend-rowstart)*(colend-colstart)*p.b*p.b);
}
template<> template<>
inline void
LazyInitializer<fp16, 1, true>::update_diag(HMGen<float>const& mg, Panels<float>& p, float scale, int row, int col, float *, size_t)
{
	Timer::beg(Timer::LAZY_INIT);
#pragma omp parallel
	{
		int nth = omp_get_num_threads();
		int tid = omp_get_thread_num();
		int js = (p.b * (0+tid)) / nth;
		int je = (p.b * (1+tid)) / nth;
		lazy_init_diag_f32_f16_in_omp_hm(mg, p, scale, row, col, js, je);
	}
	Timer::end(Timer::LAZY_INIT, false, 1ull*(p.b*p.b));
}

// with BB
extern "C"
void lazy_init_f32_f16_in_omp_hm_bb(
	HMGen<float>const& mg, Panels<float>& p, float scale, 
	int rowstart, int rowend, int colstart, int colend,
	int jthreadstart, int jthreadend, float* buf);
extern "C"
void lazy_init_diag_f32_f16_in_omp_hm_bb(
	HMGen<float>const& mg, Panels<float>& p, float scale, 
	int row, int col, int jthreadstart, int jthreadend, float* buf);

template<> template<>
inline void
LazyInitializer<fp16, 2, true>::update(HMGen<float>const& mg, Panels<float>& p, float scale, int rowstart, int rowend, int colstart, int colend, float *, size_t)
{
	Timer::beg(Timer::LAZY_INIT);
#pragma omp parallel
	{
		int nth = omp_get_num_threads();
		int tid = omp_get_thread_num();
		int js = (p.b * (0+tid)) / nth;
		int je = (p.b * (1+tid)) / nth;
		float buf[p.b];
		lazy_init_f32_f16_in_omp_hm_bb(mg, p, scale, rowstart, rowend, colstart, colend, js, je, buf);
	}
	Timer::end(Timer::LAZY_INIT, false, 1ull*(rowend-rowstart)*(colend-colstart)*p.b*p.b);
}
template<> template<>
inline void
LazyInitializer<fp16, 2, true>::update_diag(HMGen<float>const& mg, Panels<float>& p, float scale, int row, int col, float *, size_t)
{
	Timer::beg(Timer::LAZY_INIT);
#pragma omp parallel
	{
		int nth = omp_get_num_threads();
		int tid = omp_get_thread_num();
		int js = (p.b * (0+tid)) / nth;
		int je = (p.b * (1+tid)) / nth;
		float buf[p.b];
		lazy_init_diag_f32_f16_in_omp_hm_bb(mg, p, scale, row, col, js, je, buf);
	}
	Timer::end(Timer::LAZY_INIT, false, 1ull*(p.b*p.b));
}
#endif

#endif
