#ifndef BACK_BUFFER_HPP
#define BACK_BUFFER_HPP
// BackBuffer periodically write-back the partial sum to another storage.
// This will prevent large rouding error in fp16 computation with HPL-AI matrix by blocking techniuqe.
// BackBuffer utilizes the upper-part of the double-decker layout. 

// The period for write-back
// Large BB_NCYCLE results in large error.
// Small BB_NCYCLE results in large cost.
#define BB_NCYCLE 32

#include "fp16sim.hpp"
#include "panel.hpp"
#include "timer.hpp"
#include <stdio.h>

template<typename FHigh, typename FLow, int mode>
struct BackBuffer {};

template<typename FHigh, typename FLow>
struct BackBuffer<FHigh, FLow, 0> {
	BackBuffer(int /*nprow*/) {}
	// tell bb to invalidate the first row
	void pop(int /*row*/) {}
	void write_back(int /*step*/, Panels<FHigh>& /*p*/, int /*rowstart*/, int /*colstart*/) {}
};

template<typename FHigh, typename FLow>
struct BackBuffer<FHigh, FLow, 1> {
	BackBuffer(int /*nprow*/) {}
	void pop(int /*row*/ ) {}
	void write_back(int /*step*/, Panels<FHigh>& /*p*/, int /*rowstart*/, int /*colstart*/) {}
};


extern "C" void bb_init_piv(int nprow, int* piv, int* rpiv);

extern "C" void bb_writeback_impl_fp16(int b, int rowstart, int nprow, int const* rpiv, fp16* column);
extern "C" void bb_writeback_impl_fp32(int b, int rowstart, int nprow, int const* rpiv, float* column);
extern "C" void bb_writeback_impl_int_fp16(int b, int rowstart, int nprow, int const* rpiv, fp16* column);
extern "C" void bb_writeback_impl_int_fp32(int b, int rowstart, int nprow, int const* rpiv, float* column);
static void bb_writeback_impl(int b, int rowstart, int nprow, int const* rpiv, fp16* column)
{
	bb_writeback_impl_fp16(b, rowstart, nprow, rpiv, column);
}
static void bb_writeback_impl(int b, int rowstart, int nprow, int const* rpiv, float* column)
{
	bb_writeback_impl_fp32(b, rowstart, nprow, rpiv, column);
}
static void bb_writeback_impl_int(int b, int rowstart, int nprow, int const* rpiv, fp16* column)
{
	bb_writeback_impl_int_fp16(b, rowstart, nprow, rpiv, column);
}
static void bb_writeback_imple_int(int b, int rowstart, int nprow, int const* rpiv, float* column)
{
	bb_writeback_impl_int_fp32(b, rowstart, nprow, rpiv, column);
}

extern "C" void bb_writeback_b_impl_fp16(int b, fp16* column0, fp16* column1);
extern "C" void bb_writeback_b_impl_fp32(int b, float* column0, float* column1);
extern "C" void bb_writeback_b_impl_int_fp16(int b, fp16* column0, fp16* column1);
extern "C" void bb_writeback_b_impl_int_fp32(int b, float* column0, float* column1);
static void bb_writeback_b_impl(int b, fp16* column0, fp16* column1)
{
	bb_writeback_b_impl_fp16(b, column0, column1);
}
static void bb_writeback_b_impl(int b, float* column0, float* column1)
{
	bb_writeback_b_impl_fp32(b, column0, column1);
}
static void bb_writeback_b_impl_int(int b, fp16* column0, fp16* column1)
{
	bb_writeback_b_impl_int_fp16(b, column0, column1);
}
static void bb_writeback_b_impl_int(int b, float* column0, float* column1)
{
	bb_writeback_b_impl_int_fp32(b, column0, column1);
}


template<typename FHigh, typename FLow>
struct BackBuffer<FHigh, FLow, 2> {
	int nprow;
	int* piv, *rpiv;
	BackBuffer(int nprow) : nprow(nprow) {
		// we need piv to know where the blocks locate
		// the position changes as the decomposition progresses
		piv = (int*)std::malloc(sizeof(int)*nprow*3);
		rpiv = piv + 2 * nprow;
		bb_init_piv(nprow, piv, rpiv);
	}
	~BackBuffer() { std::free(piv); }
	inline void pop(int row) {
		// back-buffers will be up-casted to FHigh and marged into the forward one in lazy-init
		// To keep the data contiguous, we order and move the data in this manner:
		// The initial position is described in back_buffer.cpp
		// When pop(row) is called, the row-th back-buffer is placed at 2*row-the block
		// and the front-buffer is placed at (row+nprow)-th block  in the column
		// pop(row) backup the (2*row+1)-th block in a temporal storage, and merge the back and fron buffers to [2*row, 2*row+1]-the blocks.
		// Be careful that lazy-init increases the size of the storage for a block twice  because it up-cast the data.
		// Then, it writes-back the (2*row+1)-th block in the temporal storage to the (row+nprow)-th block.
		//
		//      2*row 2*row+1    nprow+row
		//          ||           |
		// [--------BB-----------F----]        ->    [--------XX-----------B----] 
		//
		// In this techqniue, the data in extended precision and in the front-buffer are contiguous, which ease the other part of the code.
		// Instead, the ordering of the back-buffer becomes quite mazing. That is the key of this process.
		// bb_init_piv() construct the initial position with the reversed-construction technique, in other words, 
		// it construct the initial position by applying the inverse of the pop()s to the last position.
		// See the implementation of bb_init_piv() for details.
		if(row>=nprow-1) return;
		int e = piv[2*row+1];
		piv[nprow+row] = e;
		rpiv[e] = nprow + row;
	}
	void write_back(int step, Panels<FHigh>& p, int rowstart, int colstart) {
		if(!step || (step-1)%BB_NCYCLE!= BB_NCYCLE-1) return;
		Timer::beg(Timer::WRITE_BACK);
		typedef DDAdaptor<FHigh, FLow, true> DDA;
		//return;
		if(!p.is_tile){
			int const b = p.b;
			int const nprow = p.nprow;
			int const npcol = p.npcol;
			size_t const ldl = DDA::get_ldl(p);
			FLow* data = reinterpret_cast<FLow*>(p(0,0));
			int cbegin = b * colstart;
			int cend = b * npcol;
			#if 0
			printf("B ");
			for(int i=0; i<2*nprow; ++i) printf(" %f", (float)data[ldl*b*colstart+b*i]);
			printf("\n");
			#endif
			#pragma omp parallel for
			for(int j=cbegin; j<cend; ++j){
				FLow* c = data + ldl * j;
				#if 1
				// write back single column
				bb_writeback_impl(b, rowstart, nprow, rpiv, c);
				#else
				bb_writeback_int_impl(b, rowstart, nprow, rpiv, c);
				#endif
			}
			#if 0
			printf("C ");
			for(int i=0; i<2*nprow; ++i) printf(" %f", (float)data[ldl*b*colstart+b*i]);
			printf("\n");
			#endif
		}
		else {
			int const b = p.b;
			int const npcol = p.npcol;
			int const nprow = p.nprow;
			#pragma omp parallel for collapse(2)
			for(int j=colstart; j<npcol; ++j){
				for(int i=rowstart; i<nprow; ++i){
					FLow* c0 = reinterpret_cast<FLow*>(p(i,j));
					FLow* c1 = c0 + b;
					#if 1
					bb_writeback_b_impl(b, c0, c1);
					#else
					bb_writeback_b_impl_int(b, c0, c1);
					#endif
				}
			}
		}
		Timer::end(Timer::WRITE_BACK, false, 1ull*p.b*p.b*(p.nprow-rowstart)*(p.npcol-colstart));
	}
};

#endif

