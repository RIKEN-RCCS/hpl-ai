#ifndef PANEL_HPP
#define PANEL_HPP
// panel descriptors and misc functions for panel.

#include <cstdlib>
#include <cassert>
#include <cstring>
#include "kernels/kernel.h"

static inline size_t calc_lda_c(size_t n){
	// XXX This is public version.
	// XXX FOR FUGAKU, CONTACT WITH FUJITSU AND RIKEN FOR SETTINGS WE USED IN THE BENCHMARK.
	size_t const cl_size = 256ull;
	size_t const page_size = 4096ull;
	n = (n+cl_size-1) / cl_size * cl_size;
	if(n%page_size) n += cl_size;
	return n;
}

template <typename F>
struct Panels {
	// the panel descriptor.
	// We use 2D block-cyclic layout. This is little simpler than the PBLAS's one.
	// It is like m == n, mb == nb, m%mb == n%nb == 0, and i0==j0==0 in PBLAS's descriptor.
	// Let P is the local sub-matrices and A is the global matrix:
	char *p; // &(P_{i, j}) == p+(i-1)*ldp+(j-1)*ldpp
	size_t alloc_size; // allocated memory size in bytes
	size_t lda;
	size_t ldp;
	size_t ldpp;
	int nblocks; // the number blocks of row and column of A. A_{nblocks, nblocks} is the right bottom corner
	int b; // nrow(P_{*,*}) == ncol(P_{*,*})
	int nprow, npcol; // P_{nprow,npcol} is at the right bottom corner
	int i1, j1; // P_{1, 1} == A_{i1, j1}
	int istride, jstride; // P_{i, j} == A_{i1+(i-1)*istride, j1+(j-1)*jstride}

	int epoch_size;
	bool is_tile;

	F *panel_address(int i, int j) { return reinterpret_cast<F*>(p) + i * ldp + j * ldpp; }
	F const*panel_address(int i, int j) const { return reinterpret_cast<F const*>(p) + i * ldp + j * ldpp; }
	F *operator()(int i, int j) { return panel_address(i, j); }
	F const*operator()(int i, int j) const { return panel_address(i, j); }

	// double-decker placement.
	// lower-precision data are placed on the lower half of the high-precisino matrix.
	template<typename FLow>
	FLow *lower_deck(int i, int j) {
		// sizeof(FLow)*2 == sizeof(F)
		if(is_tile)
			return reinterpret_cast<FLow*>(p + sizeof(F)*(i * ldp + j * ldpp + lda)) - lda;
		else
			return reinterpret_cast<FLow*>(p + sizeof(F)*(j * ldpp) + sizeof(FLow)*b*nprow) + i * ldp;
	}
	template<typename FLow>
	FLow const*lower_deck(int i, int j) const {
		if(is_tile)
			return reinterpret_cast<FLow const*>(p + sizeof(F)*(i * ldp + j * ldpp + lda)) - lda;
		else
			return reinterpret_cast<FLow const*>(p + sizeof(F)*(j * ldpp) + sizeof(FLow)*b*nprow) + i * ldp;
	}

	template<typename FLow>
	FLow *higher_deck(int i, int j) {
		// sizeof(FLow)*2 == sizeof(F)
		if(is_tile)
			return reinterpret_cast<FLow*>(p + sizeof(F)*(i * ldp + j * ldpp));
		else
			return reinterpret_cast<FLow*>(p + sizeof(F)*(j * ldpp)) + i * ldp;
	}
	template<typename FLow>
	FLow const*higher_deck(int i, int j) const {
		if(is_tile)
			return reinterpret_cast<FLow const*>(p + sizeof(F)*(i * ldp + j * ldpp));
		else
			return reinterpret_cast<FLow const*>(p + sizeof(F)*(j * ldpp)) + i * ldp;
	}

	template<typename FLow>
	size_t lower_deck_lda() const {
		return sizeof(F)/sizeof(FLow) * lda;
	}
	template<typename FLow> size_t higher_deck_lda() const { return lower_deck_lda(); }

};

// An wrapper to get the panel address for each matrix layout.
template<typename F, typename FLow, bool t>
struct DDAdaptor {};
template<typename F, typename FLow>
struct DDAdaptor<F, FLow, true> {
	typedef FLow FDeck;
	static FLow* get_deck(Panels<F>& p, int row, int col){
		return p.template lower_deck<FLow>(row, col);
	}
	static size_t get_ldl(Panels<F> const & p){
		return p.template lower_deck_lda<FLow>();
	}
};
template<typename F, typename FLow>
struct DDAdaptor<F, FLow, false> {
	typedef F FDeck;
	static F* get_deck(Panels<F>& p, int row, int col){
		return p(row, col);
	}
	static size_t get_ldl(Panels<F> const & p){
		return p.lda;
	}
};


template <typename F>
struct LRPanels {
	// ther working space for receiving left and right panels
	char *p; // maybe different type ie fp16.
	size_t offset;
	// offset from the alloc head
	// in case allocating all panels in a single region to have better control over the alignement.
	size_t lda;
	size_t ldp;
	int istart;
	bool is_tile;
	bool is_pack;

	void set_start(int i) { istart = i; }
	size_t get_lda() const { return is_tile ? lda: (lda-ldp*istart); }
	F *data() { return reinterpret_cast<F*>(p);}
	F const *data() const { return reinterpret_cast<F const*>(p);}
	F *address(int i) { return reinterpret_cast<F*>(p) + (i-istart) * ldp; }
	F const *address(int i) const { return reinterpret_cast<F const*>(p) + (i-istart) * ldp; }
	F *operator()(int i) { return address(i); }
	F const *operator()(int i) const { return address(i); }
};

template <typename FHigh, typename FLow>
int build_panels(int n, int b, bool tile_layout, bool pack,
	Panels<FHigh> &p, LRPanels<FLow>* lr,
	int row, int col, int nrow, int ncol, int nbuf=2)
{
	assert(n>=0);
	assert(b>=1);
	assert(n%b==0);
	assert(n/b>0);
	assert(nbuf>=2);
	int nb = n / b;
	p.i1 = row;
	p.j1 = col;
	p.nprow = (nb - p.i1 + nrow - 1) / nrow;
	p.npcol = (nb - p.j1 + ncol - 1) / ncol;
	p.istride = nrow;
	p.jstride = ncol;
	p.nblocks = nb;
	p.b = b;
	if(tile_layout){
		p.lda = b;
		p.ldp = b * b;
		p.ldpp = b * b * p.nprow;
		p.is_tile = true;
		for(int i=0; i<nbuf; ++i){
			lr[2*i].lda = b;
			lr[2*i].ldp = b * b;
			lr[2*i].is_tile = true;
			lr[2*i].is_pack = false;
			lr[2*i].istart = 0;
			lr[2*i+1].lda = b;
			lr[2*i+1].ldp = b * b;
			lr[2*i+1].is_tile = true;
			lr[2*i+1].is_pack = false;
			lr[2*i+1].istart = 0;
		}
		size_t sz_p = sizeof(FHigh) * b * b * p.nprow * p.npcol;
		sz_p = (sz_p+1023ull) & ~1023ull;
		size_t sz_l = sizeof(FLow) * b * b * p.nprow;
		sz_l = (sz_l+1023ull) & ~1023ull;
		size_t sz_r = sizeof(FLow) * b * b * p.npcol;
		sz_r = (sz_r+1023ull) & ~1023ull;
		p.alloc_size = sz_p + nbuf*sz_l + nbuf*sz_r + 0x100ull;
		char* addr = (char*)aligned_alloc(4096, p.alloc_size);
		assert(addr);
		p.p = addr;
		memset(p.p, 0, p.alloc_size);
		for(int i=0; i<nbuf; ++i){
			lr[2*i].p = addr + sz_p + sz_l * i;
			lr[2*i].offset = sz_p + sz_l * i;;
			lr[2*i+1].p = addr + sz_p + nbuf*sz_l + 0x100ull + sz_r * i;
			lr[2*i+1].offset = sz_p + nbuf*sz_l + 0x100ull + sz_r * i;;
		}
	}
	#ifdef HGEMM_PACK
	else if(pack){
		p.lda = calc_lda_c(b*p.nprow * sizeof(FHigh)) / sizeof(FHigh);
		// p.lda = b*p.nprow + 4096/sizeof(FHigh)
		p.ldp = b;
		p.ldpp = p.lda * b;
		p.is_tile = false;
		for(int i=0; i<nbuf; ++i){
			lr[2*i].lda = 0;
			lr[2*i].ldp = 0;
			lr[2*i].istart = 0;
			lr[2*i].is_tile = false;
			lr[2*i].is_pack = true;
			lr[2*i+1].lda = 0;
			lr[2*i+1].ldp = 0;
			lr[2*i+1].istart = 0;
			lr[2*i+1].is_tile = false;
			lr[2*i+1].is_pack = true;
		}
		size_t msize = (p.b * p.nprow + HGEMM_PACK_MUNIT-1)/HGEMM_PACK_MUNIT*HGEMM_PACK_MUNIT;
		size_t nsize = (p.b * p.npcol + HGEMM_PACK_NUNIT-1)/HGEMM_PACK_NUNIT*HGEMM_PACK_NUNIT;
		size_t sz_p = sizeof(FHigh) * p.ldpp * p.npcol;
		sz_p = (sz_p+1023ull) & ~1023ull;
		size_t sz_l = sizeof(FLow) * msize * b;
		sz_l = (sz_l+1023ull) & ~1023ull;
		size_t sz_r = sizeof(FLow) * nsize * b;
		sz_r = (sz_r+1023ull) & ~1023ull;
		p.alloc_size = sz_p + nbuf*sz_l + nbuf*sz_r + 0x100ull;
		char* addr;
		assert(posix_memalign(reinterpret_cast<void**>(&addr), 4096, p.alloc_size) == 0);
		p.p = addr;
		memset(p.p, 0, p.alloc_size);

		for(int i=0; i<nbuf; ++i){
			lr[2*i].p = addr + sz_p + sz_l * i;
			lr[2*i].offset = sz_p + sz_l * i;;
			lr[2*i+1].p = addr + sz_p + nbuf*sz_l + 0x100ull + sz_r * i;
			lr[2*i+1].offset = sz_p + nbuf*sz_l + 0x100ull + sz_r * i;;
		}
	}
	#endif
	else {
		p.lda = calc_lda_c(b*p.nprow * sizeof(FHigh)) / sizeof(FHigh);
		// p.lda = b*p.nprow + 4096/sizeof(FHigh)
		p.ldp = b;
		p.ldpp = p.lda * b;
		p.is_tile = false;
		for(int i=0; i<nbuf; ++i){
			lr[2*i].lda = b * p.nprow;
			lr[2*i].ldp = b;
			lr[2*i].istart = 0;
			lr[2*i].is_tile = false;
			lr[2*i].is_pack = false;
			lr[2*i+1].lda = b;
			lr[2*i+1].ldp = b * b;
			lr[2*i+1].istart = 0;
			lr[2*i+1].is_tile = true;
			lr[2*i+1].is_pack = false;
		}
		size_t sz_p = sizeof(FHigh) * p.ldpp * p.npcol;
		sz_p = (sz_p+1023ull) & ~1023ull;
		size_t sz_l = sizeof(FLow) * lr[0].lda * b;
		sz_l = (sz_l+1023ull) & ~1023ull;
		size_t sz_r = sizeof(FLow) * lr[1].ldp * p.npcol;
		sz_r = (sz_r+1023ull) & ~1023ull;
		p.alloc_size = sz_p + nbuf*sz_l + nbuf*sz_r + 0x100ull;
		char* addr;
		assert(posix_memalign(reinterpret_cast<void**>(&addr), 4096, p.alloc_size) == 0);
		p.p = addr;
		memset(p.p, 0, p.alloc_size);

		for(int i=0; i<nbuf; ++i){
			lr[2*i].p = addr + sz_p + sz_l * i;
			lr[2*i].offset = sz_p + sz_l * i;;
			lr[2*i+1].p = addr + sz_p + nbuf*sz_l + 0x100ull + sz_r * i;
			lr[2*i+1].offset = sz_p + nbuf*sz_l + 0x100ull + sz_r * i;;
		}
	}
	return nb;
}


template <typename FHigh, typename FLow>
void destruct_panels(Panels<FHigh> &p, LRPanels<FLow>* /*lr*/)
{
	free(p.p);
}

// row and column vectors are hold on the node which has the diagoanl block of same index.
// Thus, there is no duplication over nodes.
template<typename F, typename Fv>
void colv2rowv(Panels<F>const & p, Fv const* cx, Fv* rx)
{
	int b = p.b;
	for(int i=0; i<p.nprow; ++i){
		int ipos = p.i1 + i*p.istride;
		if(ipos%p.jstride == p.j1){
			int j = (ipos-p.j1)/p.jstride;
			#pragma omp parallel for simd
			for(int k=0; k<b; ++k) rx[b*j + k] = cx[b*i + k];
		}
	}
}

template<typename F, typename Fx, typename Fy>
void copycolv(Panels<F>const & p, Fx const* x, Fy* y)
{
	int b = p.b;
	for(int i=0; i<p.nprow; ++i){
		int ipos = p.i1 + i*p.istride;
		if(ipos%p.jstride == p.j1){
			#pragma omp parallel for simd
			for(int k=0; k<b; ++k) y[b*i + k] = static_cast<Fy>(x[b*i + k]);
		}
	}
}

template<typename F, typename Fv>
void addcolv(Panels<F>const & p, Fv const* x, Fv* y)
{
	int b = p.b;
	for(int i=0; i<p.nprow; ++i){
		int ipos = p.i1 + i*p.istride;
		if((ipos%p.jstride) == p.j1){
			#pragma omp parallel for simd
			for(int k=0; k<b; ++k) y[b*i + k] += x[b*i + k];
		}
	}
}

template<typename F, typename Fv>
void divcolv(Panels<F>const & p, Fv const* x, Fv* y)
{
	int b = p.b;
	for(int i=0; i<p.nprow; ++i){
		int ipos = p.i1 + i*p.istride;
		if((ipos%p.jstride) == p.j1){
			#pragma omp parallel for simd
			for(int k=0; k<b; ++k) y[b*i + k] /= x[b*i + k];
		}
	}
}

#endif
