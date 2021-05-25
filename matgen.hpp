#ifndef MATGEN_HPP
#define MATGEN_HPP
// functions to initialize matrix.
#include "panel.hpp"
#include "hpl_rand.hpp"

template <typename F>
void pmatgen(Matgen<F> const& mg, Panels<F> &p){
	int const bs = p.b;
	size_t const lda = p.lda;
	int const nprow = p.nprow;
	int const npcol = p.npcol;

	for(int pj=0; pj<npcol; ++pj)
		for(int pi=0; pi<nprow; ++pi){
			F *pp = p(pi, pj);
			const int i = bs*(p.i1 + pi*p.istride);
			const int j = bs*(p.j1 + pj*p.jstride);

			fill_one_panel_with_rand(mg.n, i, j, bs, bs, pp, lda, mg.seed, true);
		}
}

template <typename F>
void pmatgen(HMGen<F> const& mg, Panels<F> &p){
	size_t const lda = p.lda;
	int const nprow = p.nprow;
	int const npcol = p.npcol;
	int const b = p.b;
	int const i1 = p.i1;
	int const j1 = p.j1;
	int const istride = p.istride;
	int const jstride = p.jstride;
	F const alpha = mg.alpha;
	F const beta = mg.beta;
	F const ab = alpha*beta;
	F const done = 1;

	#pragma omp parallel for collapse(2)
	for(int pj=0; pj<npcol; ++pj){
		for(int j=0; j<b; ++j){
			int jstart = b*(j1 + pj*jstride);
			F const fpjj = jstart + j;
			for(int pi=0; pi<nprow; ++pi){
				int istart = b*(i1 + pi*istride);
				F* to = p(pi, pj);
				if(pi<pj){
					for(int i=0; i<b; ++i){
						// assuming no diag.
						F aij = beta + ab * (istart + i);
						to[j*lda + i] = aij;
					}
				}
				else if(pi>pj){
					for(int i=0; i<b; ++i){
						// assuming no diag.
						F aij = alpha + ab * fpjj;
						to[j*lda + i] = aij;
					}
				}
				else {
					for(int i=0; i<j; ++i){
						// assuming no diag.
						F aij = beta + ab * (jstart + i);
						to[j*lda + i] = aij;
					}
					F aij = done + ab * fpjj;
					to[j*lda + j] = aij;
					for(int i=j+1; i<b; ++i){
						// assuming no diag.
						F aij = alpha + ab * fpjj;
						to[j*lda + i] = aij;
					}
				}
			}
		}
	}
}

template <typename F>
void pmatgen0(Panels<F> &p){
	// initialize with zero
	int const bs = p.b;
	size_t const lda = p.lda;
	int const nprow = p.nprow;
	int const npcol = p.npcol;
	F const dzero = static_cast<F>(0);

	if(p.is_tile){
		#pragma omp parallel for collapse(2)
		for(int pj=0; pj<npcol; ++pj)
			for(int pi=0; pi<nprow; ++pi){
				F *pp = p(pi, pj);
				for(int j=0; j<bs; ++j)
					for(int i=0; i<bs; ++i)
						pp[j*lda + i] = dzero;
			}
	}
	else{
		F* ptr = p(0, 0);
		size_t size = static_cast<size_t>(p.ldpp) * npcol;
		#pragma omp parallel for simd
		for(size_t i=0; i<size; ++i) ptr[i] = dzero;
	}
}

template <typename F>
void pmatl1est(Matgen<F> const& mg, Panels<F> &p){
	// approximation of the decomposition
	int const bs = p.b;
	size_t const lda = p.lda;
	int const nprow = p.nprow;
	int const npcol = p.npcol;

	#pragma omp parallel for 
	for(int pj=0; pj<npcol; ++pj){
		double buf[bs];
		const int j = bs*(p.j1 + pj*p.jstride);
		for(int jj=0; jj<bs; ++jj) buf[jj] = 1./calc_diag(j+jj, mg.n, mg.seed);
		for(int pi=0; pi<nprow; ++pi){
			F *pp = p(pi, pj);
			const int i = bs*(p.i1 + pi*p.istride);
			if(i<j) continue;
			if(i==j){
				for(int jj=0; jj<bs; ++jj) {
					F d = buf[jj];
					for(int ii=0; ii<bs; ++ii){
						if(i+ii>j+jj) {
							pp[jj*lda+ii] *= d;
						}
					}
				}
			}
			else{
				for(int jj=0; jj<bs; ++jj) {
					F d = buf[jj];
					for(int ii=0; ii<bs; ++ii){
						pp[jj*lda+ii] *= d;
					}
				}
			}
		}
	}
}


template <typename F>
void pmatl1est(HMGen<F> const& mg, Panels<F> &p){
	// approximation of the decomposition
	int const bs = p.b;
	size_t const lda = p.lda;
	int const nprow = p.nprow;
	int const npcol = p.npcol;
	F const alpha = mg.alpha;
	F const beta= mg.beta;
	F const done = 1;

	#pragma omp parallel for collapse(2) schedule(dynamic)
	for(int pj=0; pj<npcol; ++pj)
		for(int pi=0; pi<nprow; ++pi){
			F *pp = p(pi, pj);
			const int i = bs*(p.i1 + pi*p.istride);
			const int j = bs*(p.j1 + pj*p.jstride);
			if(i<j){
				for(int jj=0; jj<bs; ++jj) {
					for(int ii=0; ii<bs; ++ii){
						pp[jj*lda+ii] = beta;
					}
				}
			}
			else if(i>j){
				for(int jj=0; jj<bs; ++jj) {
					for(int ii=0; ii<bs; ++ii){
						pp[jj*lda+ii] = alpha;
					}
				}
			}
			else {
				for(int jj=0; jj<bs; ++jj) {
					for(int ii=0; ii<jj; ++ii){
						pp[jj*lda+ii] = beta;
					}
					pp[jj*lda+jj] = done;
					for(int ii=jj+1; ii<bs; ++ii){
						pp[jj*lda+ii] = alpha;
					}
				}
			}
		}
}


template<typename F, typename FPanel>
void pcolvgen (Matgen<F> const& mg, Panels<FPanel>const & p, double* dx){
	int nprow = p.nprow;
	int b = p.b;
	int i1 = p.i1;
	int j1 = p.j1;
	int istride = p.istride;
	int jstride = p.jstride;
	for(int i=0; i<nprow; ++i){
		int ipos = i1 + i*istride;
		if(ipos%jstride == j1){
			fill_one_panel_with_rand(mg.n, b*ipos, mg.n, b, 1, dx+b*i, 1, mg.seed, false);
		}
	}
}

template<typename F, typename FPanel>
void pdiaggen (Matgen<F> const& mg, Panels<FPanel>const & p, double* dx){
	int nprow = p.nprow;
	int b = p.b;
	int i1 = p.i1;
	int j1 = p.j1;
	int istride = p.istride;
	int jstride = p.jstride;
	for(int i=0; i<nprow; ++i){
		int ipos = i1 + i*istride;
		if(ipos%jstride == j1){
			#pragma omp parallel for
			for(int k=0; k<b; ++k)
				dx[b*i+k] = calc_diag(b*ipos+k, mg.n, mg.seed);
		}
	}
}

template<typename F, typename FPanel>
void pdiaggen (HMGen<F> const& mg, Panels<FPanel>const & p, double* dx){
	int nprow = p.nprow;
	int b = p.b;
	int i1 = p.i1;
	int j1 = p.j1;
	int istride = p.istride;
	int jstride = p.jstride;
	F const ab = mg.alpha * mg.beta;
	F const done = 1;
	for(int i=0; i<nprow; ++i){
		int ipos = i1 + i*istride;
		if(ipos%jstride == j1){
			#pragma omp parallel for
			for(int k=0; k<b; ++k)
				dx[b*i+k] = done + ab * (b*ipos+k);
		}
	}
}

#endif
