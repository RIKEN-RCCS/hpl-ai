#ifndef PANEL_CHECK_HPP
#define PANEL_CHECK_HPP
// computing checksum of matrix for debuging.
#include "panel.hpp"
#include "grid.hpp"
#include "highammgen.hpp"
#include <mpi.h>

template<typename FPanel>
void panel_check(Panels<FPanel>const& p, Grid& g)
{
	double sigs[3] = {0., 0., 0.};
	double sigd=0., sigu=0., sigl=0.;
	size_t lda = p.lda;
	int b = p.b;
	int i1 = p.i1;
	int j1 = p.j1;
	int istride = p.istride;
	int jstride = p.jstride;
	int nprow = p.nprow;
	int npcol = p.npcol;
	//#pragma omp parallel for collapse(2) schedule(dynamic) reduction(+:sigd,sigu,sigl)
	for(int j=0; j<npcol; ++j){
		for(int i=0; i<nprow; ++i){
			int jpos = j1 + j*jstride;
			int ipos = i1 + i*istride;
			FPanel const* data = p(i, j);
			if(ipos == jpos){
				for(int jj=0; jj<b; ++jj){
					for(int ii=0; ii<jj; ++ii){
						double t = fabs(data[jj*lda+ii]);
						sigu += t;
					}
					sigd += fabs(data[jj*lda+jj]);
					for(int ii=jj+1; ii<b; ++ii){
						double t = fabs(data[jj*lda+ii]);
						sigl += t;
					}
				}
			}
			else if(ipos < jpos){
				for(int jj=0; jj<b; ++jj)
					for(int ii=0; ii<b; ++ii){
						double t = fabs(data[jj*lda+ii]);
						sigu += t;
					}
			}
			else {
				for(int jj=0; jj<b; ++jj)
					for(int ii=0; ii<b; ++ii){
						double t = fabs(data[jj*lda+ii]);
						sigl += t;
					}
			}
		}
	}
	sigs[0] = sigd;
	sigs[1] = sigu;
	sigs[2] = sigl;
	MPI_Allreduce(MPI_IN_PLACE, sigs, 3, MPI_DOUBLE, MPI_SUM, g.commworld);
	if(g.row==0 && g.col==0) {
		std::printf("check %22.17e %22.17e, %22.17e\n", sigs[0], sigs[1], sigs[2]);
		std::fflush(stdout);
	}
}

template<typename FPanel>
void panel_check(HMGen<double> const& hmg, Panels<FPanel>const& p, Grid& g)
{
	double sigs[3] = {0., 0., 0.};
	double sigd=0., sigu=0., sigl=0.;
	size_t lda = p.lda;
	int b = p.b;
	int i1 = p.i1;
	int j1 = p.j1;
	int istride = p.istride;
	int jstride = p.jstride;
	int nprow = p.nprow;
	int npcol = p.npcol;
	double alpha = hmg.alpha;
	double beta = hmg.beta;
	double done = 1;
	//#pragma omp parallel for collapse(2) schedule(dynamic) reduction(+:sigd,sigu,sigl)
	for(int j=0; j<npcol; ++j){
		for(int i=0; i<nprow; ++i){
			int jpos = j1 + j*jstride;
			int ipos = i1 + i*istride;
			FPanel const* data = p(i, j);
			double blockmax = 0.;
			if(ipos == jpos){
				for(int jj=0; jj<b; ++jj){
					for(int ii=0; ii<jj; ++ii){
						double t = fabs(data[jj*lda+ii]-beta)/beta;
						//printf("9871 %d %d %e %e %e\n", b*ipos+ii, b*jpos+jj, data[jj*lda+ii], beta, t);
						sigu = sigu > t ? sigu: t;
						blockmax = blockmax > t ? blockmax: t;
					}
					double t = fabs(data[jj*lda+jj]-done);
					//if(t>1e-1) printf("9871 %d %d %e %e %e\n", b*jpos+jj, b*jpos+jj, data[jj*lda+jj], done, t);
					sigd = sigd > t ? sigd: t;
					blockmax = blockmax > t ? blockmax: t;
					for(int ii=jj+1; ii<b; ++ii){
						double t = fabs(data[jj*lda+ii]-alpha)/alpha;
						//printf("9871 %d %d %e %e %e\n", b*ipos+ii, b*jpos+jj, data[jj*lda+ii], alpha, t);
						sigl = sigl > t ? sigl: t;
						blockmax = blockmax > t ? blockmax: t;
					}
				}
			}
			else if(ipos < jpos){
				for(int jj=0; jj<b; ++jj)
					for(int ii=0; ii<b; ++ii){
						double t = fabs(data[jj*lda+ii]-beta)/beta;
						//printf("9871 %d %d %e %e %e\n", b*ipos+ii, b*jpos+jj, data[jj*lda+ii], beta, t);
						sigu = sigu > t ? sigu: t;
						blockmax = blockmax > t ? blockmax: t;
					}
			}
			else {
				for(int jj=0; jj<b; ++jj)
					for(int ii=0; ii<b; ++ii){
						double t = fabs(data[jj*lda+ii]-alpha)/alpha;
						//printf("9871 %d %d %e %e %e\n", b*ipos+ii, b*jpos+jj, data[jj*lda+ii], alpha, t);
						sigl = sigl > t ? sigl: t;
						blockmax = blockmax > t ? blockmax: t;
					}
			}
			/*if(g.row==0&&g.col==0) {
				printf("9871 %d %d %e\n", ipos, jpos, blockmax);
				fflush(stdout);
			}*/
		}
	}
	sigs[0] = sigd;
	sigs[1] = sigu;
	sigs[2] = sigl;
	MPI_Allreduce(MPI_IN_PLACE, sigs, 3, MPI_DOUBLE, MPI_MAX, g.commworld);
	if(g.row==0 && g.col==0) {
		std::printf("check %22.17e %22.17e, %22.17e\n", sigs[0], sigs[1], sigs[2]);
		std::fflush(stdout);
	}
}


#endif
