#ifndef GRID_HPP
#define GRID_HPP
#include <mpi.h>
#include <cassert>
#include <cstring>
#include "fp16sim.hpp"

#ifdef __APPLE__
#define aligned_alloc(alignment, size) malloc(size)
#endif
#if (defined __FUJITSU) || (defined __CLANG_FUJITSU)
#include <omp.h>
#include <mpi-ext.h>
#include "remap.hpp"
#endif

enum NumaMap {
	// How to destribute NUMA processes to the process grid.
	ROWCONT, // continuous in row
	COLCONT, // continuous in column
	ROWDIST, // distributed (cyclic) over row
	COLDIST, // distributed (cyclic) over column
	CONT2D // continuous in 2x2. this is only for nnuma==4
};

struct Grid {
	// vcomm is a communicator for vertical communication (inside a column)
	// row = id(vcomm), nrow = sz(vcomm)
	// hcomm is a communicator for horizontal communication (inside a row)
	// col = id(hcomm), ncol = sz(hcomm)
	int row, col;
	int nrow, ncol;
	int idnuma, nnuma;
	MPI_Comm vcomm, hcomm, commworld;
	Grid(MPI_Comm comm, int nrow, int numasize=0, NumaMap map=NumaMap::ROWCONT): commworld(comm) {
		assert(numasize>=0);
		assert(numasize!=0 || map!=NumaMap::ROWCONT);

		int rank, size;
		MPI_Comm_rank(comm, &rank);
		MPI_Comm_size(comm, &size);
		if (size % nrow) MPI_Abort(MPI_COMM_WORLD, 4);
		int ncol = size / nrow;
		int myrow, mycol;
		if(numasize==0){
			idnuma = 0;
			nnuma = 1;
			myrow = rank % nrow;
			mycol = rank / nrow;
		}
		#if defined(__FUJITSU) || defined(__CLANG_FUJITSU)
		// special mappings for Fugaku
		else if(size == (22*20*24*2*3*2*4) && nrow==1056 && numasize==4 && map==NumaMap::CONT2D){ // case 528x240
			if(rank == 0){
				fprintf(stdout, "!REMAP FOR 330 RACKS\n");
				fprintf(stderr, "!REMAP FOR 330 RACKS\n");
				fflush(stdout);
				fflush(stderr);
			}
			int noderow, nodecol;
			int coords[6];
			remap330(noderow, nodecol, coords);

			idnuma = rank % numasize;
			nnuma = numasize;

			myrow = 2*noderow + idnuma%2;
			mycol = 2*nodecol + idnuma/2;
		}
		else if(size == (24*20*24 * 2*3*2 * 4) && nrow==480*2 && numasize==4 && map==NumaMap::CONT2D){ // case 480x288
			if(rank == 0){
				fprintf(stdout, "!REMAP FOR 330 RACKS\n");
				fprintf(stderr, "!REMAP FOR 330 RACKS\n");
				fflush(stdout);
				fflush(stderr);
			}
			int noderow, nodecol;
			int coords[6];
			remap360(noderow, nodecol, coords);

			idnuma = rank % numasize;
			nnuma = numasize;

			myrow = 2*noderow + idnuma%2;
			mycol = 2*nodecol + idnuma/2;
		}
		else if(size == (24*22*24 * 2*3*2 * 4) && nrow==528*2 && numasize==4 && map==NumaMap::CONT2D){ // case 528x288
			if(rank == 0){
				fprintf(stdout, "!REMAP FOR 330 RACKS\n");
				fprintf(stderr, "!REMAP FOR 330 RACKS\n");
				fflush(stdout);
				fflush(stderr);
			}
			int noderow, nodecol;
			int coords[6];

			remap392(noderow, nodecol, coords);

			idnuma = rank % numasize;
			nnuma = numasize;

			myrow = 2*noderow + idnuma%2;
			mycol = 2*nodecol + idnuma/2;
		}
		#endif
		else {
			assert(size%numasize == 0);
			idnuma = rank % numasize;
			nnuma = numasize;
			switch(map){
			case NumaMap::ROWCONT:{
				assert(nrow%nnuma == 0);
				myrow = rank % nrow;
				mycol = rank / nrow;
			} break;

			case NumaMap::COLCONT:{
				assert((size/nrow)%nnuma == 0);
				int t = rank / nnuma;
				myrow = t % nrow;
				mycol = (t / nrow) * nnuma + idnuma;
			} break;

			case NumaMap::ROWDIST:{
				assert(nrow%nnuma == 0);
				int rs = nrow / nnuma;
				int t = rank / nnuma;
				myrow = (t%rs) + idnuma * rs;
				mycol = rank / nrow;
			} break;

			case NumaMap::COLDIST:{
				assert((size/nrow)%nnuma == 0);
				int t = rank / nnuma + (size/nnuma) * idnuma;
				myrow = t % nrow;
				mycol = t / nrow;
			} break;

			case NumaMap::CONT2D: {
				assert(nnuma%2==0); // others are not implemented yet
				assert(nrow%2==0);
				assert((size/nrow)%(nnuma/2)==0);
				int t = rank / nnuma;
				int grow = t%(nrow/2);
				int gcol = t/(nrow/2);
				myrow = grow*2 + idnuma%2;
				mycol = gcol*(nnuma/2) + idnuma/2;
			} break;
			default:
				std::abort();
			}
		}
		#if 0
		// DEBUG, shuffle the rankmap up
		if((myrow%2)==0) mycol = (mycol+12)%ncol;
		if((myrow/2%2)==0) mycol = (mycol+6)%ncol;
		if((mycol%3)==1) myrow = (myrow+9)%nrow;
		if((mycol%3)==2) myrow = (myrow+15)%nrow;
		#endif

		MPI_Comm_split(comm, mycol, myrow, &vcomm);
		MPI_Comm_split(comm, myrow, mycol, &hcomm);
		this->row = myrow;
		this->col = mycol;
		this->nrow = nrow;
		this->ncol = ncol;
	}
	~Grid() {
		MPI_Comm_free(&vcomm);
		MPI_Comm_free(&hcomm);
	}
};


template <typename T>
struct Mpi_type_wrappe{};

template <>
struct Mpi_type_wrappe<fp16>{
	operator MPI_Datatype(){ return MPI_SHORT; }
};

template <>
struct Mpi_type_wrappe<float>{
	operator MPI_Datatype(){ return MPI_FLOAT; }
};

template <>
struct Mpi_type_wrappe<double>{
	operator MPI_Datatype(){ return MPI_DOUBLE; }
};

template <typename F>
struct T2MPI {
	static Mpi_type_wrappe<F> type;
};

template <typename F>
Mpi_type_wrappe<F> T2MPI<F>::type;

#endif
