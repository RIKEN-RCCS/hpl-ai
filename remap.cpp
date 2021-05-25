#include <assert.h>

#if defined(__FUJITSU) || defined(__CLANG_FUJITSU)
#include <mpi.h>
#include <mpi-ext.h>
#endif
#ifndef UNIT_TEST
#include <mpi.h>
#endif

#ifdef UNIT_TEST
// fake comm ranks
#define MPI_COMM_WORLD 1
int comm_rank;
int MPI_Comm_rank(int /*comm*/, int* rank)
{
	*rank = comm_rank;
	return 0;
}

#define FJMPI_SUCCESS 0
#define FJMPI_ERR_TOPOLOGY_NODE_SHARED_JOB 1
#define FJMPI_TOFU_REL 2
int nx, ny, nz, na, nb, nc;
int FJMPI_Topology_get_coords(int comm, int rank, int view, int maxdims, int* coords)
{
	assert(comm==MPI_COMM_WORLD);
	assert(view==FJMPI_TOFU_REL);
	assert(maxdims==6);
	coords[5] = rank % nc;
	rank /= nc;
	coords[4] = rank % nb;
	rank /= nb;
	coords[3] = rank % na;
	rank /= na;
	coords[0] = rank % nx;
	rank /= nx;
	coords[1] = rank % ny;
	rank /= ny;
	coords[2] = rank % nz;
	rank /= nz;
	return FJMPI_SUCCESS;
}
#endif // UNIT_TEST

static int remap2d(int x, int y, int xsize, int ysize)
{
	assert(xsize%2 == 0);
	if(y==0) return x;
	if(x%2){
		// up
		int shift = xsize + (ysize-1) * (xsize-x-1);
		return shift + y - 1;
	}
	else {
		// down
		int shift = xsize + (ysize-1) * (xsize-x-1);
		return shift + ysize - y - 1;
	}
}

#if defined(__FUJITSU) || defined(__CLANG_FUJITSU)
void remap330(int& row, int& col, int coords[6])
{
	int mpirank;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
	int error = FJMPI_Topology_get_coords(MPI_COMM_WORLD, mpirank, FJMPI_TOFU_REL, 6, coords);
	assert(error == FJMPI_SUCCESS);
	int x = coords[0];
	int y = coords[1];
	int z = coords[2];
	int a = coords[3];
	int b = coords[4];
	int c = coords[5];
	assert(x>=0 && x<22);
	assert(y>=0 && y<20);
	assert(z>=0 && z<24);
	assert(a>=0 && a<2);
	assert(b>=0 && b<3);
	assert(c>=0 && c<2);

	// build row rank
#if 0
	// row = YZ = 20x24
	row = remap2d(y, z, 20, 24);
	assert(row>=0 && row < 20*24);

	// build col rank
	// col = ((Xb)a)c = 22*3*2*2
	int xb = remap2d(x, b, 22, 3);
	int xba = remap2d(xb, a, 66, 2);
	col = remap2d(xba, c, 132, 2);
	assert(col>=0 && col < 22*3*2*2);
#else
	// row = XZ = 22x24
	row = remap2d(x, z, 22, 24);
	assert(row>=0 && row < 22*24);

	// build col rank
	// col = ((Xb)a)c = 22*3*2*2
	int yb = remap2d(y, b, 20, 3);
	int yba = remap2d(yb, a, 60, 2);
	col = remap2d(yba, c, 120, 2);
	assert(col>=0 && col < 20*3*2*2);
#endif
}
#endif
#if defined(__FUJITSU) || defined(__CLANG_FUJITSU)
static void remap_XY_Zabc(int& row, int& col, int coords[6], int const xdim, int const ydim, int const zdim)
{
	int mpirank;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
	int error = FJMPI_Topology_get_coords(MPI_COMM_WORLD, mpirank, FJMPI_TOFU_REL, 6, coords);
	assert(error == FJMPI_SUCCESS);
	int x = coords[0];
	int y = coords[1];
	int z = coords[2];
	int a = coords[3];
	int b = coords[4];
	int c = coords[5];
	assert(x>=0 && x<xdim);
	assert(y>=0 && y<ydim);
	assert(z>=0 && z<zdim);
	assert(a>=0 && a<2);
	assert(b>=0 && b<3);
	assert(c>=0 && c<2);

	row = remap2d(x, y, xdim, ydim);
	assert(row>=0 && row < xdim*ydim);

	// build col rank
	int const zb = remap2d(z, b, zdim, 3);
	int const zba = remap2d(zb, a, zdim*3, 2); 
	col = remap2d(zba, c, zdim*3*2, 2);
	assert(col>=0 && col < zdim*3*2*2);
}
#endif

#if defined(__FUJITSU) || defined(__CLANG_FUJITSU)
void remap360(int& row, int& col, int coords[6]) // for 24x20x24
{
	remap_XY_Zabc(row, col, coords, 24, 20, 24);
}
#endif

#if defined(__FUJITSU) || defined(__CLANG_FUJITSU)
void remap392(int& row, int& col, int coords[6]) // for 24x22x24
{
	remap_XY_Zabc(row, col, coords, 24, 22, 24);
}
#endif

#ifdef UNIT_TEST
#include <stdio.h>
int main()
{
	nx = 22;
	ny = 20;
	nz = 24;
	na = 2;
	nb = 3;
	nc = 2;
	int nrank = 22*20*24*2*3*2;
	for(int i=0; i<nrank; ++i){
		comm_rank = i;
		int coords[6];
		int row, col;
		remap330(row, col, coords);
		printf("%d %d %d %d %d %d %d %d %d\n", i, row, col,
			coords[0], coords[1], coords[2], coords[3], coords[4], coords[5]);
	}
	return 0;
}
#endif

