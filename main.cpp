
#include "panel.hpp"
#include "grid.hpp"
#include "tofu.hpp"
#include "iterative_refinement.hpp"
#include "panel_norm.hpp"
#include "panel_check.hpp"
#include "hpl_rand.hpp"
#include "panel_trf.hpp"
#include "matgen.hpp"
#include "fp16sim.hpp"
#include "timer.hpp"
#include "kernels/kernel.h"
#include "lda_tuner.hpp"
#include "highammgen.hpp"
#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <ctime>
#include <utility>

static int down_clock();

// FHIGH: the floating-point type for panel decompositions
// FLOW : that for GEMM

#define FHIGH float
#define FLOW fp16

static void get_flags(int argc, char** argv, bool& tile, int& dd_layout,
	bool& manual, bool& rdma, bool& warmup, bool& pack, bool &checksum,
	int& numasize, NumaMap& numamap, int& nbuf, int& jobid, bool& l1est, bool& higham, bool& pwrtune)
{
	tile = true;
	dd_layout = 0;
	manual = false;
	rdma = false;
	warmup = false;
	pack = false;
	numasize = 0;
	numamap = NumaMap::ROWDIST;
	nbuf = 2;
	jobid = 0;
	l1est = false;
	higham = false;
	pwrtune = false;
	for(int i=0; i<argc; ++i){
		if(!std::strcmp(argv[i], "-not") || !std::strcmp(argv[i], "-nt") || !std::strcmp(argv[i], "--notile")) tile = false;
		else if(!std::strcmp(argv[i], "-d") || !std::strcmp(argv[i], "--delayed")) dd_layout = 1;
		else if(!std::strcmp(argv[i], "-l") || !std::strcmp(argv[i], "--lazy-init")) dd_layout = 1;
		else if(!std::strcmp(argv[i], "--back_buffer")) dd_layout = 2;
		else if(!std::strcmp(argv[i], "-m") || !std::strcmp(argv[i], "--manual-progress")) manual = true;
		else if(!std::strcmp(argv[i], "-r") || !std::strcmp(argv[i], "--rdma")) rdma = true;
		else if(!std::strcmp(argv[i], "-w") || !std::strcmp(argv[i], "--warmup")) warmup= true;
		else if(!std::strcmp(argv[i], "--pack")) pack = true;
		else if(!std::strcmp(argv[i], "--checksum")) checksum = true;
		else if(!std::strcmp(argv[i], "--numa")) {
			if(i==argc-1) break;
			numasize = std::atoi(argv[i+1]);
			++i;
		}
		else if(!std::strcmp(argv[i], "--numa-map")){
			if(i==argc-1) break;
			if(!std::strcmp(argv[i+1], "ROWCONT")) numamap = NumaMap::ROWCONT;
			else if(!std::strcmp(argv[i+1], "COLCONT")) numamap = NumaMap::COLCONT;
			else if(!std::strcmp(argv[i+1], "ROWDIST")) numamap = NumaMap::ROWDIST;
			else if(!std::strcmp(argv[i+1], "COLDIST")) numamap = NumaMap::COLDIST;
			else if(!std::strcmp(argv[i+1], "CONT2D")) numamap = NumaMap::CONT2D;
			++i;
		}
		else if(!std::strcmp(argv[i], "--nbuf")) {
			if(i==argc-1) break;
			nbuf = std::atoi(argv[i+1]);
			++i;
		}
		else if(!std::strcmp(argv[i], "--job-id")) {
			if(i==argc-1) break;
			jobid = std::atoi(argv[i+1]);
			++i;
		}
		else if(!std::strcmp(argv[i], "--l1est")) l1est = true;
		else if(!std::strcmp(argv[i], "--highammat")) higham = true;
		else if(!std::strcmp(argv[i], "--pwrtune")) pwrtune = true;
		else if(!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
			const char* helpmessage =
R"*****(HPL-AI for MPI

usage: mpirun -n <numprocs> [mpiarguments] ./driver.out <n> <b> <P> [arguments]

Arguments:
   n                       The matrix size.
   b                       The block size. Requires mod(n,b) = 0.
   P                       The first dimension of the process grid. Requires mod(numprocs,P) = 0.
   -not or -nt             Disable tile layout. Default uses tile layout.
   --lazy-init or -l       Enable double-decker layout. Disabled by default.
   --delayed or -d         Same as --lazy-init and -l.
   --back_buffer           Enable back-buffer. This introduces --lazy-init. Disabled by default
   --rdma or -r            Enable RDMA communication (with TofuD). Disabled by default.
   --warmup or -w          Interrupt the decomposition for benchmarking purpose. Disabled by default.
   --pack                  Enable packed data layout for L and U panels. Disabled by default. Force disabled if not supported.
   --checksum              Compute checksum after the decomposition for debugging purpose. Disabled by default.
   --numa <nnumaprocs>     Set the number of numa processes. It implicitly assumes that the MPI-ranks of numa processes in a node are continuous. Default uses nnumaprocs = 1.
   --numa-map <numamap>    How to locate numa-processes to the process grid.
                           ROWCONT : continuous in row. (Default)
                           COLCONT : continuous in column.
                           ROWDIST : distributed over row.
                           COLDIST : distributed over col.
                           CONT2D  : continuoust in 2x2 grid. (only for nnuma = 4)
   --nbuf <nbuf>           The number of working buffer for broadcasting L and U panels. This is effective with -r. nbuf >= 2. Default uses nbuf = 2.
   --job-id <job-id>       An integer tag for the output of the timing statistics. Default uses job-id = 0.
   --l1est                 Approximate LU factors and skip decomposition for debugging purpose. Disabled by default.
   --highammat             Change the initial matrix to the Higham's HPL-AI matrix. See "https://github.com/higham/hpl-ai-matrix" for details of the matrix. Disabled by default.
   --pwrtune               (For Fugaku) down-clock in IR phase. Disabled by default.
   --help or -h            Print this messages.
)*****";
			printf("%s", helpmessage);
		}
		else{
			if(i > 0){
				printf("Ignoring unknown option: %s\n", argv[i]);
			}
		}
	}
	#ifndef HGEMM_PACK
	// XXX force disable pack if not implemented
	pack = false;
	#endif
}

static void print_ctime(const char *msg, std::FILE *fp=stdout){
	std::time_t t;
	std::time(&t);
	std::fprintf(fp, "%s%s", msg, std::ctime(&t));
}

static int gcd(int a, int b){
	if(a<b) std::swap(a, b);
	return b ? gcd(b, a%b) : a;
}

int main(int argc, char **argv)
{
	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
	{
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if(0 == rank){
			fprintf(stdout, "done MPI_Init_thread, provided = %d\n", provided);
			print_ctime("#MPI_Init_thread: ", stdout);
			fflush(stdout);
		}
	}
	int n = 120;
	int b = 3;
	int nrow = 2;
	bool tile_layout = false;
	int dd_layout = 0;
	bool manual_progress = false;
	bool rdma = false;
	bool warmup = false;
	bool pack = false;
	bool checksum = false;
	int numasize = 0;
	int nbuf = 2;
	int jobid = 0;
	bool l1est = false;
	bool higham = false;
	bool pwrtune = false;
	NumaMap numamap = NumaMap::ROWDIST;
	int seed = 42;
	if(argc > 1) n = std::atoi(argv[1]);
	if(argc > 2) b = std::atoi(argv[2]);
	if(argc > 3) nrow = std::atoi(argv[3]);
	get_flags(argc-3, argv+3, tile_layout, dd_layout, manual_progress,
		rdma, warmup, pack, checksum, numasize, numamap, nbuf, jobid, l1est, higham, pwrtune);

	assert(!pack || (pack&&rdma));
	assert(!rdma || (rdma&&!tile_layout));


	{
		// memory allocations and set-up of pandel descriptions
		Grid g(MPI_COMM_WORLD, nrow, numasize, numamap);
		Panels<FHIGH> p;
		LRPanels<FLOW> lr[nbuf*2];
		build_panels(n, b, tile_layout, pack, p, lr, g.row, g.col, g.nrow, g.ncol, nbuf);

		auto lcm = [](int a, int b){ return a * b / gcd(a, b); };
		int epoch_size = b * lcm(g.nrow, g.ncol);
		p.epoch_size = epoch_size;

		FHIGH* piv;
		assert(posix_memalign(reinterpret_cast<void**>(&piv), 256, sizeof(FHIGH)*b*b*4) == 0);

		size_t ldv = b * (p.npcol + p.nprow);
		double*d = static_cast<double*>(std::malloc(sizeof(double) * ldv));
		FHIGH*dhigh = static_cast<FHIGH*>(std::malloc(sizeof(FHIGH) * ldv));
		double*rhs = static_cast<double*>(std::malloc(sizeof(double) * ldv));
		double*v = static_cast<double*>(std::malloc(sizeof(double) * ldv));
		bool wallocated = false;
		double*w;
		if(sizeof(double)*(ldv*10+2*b*b) < sizeof(FLOW)*p.lda*b*2) {
			w = reinterpret_cast<double*>(lr[0].p);
			wallocated = false;
		}
		else {
			w = static_cast<double*>(malloc(sizeof(double)*(ldv*10+2*b*b)));
			wallocated = true;
		}

		// print-out settings
		if(g.row == 0 && g.col == 0){
			std::printf("jobid=%d\n", jobid);
			std::printf("n=%d b=%d r=%d c=%d\n", n, b, g.nrow, g.ncol);
			std::printf("%s %s %s %s %s %s %s %s\n",
				tile_layout?"tile": "2dbc",
				dd_layout==0? "precomp":
				dd_layout==1? "lazy": "back_buffer",
				rdma?"rdma":"mpi",
				warmup?"warmup": "full",
				pack?"pack": "nopack",
				checksum?"checksum": "nocheck",
				l1est?"skiplu":"noskiplu",
				higham?"hmat": "ddmat"
				);
			std::printf("numasize=%d numamap=%s nbuf=%d\n", numasize,
				numamap==NumaMap::ROWCONT?"ROWCONT":
				(numamap==NumaMap::COLCONT?"COLCONT":
				(numamap==NumaMap::ROWDIST?"ROWDIST":
				(numamap==NumaMap::COLDIST?"COLDIST":
				(numamap==NumaMap::CONT2D?"CONT2D":"ERROR")))),
				nbuf);
			std::printf("epoch_size = %d\n", epoch_size);
			if(n % epoch_size){
				std::fprintf(stdout, "WARNING: n%%epoch_size=%d\n", n%epoch_size);
				std::fprintf(stderr, "WARNING: n%%epoch_size=%d\n", n%epoch_size);
			}
			print_ctime("#BEGIN: ", stdout);
			print_ctime("#BEGIN: ", stderr);
			std::fflush(stdout);
			std::fflush(stderr);
		}

		/*if(rdma&&pack)
			tune_lda<FHIGH,FLOW,true,true,true>(p, lr, g.row==0 && g.col==0);
		else if(delayed_update)
			tune_lda<FHIGH,FLOW,true,true,false>(p, lr, g.row==0 && g.col==0);
		else
			tune_lda<FHIGH,FLOW,true,false,false>(p, lr, g.row==0 && g.col==0);*/

		// initialize matrices. 
		Matgen<FHIGH> mg(seed, n, p.b*(p.istride-1), p.b*(p.jstride-1), dhigh);
		Matgen<double> mgir(seed, n, p.b*(p.istride-1), p.b*(p.jstride-1), d);
		HMGen<FHIGH> hmg(n, 1000., 0.5, dhigh);
		HMGen<double> hmgd(n, 1000., 0.5, d);
		if(dd_layout && !l1est)
			pmatgen0(p);
		else if(!higham)
			pmatgen(mg, p);
		else
			pmatgen(hmg, p);
		pcolvgen(mgir, p, rhs);
		if(!higham)
			pdiaggen(mgir, p, d);
		else
			pdiaggen(hmgd, p, d);
		copycolv(p, rhs, v);
		copycolv(p, d, dhigh);
		double normb = colv_infnorm(p, rhs, g);
		double norma;
		if(!higham) norma = hpl_infnorm(p, d, g);
		else norma = higham_infnorm(hmgd, p, w, g);
		//double norma = panel_infnorm(mg, p, w, piv, g);


		// call blas functions for warming up.
		warmup_trf(b, piv, b);

		double start_time, stop_time;
		IRErrors er;
		if(l1est){
			// Compute approximated LU decomp.
			// This will be useful to debug IR process or compare the checksum with later one.
			Timer::beg(Timer::TOTAL);

			// start
			start_time = MPI_Wtime();
			MPI_Barrier(MPI_COMM_WORLD); // ensure all the procs start after
			if(!higham)
				pmatl1est(mg, p);
			else
				pmatl1est(hmg, p);

			if(checksum){
				if(!higham)
					panel_check(p, g);
				else
					panel_check(hmgd, p, g);
			}
			
			if(!higham)
				er = iterative_refinement(p, mgir, v, w, ldv, rhs, norma, normb, (warmup? 1: 50), g);
			else
				er = iterative_refinement(p, hmgd, v, w, ldv, rhs, norma, normb, (warmup? 1: 50), g);
			MPI_Barrier(MPI_COMM_WORLD); // ensure all the procs stop before
			// stop
			stop_time = MPI_Wtime();
			Timer::end(Timer::TOTAL);
		}
		else if(!rdma){
			// MPI implementation of LU decomp. The performance is hardly depends on the MPI_Ibcast.

			#if (defined __FUJITSU) || (defined __CLANG_FUJITSU)
			// Start the progress (or spare) thread for MPI communication.
			FJMPI_Progress_start();
			#endif
			MPI_Barrier(MPI_COMM_WORLD); // to callibrate the epoc time
			Timer::initialize();
			Timer::beg(Timer::TOTAL);

			// start
			start_time = MPI_Wtime();
			MPI_Barrier(MPI_COMM_WORLD); // ensure all the procs start after
			if(dd_layout==1){
				if(!higham)
					panel_lu_async<FHIGH, FLOW, Matgen, 1, true>(p, lr, mg, piv, b, g, warmup);
				else
					panel_lu_async<FHIGH, FLOW, HMGen, 1, true>(p, lr, hmg, piv, b, g, warmup);
			}
			else{
				if(!higham)
					panel_lu_async<FHIGH, FLOW, Matgen, 0, false>(p, lr, mg, piv, b, g, warmup);
				else	
					panel_lu_async<FHIGH, FLOW, HMGen, 0, false>(p, lr, hmg, piv, b, g, warmup);
			}
			MPI_Barrier(MPI_COMM_WORLD); // for extra safety
			#if (defined __FUJITSU) || (defined __CLANG_FUJITSU)
			FJMPI_Progress_stop();
			#endif

			if(checksum){
				if(!higham)
					panel_check(p, g);
				else
					panel_check(hmgd, p, g);
			}
			
			if(!higham)
				er = iterative_refinement(p, mgir, v, w, ldv, rhs, norma, normb, (warmup? 1: 50), g);
			else
				er = iterative_refinement(p, hmgd, v, w, ldv, rhs, norma, normb, (warmup? 1: 50), g);
			MPI_Barrier(MPI_COMM_WORLD); // ensure all the procs stop before
			// stop
			stop_time = MPI_Wtime();
			Timer::end(Timer::TOTAL);

		}
		else {
			// This branch uses TofuD native apis.
			// On platforms with no Tofu, it is deligated to MPI send/recv communications.
			TofudComm lcom(g.hcomm), rcom(g.vcomm);
			int tni1, tni2;
			tofu_tni_mapping(0, g.idnuma, g.nnuma, TofudMapping::DEFAULT, tni1, tni2);
			lcom.config(tni1, tni2);
			tofu_tni_mapping(1, g.idnuma, g.nnuma, TofudMapping::DEFAULT, tni1, tni2);
			rcom.config(tni1, tni2);
			
			RDMAPanelLU<FHIGH,FLOW,TofudComm> plu(p, lr, piv, b, g, lcom, rcom, nbuf);

			MPI_Barrier(MPI_COMM_WORLD); // to callibrate the epoc time
			Timer::initialize();
			Timer::beg(Timer::TOTAL);
			
			// start
			start_time = MPI_Wtime();
			plu.start_time = start_time;
			MPI_Barrier(MPI_COMM_WORLD); // ensure all the procs start after
			if(pack){
				if(dd_layout==1){
					if(!higham)
						plu.run<Matgen<FHIGH>, 1, true, true>(mg, warmup);
					else
						plu.run<HMGen<FHIGH>, 1, true, true>(hmg, warmup);
				}
				else if(dd_layout==2) {
					if(!higham)
						plu.run<Matgen<FHIGH>, 2, true, true>(mg, warmup);
					else
						plu.run<HMGen<FHIGH>, 2, true, true>(hmg, warmup);
				}
			}
			else {
				if(dd_layout==1){
					if(!higham)
						plu.run<Matgen<FHIGH>, 1, true, false>(mg, warmup);
					else
						plu.run<HMGen<FHIGH>, 1, true, false>(hmg, warmup);
				}
				else if(dd_layout==2){
					if(!higham)
						plu.run<Matgen<FHIGH>, 2, true, false>(mg, warmup);
					else
						plu.run<HMGen<FHIGH>, 2, true, false>(hmg, warmup);
				}
				else{
					if(!higham)
						plu.run<Matgen<FHIGH>, 0, false, false>(mg, warmup);
					else
						plu.run<HMGen<FHIGH>, 0, false, false>(hmg, warmup);
				}
			}
			MPI_Barrier(MPI_COMM_WORLD); // for extra safety

			if(pwrtune) down_clock();
			
			if(checksum){
				if(!higham)
					panel_check(p, g);
				else
					panel_check(hmgd, p, g);
			}

			if(!higham)
				er = iterative_refinement(p, mgir, v, w, ldv, rhs, norma, normb, (warmup? 1: 50), g);
			else
				er = iterative_refinement(p, hmgd, v, w, ldv, rhs, norma, normb, (warmup? 1: 50), g);
			MPI_Barrier(MPI_COMM_WORLD);
			// stop
			stop_time = MPI_Wtime(); // ensure all the procs stop before 
			Timer::end(Timer::TOTAL);
		}

		MPI_Barrier(MPI_COMM_WORLD);

		if(g.row == 0 && g.col == 0){
			double time_duration = stop_time - start_time;
			double hplflops = 2./3*n*n*n+3./2*n*n;
			if(warmup){
				int nn = n-(g.nrow+2)*b;
				double skip = 2./3 * nn * nn * nn + 3./2*nn * nn;
				hplflops -= skip;
			}

			print_ctime("#END__: ", stdout);
			print_ctime("#END__: ", stderr);
			
			std::printf("%.9f sec. %.9f GFlop/s resid = %20.15e hpl-harness = %.9f\n",
				time_duration, hplflops/time_duration*1e-9, er.residual, er.hpl_harness);
			std::fflush(stdout);
#if (defined __FUJITSU) || (defined __CLANG_FUJITSU)
			{
				int procs, threads;
				MPI_Comm_size(MPI_COMM_WORLD, &procs);
#pragma omp parallel
				{
					threads = omp_get_num_threads();
				}
				double peak = 256.e9 * threads * procs; // 32-SIMD dual FMA (half) in 2.0 GHz
				double flops = hplflops/time_duration;
				double ratio = 100.0 * flops / peak;
				std::printf("%f Pflops, %f %%\n", flops*1.e-15, ratio);
				std::fflush(stdout);
			}
#endif
		}
		{
			int rank, size;
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			MPI_Comm_size(MPI_COMM_WORLD, &size);
			static char filename[1024];
			std::sprintf(filename, (warmup?"Warmup.%06d.%09d.%04d.%05d.%05d": "Timerdump.%06d.%09d.%04d.%05d.%05d"), jobid, n, b, size, rank);
			if(size<10 || rank==0 || rank==size-1 || rank == g.nrow || rank == (g.nrow+1)*(g.ncol/2) || rank == size-g.nrow)
				Timer::dump_mp(size, rank, g.row, g.col, filename);
		}


		destruct_panels(p, lr);
		std::free(piv);
		std::free(rhs);
		std::free(v);
		if(wallocated) std::free(w);
		std::free(d);
		std::free(dhigh);
	}

	MPI_Finalize();
	return 0;
}

// #if (defined __FUJITSU) || (defined __CLANG_FUJITSU)
#if 0
#include <mpi-ext.h>
// from /home/system/sample/PowerAPI/c/pwrset.c
#include "pwr.h"
static int down_clock(){
	PWR_Cntxt cntxt = NULL;
	PWR_Obj obj = NULL;
	int rc;
	double freq = 0.0;

	// Power APIのコンテキスト取得
	rc = PWR_CntxtInit(PWR_CNTXT_DEFAULT, PWR_ROLE_APP, "app", &cntxt);
	if (rc != PWR_RET_SUCCESS) {
		printf("CntxtInit Failed\n");
		return 1;
	}

	// 周波数設定の対象となるObjectの取得
	rc = PWR_CntxtGetObjByName(cntxt, "plat.node.cpu", &obj);
	if (rc != PWR_RET_SUCCESS) {
#if 0 // 今回は結果は無視する
		printf("CntxtGetObjByName Failed\n");
		return 1;
#endif
	}

	// 設定する周波数を指定
	freq = 2000000000.0;
	// Objectに周波数を設定
	rc = PWR_ObjAttrSetValue(obj, PWR_ATTR_FREQ, &freq);
	if (rc != PWR_RET_SUCCESS) {
#if 0 // 今回は結果は無視する
		printf("ObjAttrSetValue Failed (rc = %d)\n", rc);
		return 1;
#endif
	}

	// コンテキスト破棄
	PWR_CntxtDestroy(cntxt);

	return 0;
}
#else
static int down_clock(){ return 0; }
#endif

extern "C" void fjtrad_omp_barrer(){
#pragma omp barrier
}

