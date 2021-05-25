#pragma once
#include <cstdio>
#include <cstdint>

// Default take statistics
// #define TIMER_VERBOSE // save all the timing for visualization
// #define TIMER_SILENT // Disable all timer

#ifdef TIMER_VERBOSE
#include <vector>
#  define TIMER_NUM 2
#elif defined(TIMER_SILENT)
#  define TIMER_NUM 0
#else
#  define TIMER_NUM 1
#endif

extern "C" int MPI_Get_processor_name(char *, int *);

#ifdef __aarch64__
static int64_t get_utime(){
	uint64_t tsc;
	asm volatile ("mrs %0, cntvct_el0" : "=r" (tsc));
	return tsc;
}
static double tick2second(uint64_t tick){
	auto frequency = []{
		uint64_t frq;
		asm volatile ("mrs %0, cntfrq_el0" : "=r" (frq));
		return frq;
	};
	static double invfreq = 1.0 / frequency();
	return invfreq * (double)tick;
}
#else
#  ifdef __APPLE__
#  include <sys/time.h>
static int64_t get_utime(){
	timeval tv;
	gettimeofday(&tv, NULL);

	return tv.tv_usec*1000ll + tv.tv_sec*1000000000ll;
}
#  elif defined __linux__
#  include <time.h>
static int64_t get_utime(){
	timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);

	return ts.tv_nsec + ts.tv_sec*1000000000ll;
}
#  endif
static double tick2second(uint64_t tick){
	return 1.e-9 * (double)tick;
}
#endif

struct Timer_template_base {
	enum Items{
		DIAG_BCAST = 0,
		LCOL_BCAST,
		RROW_BCAST, 
		TEST,
		WAIT,
		DIAG_LU,
		TRSM_L,
		TRSM_R,
		CONV_L,
		CONV_R,
		GEMM_UPDATE,
		GEMM_PROGRESS,
		LAZY_INIT,
		WRITE_BACK,
		IR_GEMV,
		IR_GEMV_COMM,
		IR_TRSV,
		IR_TRSV_MV,
		IR_TRSV_COMM,
		// don't touch below
		TOTAL,
		MISC,
		NUM_ITEMS,
	};
};
template <int>
struct Timer_template: Timer_template_base {
	// caleld fromo initialize()
	static void flush(){} 
	// initialize timer with 0
	static void initialize(){}
	// start a region
	//   elem: which item to collect
	//   reuse: reuse last timing for reduce timer cost. 
	static void beg(const Items /*elem*/, bool const /*reuse*/ = false){}
	// stop a region
	//   elem: which item to collect
	//   reuse: reuse last timing for reduce timer cost. 
	//   acc: number of operations in this region. ex. flops, mips, or, bytes
	static void end(const Items /*elem*/, bool const /*reuse*/ = false, int64_t /*acc*/ = 0ll){}
	// save a timing
	static double put(const Items /*elem*/, bool const /*reuse*/ = false){ return 0.0; }
	// dump out all the data
	static void show(FILE* /*fp*/ = stderr, char const */*fmt*/ = ""){}
	// open file and call show()
	//    size: number of procs
	//    rank: process id
	//    row : row id
	//    col : col id
	//    filename : filename
	static void dump_mp( const int /*size*/, const int /*rank*/,
		const int /*row*/, const int /*col*/, const char* /*filename*/ = ""){} // do nothing
};

template<>
struct Timer_template<1> : Timer_template_base{
	static char const *name(int const i){
		static const char *strs[NUM_ITEMS] = {
			"DIAG_BCAST",
			"LCOL_BCAST",
			"RROW_BCAST", 
			"TEST",
			"WAIT",
			"DIAG_LU",
			"TRSM_L",
			"TRSM_R",
			"CONV_L",
			"CONV_R",
			"GEMM_UPDATE",
			"GEMM_PROGRESS",
			"LAZY_INIT",
			"WRITE_BACK",
			"IR_GEMV",
			"IR_GEMV_COMM",
			"IR_TRSV",
			"IR_TRSV_MV",
			"IR_TRSV_COMM",
			"TOTAL",
			"MISC",
		};
		return strs[i];
	}

	static void flush(){
		for(int i=0; i<NUM_ITEMS; i++){
			time (i) = 0ll;
			accum(i) = 0ll;
		}
	}

	static void initialize(){
		flush();
		tprev(1) = get_utime();
	}

	static void beg(const Items elem, bool const reuse = false){
		if(reuse) time(elem) -= tprev();
		else time(elem) -= (tprev() = get_utime());
	}	

	static void end(const Items elem, bool const reuse = false){
		if(reuse) time(elem) += tprev();
		else time(elem) += (tprev() = get_utime());
	}
	static void end(const Items elem, bool const reuse, int64_t acc){
		end(elem, reuse);
		accum(elem) += acc;
	}

	static double put(const Items /*elem*/, bool const reuse = false){
		uint64_t tt = reuse ? get_utime() : tprev();
		return tick2second(tt - tprev(1));
	}

	static void show(
			FILE *fp = stderr,
			const char *fmt = " %-12s : %e sec : %6.2f %% : %20lld : %e Gop/s\n")

	{
		fflush(fp);

		time(MISC) = time(TOTAL);

		for(int i=0; i<NUM_ITEMS-2; i++){
			time(MISC) -= time(i);
		}

		for(int i=0; i<NUM_ITEMS; i++){
			fprintf(fp, fmt, name(i),
					rtime(i),
					100.0 * time(i)/ time(TOTAL),
					accum(i),
					1e-9*accum(i) / rtime(i));
		}
		const double flops = (double)(accum(GEMM_UPDATE)+accum(GEMM_PROGRESS)) /
			(rtime(GEMM_UPDATE)+rtime(GEMM_PROGRESS));
		fprintf(fp, "GEMM_TOTAL: %f Tflops\n", 1.e-12 * flops);

		fflush(fp);
	}

	static void dump_mp(
			const int size,
			const int rank,
			const int row,
			const int col,
			const char *fmt = "Timerdump.%04d.%04d")
	{
		static char filename[1024];
		sprintf(filename, fmt, size, rank);
		FILE *fp = fopen(filename, "w");
		if(fp){
			int len;
			MPI_Get_processor_name(filename, &len);
			fprintf(fp, "# row=%d, col=%d, host=%s\n", row, col, filename);
			show(fp);
			fclose(fp);
		}
	}

	private:
	static int64_t &time(int const i){
		static int64_t buf[NUM_ITEMS];
		return buf[i];
	}
	static int64_t &accum(int const i){
		static int64_t buf[NUM_ITEMS];
		return buf[i];
	}
	static int64_t &tprev(int const ch=0){
		static int64_t t[2]; /* 0 : previous time */
		                     /* 1 : initial time  */
		return t[ch];
	}
	static double rtime(int const i){
		return tick2second(time(i));
	}
};

#ifdef TIMER_VERBOSE
// pritout all the timings with binary format
template<>
struct Timer_template<2> : Timer_template_base{
	static char const *name(int const i){
		static const char *strs[NUM_ITEMS] = {
			"DIAG_BCAST",
			"LCOL_BCAST",
			"RROW_BCAST", 
			"TEST",
			"WAIT",
			"DIAG_LU",
			"TRSM_L",
			"TRSM_R",
			"CONV_L",
			"CONV_R",
			"GEMM_UPDATE",
			"GEMM_PROGRESS",
			"LAZY_INIT",
			"WRITE_BACK",
			"IR_GEMV",
			"IR_GEMV_COMM",
			"IR_TRSV",
			"IR_TRSV_MV",
			"IR_TRSV_COMM",
			"TOTAL",
			"MISC",
		};
		return strs[i];
	}

	static void flush(){
		for(int i=0; i<NUM_ITEMS; i++){
			time(i) = 0ll;
			accum(i) = 0ll;
			tvec_beg(i).clear();
			tvec_end(i).clear();
			tvec_put(i).clear();
			avec(i).clear();
		}
	}

	static void initialize(){
		flush();
		for(int i=0; i<NUM_ITEMS; i++){
			tvec_beg(i).reserve(10000);
			tvec_end(i).reserve(10000);
			tvec_put(i).reserve(10000);
			avec(i).reserve(10000);
		}
		tprev(1) = get_utime();
	}

	static void beg(const Items elem, bool const reuse = false){
		//fprintf(stderr, "%s: DEBUG BEG %s\n", hostname(), name(elem)); fflush(stderr);
		if(reuse) time(elem) -= tprev();
		else time(elem) -= (tprev() = get_utime());

		tvec_beg(elem).push_back(tprev());
	}	

	static void end(const Items elem, bool const reuse = false, int64_t acc=0ll){
		//fprintf(stderr, "%s: DEBUG END %s\n", hostname(), name(elem)); fflush(stderr);
		if(reuse) time(elem) += tprev();
		else time(elem) += (tprev() = get_utime());
		accum(elem) += acc;

		tvec_end(elem).push_back(tprev());
		avec(elem).push_back(acc);
	}

	static double put(const Items elem, bool const reuse = false){
		uint64_t tt = reuse ? get_utime() : tprev();
		tvec_put(elem).push_back(tt);
		return tick2second(tt - tprev(1));
	}

	static void show(
			FILE *fp = stderr,
			const char *fmt = " %-12s : %e sec : %6.2f %% : %20ld : %e Gop/s\n",
			FILE *fp2 = nullptr)
	{
		fflush(fp);

		time(MISC) = time(TOTAL);

		for(int i=0; i<NUM_ITEMS-2; i++){
			time(MISC) -= time(i);
		}

		for(int i=0; i<NUM_ITEMS; i++){
			fprintf(fp, fmt, name(i),
					rtime(i),
					100.0 * time(i)/ time(TOTAL),
					accum(i),
					1e-9*accum(i)/rtime(i));
		}
		const double flops = (double)(accum(GEMM_UPDATE)+accum(GEMM_PROGRESS)) /
			(rtime(GEMM_UPDATE)+rtime(GEMM_PROGRESS));
		fprintf(fp, "GEMM_TOTAL: %f Tflops\n", 1.e-12 * flops);

		fflush(fp);

		// dump event vectors
		if(!fp2) fp2 = fp;
		for(int i=0; i<NUM_ITEMS; i++){
			dump_vector(fp2, tvec_beg(i), "BEG_", name(i));
			dump_vector(fp2, tvec_end(i), "END_", name(i));
			dump_vector(fp2, tvec_put(i), "PUT_", name(i));
			dump_accum( fp2, avec(i),     "ACC_", name(i));
		}
		fflush(fp);
	}

	static void dump_mp(
			const int /*size*/,
			const int rank,
			const int row,
			const int col,
			const char *filename)
	{
		fprintf(stderr, "%d: (%d, %d)\n", rank, row, col);
		FILE *fp = fopen(filename, "w");
		if(fp){
			int len;
			char hostname[1024];
			MPI_Get_processor_name(hostname, &len);
			fprintf(fp, "# row=%d, col=%d, host=%s\n", row, col, hostname);
			show(fp);
			fclose(fp);
		}
	}

	static const char *hostname(){
		static char name[1024] = {0,};
		if(!name[0]){
			int len;
			MPI_Get_processor_name(name, &len);
		}
		return name;
	}


	private:
	static int64_t &time(int const i){
		static int64_t buf[NUM_ITEMS];
		return buf[i];
	}
	static int64_t &accum(int const i){
		static int64_t buf[NUM_ITEMS];
		return buf[i];
	}

	using tvec = std::vector<int64_t>;
	static tvec &tvec_beg(int const i){
		static tvec buf[NUM_ITEMS];
		return buf[i];
	}
	static tvec &tvec_end(int const i){
		static tvec buf[NUM_ITEMS];
		return buf[i];
	}
	static tvec &tvec_put(int const i){
		static tvec buf[NUM_ITEMS];
		return buf[i];
	}
	static tvec &avec(int const i){
		static tvec buf[NUM_ITEMS];
		return buf[i];
	}

	static void dump_vector(FILE *fp,const tvec &v, const char *s0, const char *s1){
		const int n = v.size();
		if(fp!=stderr){
			fprintf(fp, "bio, %d, %s%s\n", n, s0, s1);
			for(int i=0; i<n; i++){
				unsigned long utime = v[i];
				double dtime = tick2second(utime - tprev(1));
				fwrite(&dtime, sizeof(double), 1, fp);
			}
		}
		else {
			for(int i=0; i<n; i++){
				unsigned long utime = v[i];
				double   dtime = tick2second(utime - tprev(1));

				fprintf(fp,"%ld, %16.12f, %s%s, %d\n", utime, dtime, s0, s1, i);
			}
		}
	}

	static void dump_accum(FILE *fp,const tvec &v, const char *s0, const char *s1){
		const int n = v.size();
		if(fp!=stderr){
			fprintf(fp, "bio, %d, %s%s\n", n, s0, s1);
			fwrite(v.data(), sizeof(int64_t), n, fp);
		}
		else {
			for(int i=0; i<n; i++){
				fprintf(fp,"%ld, 0.0, %s%s, %d\n", v[i], s0, s1, i);
			}
		}
	}

	static int64_t &tprev(int const ch=0){
		static int64_t t[2];
		return t[ch];
	}
	static double rtime(int const i){
		return tick2second(time(i));
	}
};
#endif


using Timer = Timer_template<TIMER_NUM>;
