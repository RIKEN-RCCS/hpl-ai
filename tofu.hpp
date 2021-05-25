#ifndef TOFU_HPP
#define TOFU_HPP
#include <cstdlib>
#include <cassert>
#include "timer.hpp"

enum class TofudMapping {
	// Tofu TNI mapping
	// MAPXxY_ZRING
	// XxY are placement of numa processes
	// ZRing is the broadcasting algorithm (but currently, Z==2 is not supported)
	MAP1x1_1RING,
	MAP2x1_1RING,
	MAP1x2_1RING,
	MAP2x2_1RING,
	MAP4x1_1RING,
	MAP1x4_1RING,
	MAP1x1_2RING,
	MAP2x1_2RING,
	MAP1x2_2RING,
	MAP2x2_2RING,
	MAP4x1_2RING,
	MAP1x4_2RING,
	DEFAULT,
};

// #if defined(__FUJITSU) || defined(__CLANG_FUJITSU)
#ifdef USE_TOFU
#include <mpi.h>
#include <utofu.h>
void tofu_tni_mapping(int rowcol, int numaid, int nnuma, TofudMapping map, int& tni1, int& tni2)
{
	// rowcol==0: this is for row communicator, rowcol==1 this is for col communicator
	// numaid, nnuma: numa process mapping of this process
	// map is described above
	// tni1, tni2: which tnis to use
	// sanity checks
	assert(numaid >= 0 && numaid < nnuma);
	assert(0 <= rowcol && rowcol < 2);
	// defaults
	//fprintf(stderr, "tni map :: rowcol = %d numaid = %d nnuma = %d\n", rowcol, numaid, nnuma); fflush(stderr);
	if(map==TofudMapping::DEFAULT){
		if(nnuma==1) map = TofudMapping::MAP1x1_1RING;
		else if(nnuma==2) map = TofudMapping::MAP2x1_1RING;
		else if(nnuma==4) map = TofudMapping::MAP2x2_1RING;
		else std::abort();
	}
	tni2 = 0; // not considred yet
	switch(map){
	case TofudMapping::MAP1x1_1RING:
		assert(nnuma==1);
		tni1 = (rowcol==0 ? 0: 1);
	break;
	case TofudMapping::MAP2x1_1RING: // fall through
	case TofudMapping::MAP1x2_1RING:
		assert(nnuma==2);
		// use all four tnis. I don't know how useful this is.
		tni1 = (rowcol==0 ? 0: 1) + numaid * 2;
	break;
	case TofudMapping::MAP2x2_1RING: // fall through
	case TofudMapping::MAP1x4_1RING:
		assert(nnuma==4);
		// better row.
		if(rowcol==0) tni1 = numaid;
		else tni1 = 4 + numaid/2;
	break;
	case TofudMapping::MAP4x1_1RING:
		assert(nnuma==4);
		// better col. intra node is little faster than inter.
		if(rowcol==1) tni1 = numaid;
		else tni1 = 4 + numaid/2;
	break;
	default:
		std::abort();
	}
}

struct TofudComm {
	static constexpr int addr_max = 20; // maximum number of addresses to register
	static constexpr size_t minchunksize = 1920; // chunk sizes
	static constexpr size_t maxchunksize = 16*1024*1024-1;
	static constexpr int nchunks_base = 100;
	MPI_Comm comm;
	int id, np;
	int sourceid(int id, int np, int root) const {
		if(id==root) return -1;
		else if(id==0) return np-1;
		else return id-1;
	}
	int sinkid(int id, int np, int root) const{
		int tid = id<root ? id+np-root: id-root;
		if(tid==np-1) return -1;
		else if(id==np-1) return 0;
		else return id+1;
	}
	struct comminfo {
		// tofu infos of source and sink nodes
		utofu_vcq_id_t freeport, sessport; // remote cqs
		utofu_stadd_t rptrs[addr_max]; // remote addresses
		utofu_stadd_t rptr_step, myptr_step;
		int id;
		volatile int64_t step __attribute__((aligned(256))); // used for synchronizatino
		// volatile is required in the manual
	} source_info, sink_info;
	utofu_vcq_hdl_t freevcq, sessvcq; // local cqs
	utofu_stadd_t myptrs[addr_max];
	size_t maxsizes[addr_max/2];
	int nhandles;
	static constexpr int ndata_max = 4; // number of on-going communication is limited to 4
	struct datainfo {
		// descriptor for communication
		int handle;
		size_t offset;
		size_t size;
		size_t chunk_size;
		int nchunks;
		int ndetached;
		int root;
		int step;
	} data[ndata_max];
	int icompleted, idetached, ndata;
	// the constructor is a global operation and must be called from all the node in base_comm synchronously.
	TofudComm(MPI_Comm base_comm, int tni1=-1, int tni2=-1): comm(base_comm)
	{
		MPI_Comm_rank(comm, &id);
		MPI_Comm_size(comm, &np);
		nhandles = -1;
		source_info.step = INT64_MIN;
		source_info.id = id==0?np-1: id-1;
		sink_info.step = INT64_MIN;
		sink_info.id = id==np-1?0: id+1;
		clear();
		if(tni1 != -1) config(tni1, tni2);
	}

	void config(int tni1, int ){
		assert(nhandles == -1);
		//fprintf(stderr, "tni config :: rank = %d, tni = %d\n", id, tni1); fflush(stderr);
		nhandles = 0;
		size_t num_tnis;
		utofu_tni_id_t *tni_ids;
		int rc = utofu_get_onesided_tnis(&tni_ids, &num_tnis);
		if(rc != UTOFU_SUCCESS || num_tnis == 0) MPI_Abort(comm, 500);
		//fprintf(stderr, "get tni :: rank = %d tni = %d num_tnis = %d\n", id, tni1, num_tnis); fflush(stderr);
		utofu_tni_id_t tni_id = tni_ids[tni1%num_tnis];
		//free(tni_ids); something wrong?

		
		// may easily fault because the # of session cq is small
		check_tofu_err(utofu_create_vcq(tni_id, 0, &freevcq));
		check_tofu_err(utofu_create_vcq(tni_id, UTOFU_VCQ_FLAG_SESSION_MODE, &sessvcq));

		// sync step
		utofu_reg_mem(freevcq, (void*)&source_info.step, sizeof(int64_t), 0, &source_info.myptr_step);
		utofu_reg_mem(freevcq, (void*)&sink_info.step, sizeof(int64_t), 0, &sink_info.myptr_step);

		utofu_vcq_id_t freevcq_id, sessvcq_id;
		utofu_query_vcq_id(freevcq, &freevcq_id);
		utofu_query_vcq_id(sessvcq, &sessvcq_id);
		
		MPI_Barrier(comm);
		// exchange handles
		// send forward
		MPI_Sendrecv(&freevcq_id, 1, MPI_UINT64_T, source_info.id, 0,
			&sink_info.freeport, 1, MPI_UINT64_T, sink_info.id, 0, comm, MPI_STATUS_IGNORE);
		MPI_Sendrecv(&sessvcq_id, 1, MPI_UINT64_T, source_info.id, 0,
			&sink_info.sessport, 1, MPI_UINT64_T, sink_info.id, 0, comm, MPI_STATUS_IGNORE);
		MPI_Sendrecv(&source_info.myptr_step, 1, MPI_UINT64_T, source_info.id, 0,
			&sink_info.rptr_step, 1, MPI_UINT64_T, sink_info.id, 0, comm, MPI_STATUS_IGNORE);
		// send backward
		MPI_Sendrecv(&freevcq_id, 1, MPI_UINT64_T, sink_info.id, 0,
			&source_info.freeport, 1, MPI_UINT64_T, source_info.id, 0, comm, MPI_STATUS_IGNORE);
		MPI_Sendrecv(&sessvcq_id, 1, MPI_UINT64_T, sink_info.id, 0,
			&source_info.sessport, 1, MPI_UINT64_T, source_info.id, 0, comm, MPI_STATUS_IGNORE);
		MPI_Sendrecv(&sink_info.myptr_step, 1, MPI_UINT64_T, sink_info.id, 0,
			&source_info.rptr_step, 1, MPI_UINT64_T, source_info.id, 0, comm, MPI_STATUS_IGNORE);
		//utofu_set_vcq_id_path(&sink_info.freeport, NULL);
		//utofu_set_vcq_id_path(&sink_info.sessport, NULL);
		//utofu_set_vcq_id_path(&source_info.freeport, NULL);
		//utofu_set_vcq_id_path(&source_info.sessport, NULL);
		MPI_Barrier(comm);
	}
	void destroy() {
		if(nhandles!= -1){
			MPI_Barrier(comm);
			utofu_dereg_mem(freevcq, source_info.myptr_step, 0);
			utofu_dereg_mem(freevcq, sink_info.myptr_step, 0);
			for(int i=0; i<nhandles; i+=2) {
				utofu_dereg_mem(freevcq, myptrs[i], 0);
				utofu_dereg_mem(sessvcq, myptrs[i+1], 0);
			}
			utofu_free_vcq(freevcq);
			utofu_free_vcq(sessvcq);
			MPI_Barrier(comm);
			nhandles = -1;
		}
	}
	// NOTE. THE DESTRUCTOR MUST BE CALLED INSIDE MPI-REGION.
	// the destructor is a global operation and ust be called fro all the node in comm synchronously
	~TofudComm() {
		destroy();
	}
	// get_handle is a global operation and must be called from all the node in comm synchronously.
	int get_handle(char* ptr, size_t maxsize) {
		assert(nhandles != -1);
		memset(ptr, 1, maxsize);
		maxsizes[nhandles/2] = maxsize;
		MPI_Barrier(comm);
		assert(nhandles+2 <= addr_max);
		utofu_stadd_t fp, sp;
		utofu_reg_mem(freevcq, (void*)ptr, maxsize, 0, &fp);
		utofu_reg_mem(sessvcq, (void*)ptr, maxsize, 0, &sp);
		// send forward
		MPI_Sendrecv(&fp, 1, MPI_UINT64_T, source_info.id, 0,
			&sink_info.rptrs[nhandles], 1, MPI_UINT64_T, sink_info.id, 0, comm, MPI_STATUS_IGNORE);
		MPI_Sendrecv(&sp, 1, MPI_UINT64_T, source_info.id, 0,
			&sink_info.rptrs[nhandles+1], 1, MPI_UINT64_T, sink_info.id, 0, comm, MPI_STATUS_IGNORE);
		// send backward
		MPI_Sendrecv(&fp, 1, MPI_UINT64_T, sink_info.id, 0,
			&source_info.rptrs[nhandles], 1, MPI_UINT64_T, source_info.id, 0, comm, MPI_STATUS_IGNORE);
		MPI_Sendrecv(&sp, 1, MPI_UINT64_T, sink_info.id, 0,
			&source_info.rptrs[nhandles+1], 1, MPI_UINT64_T, source_info.id, 0, comm, MPI_STATUS_IGNORE);
		myptrs[nhandles] = fp;
		myptrs[nhandles+1] = sp;
		nhandles += 2;
		MPI_Barrier(comm);
		return nhandles/2 - 1;
	}
	// schedule a new communication
	// step: wait until neighbors to send send_sync(step)
	void schedule(int step, int handle, size_t offset, size_t size, int root) {
		if(np==1 || size==0) return;
		assert(size < nchunks_base * maxchunksize);
		assert(ndata < ndata_max);
		assert(offset + size <= maxsizes[handle]);
		data[ndata].handle = handle;
		data[ndata].offset = offset;
		data[ndata].size = size;
		if(size < nchunks_base * minchunksize){
			data[ndata].chunk_size = minchunksize;
			data[ndata].nchunks = (size+minchunksize-1)/minchunksize;
		}
		else {
			data[ndata].chunk_size = (size+nchunks_base-1)/nchunks_base;
			data[ndata].nchunks = (size+data[ndata].chunk_size-1)/data[ndata].chunk_size;
		}
		if(sinkid(id,np,root)!=-1) data[ndata].ndetached = 0;
		else data[ndata].ndetached = data[ndata].nchunks;
		data[ndata].root = root;
		data[ndata].step = step;
		++ndata;
	}
	// invoke = schedule() + progress()
	void invoke(int step, int handle, size_t offset, size_t size, int root) {
		schedule(step, handle, offset, size, root);
		progress();
	}

	// notify neighbors my step.
	void send_sync(int step) {
		send_sync_impl(step);
	}

	//  check if sink process steps
	bool check_sync(int step) {
		int64_t cstep = step;
		int64_t sstep = sink_info.step; // volatile
		return sstep >= cstep;
	}

	// progress communication. 
	bool progress(bool timer=true) {
		if(timer) Timer::beg(Timer::TEST);
		if(!detached() && check_sync(data[idetached].step)){
			if(send_impl(data[idetached])) ++ idetached;
		}
		if(timer) Timer::end(Timer::TEST);
		return detached();
	}

	// Whether all the communications are scheduled or not.
	// While !detached(), you need to call progress() to progress the communication.
	// detached() only means the communication is offloaded to th hardware.
	// If you wan to do something behind the communication, you should test() and compute.
	bool detached() const { return idetached == ndata; }

	// wait for completion. you are free to use the buffer after wait().
	void wait(bool timer=true) {
		if(timer) Timer::beg(Timer::WAIT);
		//printf("rank = %d wait\n", id); fflush(stdout);
		if(icompleted < ndata ){
			if(!detached()) while(!progress(false));
			while(icompleted < ndata) {
				wait_impl(data[icompleted]);
				++icompleted;
			}
		}
		clear();
		if(timer) Timer::end(Timer::WAIT);
	}
	bool test(bool timer=true) {
		bool ret = true;;
		if(timer) Timer::beg(Timer::TEST);
		if(!detached() && !progress()) ret = false;
		else {
			while(icompleted < ndata){
				if(!try_wait_impl(data[icompleted])){
					ret = false;
					break;
				}
				++icompleted;
			}
		}
		if(timer) Timer::end(Timer::TEST);
		return ret;
	}

	void clear() {
		icompleted = idetached = ndata = 0;
	}
	
	void check_tofu_err(int err){
		if(err==UTOFU_ERR_FULL) { MPI_Abort(comm, 501); }
		if(err==UTOFU_ERR_NOT_AVAILABLE) { MPI_Abort(comm, 502); }
		if(err==UTOFU_ERR_NOT_SUPPORTED) { MPI_Abort(comm, 503); }
	}
	void check_tofu_err2(int err){
		#if 0
		if(rc==UTOFU_ERR_TCQ_DESC) {printf("rank=%d tcq error desc %d %d\n", id, rc, (int)cbdata); fflush(stdout); }
		if(rc==UTOFU_ERR_TCQ_MEMORY) {printf("rank=%d tcq error memory %d %d\n", id, rc, (int)cbdata); fflush(stdout); }
		if(rc==UTOFU_ERR_TCQ_STADD) {printf("rank=%d tcq error stadd %d %d\n", id, rc, (int)cbdata); fflush(stdout); }
		if(rc==UTOFU_ERR_TCQ_LENGTH) {printf("rank=%d tcq error length %d %d\n", id, rc, (int)cbdata); fflush(stdout); }
		#else
		(void)err; // do nothing
		#endif
	}

	void poll_tcq_impl() {
		void* cbdata;
		int rc = utofu_poll_tcq(freevcq, 0, &cbdata);
		assert(rc==UTOFU_SUCCESS || rc==UTOFU_ERR_BUSY);
		if(rc == UTOFU_SUCCESS) ++icompleted; // !!! 
	}
	void send_sync_impl(int step) {
		//printf("rank = %d send_sync %d\n", id, step); fflush(stdout);
		unsigned long int flags = UTOFU_ONESIDED_FLAG_STRONG_ORDER;
		while(true){
			int rc = utofu_put_piggyback8(freevcq, source_info.freeport, (uint64_t)step, source_info.rptr_step, 8, 0, flags, (void*)-1);
			assert(rc==UTOFU_SUCCESS || rc==UTOFU_ERR_BUSY);
			if(rc==UTOFU_SUCCESS) break;
			poll_tcq_impl();
		}
	}
	bool send_impl(datainfo& data) {
		if(sinkid(id,np,data.root)==-1) return true;
		if(data.ndetached == data.nchunks) return true;
		//printf("rank = %d send\n", id); fflush(stdout);
		bool nosps = (np==2 || sinkid(sink_info.id,np,data.root)==-1);
		utofu_vcq_hdl_t myport = (id==data.root ? freevcq: sessvcq);
		utofu_stadd_t myaddr = (id==data.root ? myptrs[data.handle*2]: myptrs[data.handle*2+1]);
		int nb = data.nchunks;
		size_t offset = data.offset;
		size_t chunk_size = data.chunk_size;
		size_t length = data.size;
		utofu_stadd_t rptr = sink_info.rptrs[data.handle*2+1];

		if(nb>1){
			unsigned long int flags = UTOFU_ONESIDED_FLAG_STRONG_ORDER;
			flags |= nosps? 0: UTOFU_ONESIDED_FLAG_SPS(1);
			while(true) {
				int rc = utofu_put_stride(myport, sink_info.sessport, myaddr+offset, rptr+offset, chunk_size, chunk_size, nb-1, 0, flags, (void*)data.handle);
				assert(rc==UTOFU_SUCCESS || rc==UTOFU_ERR_BUSY);
				if(rc==UTOFU_SUCCESS) break;
				poll_tcq_impl();
			}
		}
		{
			unsigned long int flags = UTOFU_ONESIDED_FLAG_STRONG_ORDER;
			flags |= UTOFU_ONESIDED_FLAG_TCQ_NOTICE;
			flags |= nosps? UTOFU_ONESIDED_FLAG_REMOTE_MRQ_NOTICE: UTOFU_ONESIDED_FLAG_SPS(1);
			size_t start = chunk_size * (nb-1);
			assert(start < length);
			size_t clength = length - start; //start + chunk_size > length ? length - start: chunk_size;
			while(true) {
				int rc = utofu_put(myport, sink_info.sessport, myaddr+offset+start, rptr+offset+start, clength, 0, flags, (void*)data.handle);
				assert(rc==UTOFU_SUCCESS || rc==UTOFU_ERR_BUSY);
				if(rc==UTOFU_SUCCESS) break;
				poll_tcq_impl();
			}
		}
		data.ndetached = data.nchunks;
		return true;
	}

	void wait_impl(datainfo& data) {
		if(id==data.root || sinkid(id,np,data.root)!=-1){
			//printf("rank = %d waitm\n", id); fflush(stdout);
			utofu_vcq_hdl_t myport = (id==data.root ? freevcq: sessvcq);
			void *cbdata;
			int rc;
			do {
				rc = utofu_poll_tcq(myport, 0, &cbdata);
			} while(rc==UTOFU_ERR_NOT_FOUND);
			check_tofu_err2(rc);
			assert(rc==UTOFU_SUCCESS);
		}
		else if(data.nchunks) {
			//printf("rank = %d waitc\n", id); fflush(stdout);
			int rc;
			struct utofu_mrq_notice notice;
			do {
				rc = utofu_poll_mrq(sessvcq, 0, &notice);
			} while(rc==UTOFU_ERR_NOT_FOUND);
			assert(rc==UTOFU_SUCCESS);
			data.ndetached = data.nchunks = 0;
		}

	}
	bool try_wait_impl(datainfo& data) {
		if(id==data.root || sinkid(id,np,data.root)!=-1){
			utofu_vcq_hdl_t myport = (id==data.root ? freevcq: sessvcq);
			void *cbdata;
			int rc = utofu_poll_tcq(myport, 0, &cbdata);
			check_tofu_err2(rc);
			assert(rc==UTOFU_SUCCESS || rc==UTOFU_ERR_NOT_FOUND);
			return rc != UTOFU_ERR_NOT_FOUND;
		}
		else if(data.nchunks) {
			struct utofu_mrq_notice notice;
			int rc = utofu_poll_mrq(sessvcq, 0, &notice);
			assert(rc==UTOFU_SUCCESS || rc==UTOFU_ERR_NOT_FOUND);
			if(rc==UTOFU_ERR_NOT_FOUND) return false;
			data.ndetached = data.nchunks = 0;
		}
		return true;
	}
};
#else
// backup 
#include "chain_schedule.hpp"
void tofu_tni_mapping(int rowcol, int numaid, int nnuma, TofudMapping, int& tni1, int& tni2 )
{
	assert(numaid >= 0 && numaid < nnuma);
	assert(0 <= rowcol && rowcol < 2);
	tni1 = tni2 = 0;
}
struct TofudComm {
	// use ChainSchedule for machiens without Tofu
	static constexpr int adr_max = 10;
	ChainSchedule base;
	char* ptrs[adr_max];
	int nhandles;
	TofudComm(MPI_Comm base_comm, int /*tni1*/=-1, int /*tni2*/=-1): base(base_comm), nhandles(0) {}

	void config(int, int ){ }
	~TofudComm() {}

	int get_handle(char* ptr, size_t ) {
		ptrs[nhandles] = ptr;
		return nhandles++;
	}

	void schedule(int , int handle, size_t offset, size_t size, int root=-1) {
		base.schedule(ptrs[handle]+offset, size, root);
	}
	void invoke(int step, int handle, size_t offset, size_t size, int root) {
		schedule(step, handle, offset, size, root);
	}
	void send_sync(int ) {}
	bool check_sync(int ) { return true; }
	bool progress(bool timer=true) { return base.progress(timer); }
	bool detached() const { return base.done(); }
	bool test(bool timer=true) { return base.progress(timer); }
	void wait(bool /*timer*/=true) { base.force_complete(); }
	void clear() { base.clear(); }
};

#endif
void wait_all(TofudComm& lcom, TofudComm& rcom){
	Timer::beg(Timer::WAIT);
	if(!lcom.detached() && !rcom.detached()) while(true){
		if(lcom.progress(false)) break;
		if(rcom.progress(false)) break;
	}
	lcom.wait(false);
	rcom.wait(false);
	Timer::end(Timer::WAIT);
}


#endif


#ifdef UNIT_TEST
#include <time.h>
static int64_t get_utime(){
	timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);

	return ts.tv_nsec + ts.tv_sec*1000000000ll;
}
int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	int numprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	if(numprocs<2) MPI_Abort(MPI_COMM_WORLD, 1);

	int myrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	size_t maxlength = 1024*1024*100;
	char* buffer0 = (char*)malloc(maxlength);
	char* buffer1 = (char*)malloc(maxlength);
	memset(buffer0, 0, maxlength);
	memset(buffer1, 0, maxlength);

	{
		TofudComm tcom(MPI_COMM_WORLD, myrank);
		int handle = tcom.get_handle(buffer0, maxlength);
		int* ptr = (int*)buffer0;
		for(int step=0; step<100; ++step){
			int root = step % numprocs;
			tcom.send_sync(step);
			if(myrank==root){
				for(int i=0; i<step+1; ++i) ptr[i] = step;
			}
			tcom.invoke(step, handle, (step+1)*sizeof(int), root);
			tcom.wait();
			for(int i=0; i<step+1; ++i) {
				printf("rank=%d, %d == %d\n", myrank, ptr[i], step ); fflush(stdout);
			}
		}
	}

	{
		TofudComm tcom(MPI_COMM_WORLD, myrank);
		int handle0 = tcom.get_handle(buffer0, maxlength);
		int handle1 = tcom.get_handle(buffer0, maxlength);
		for(int step=0; step<10; ++step){
			tcom.send_sync(step);
			tcom.invoke(step, handle0, maxlength-step*1024*1024, step%numprocs);

			uint64_t time0 = get_utime();
			int* t1 = (int*)buffer1;
			int x = 0;
			for(int i=0; i<(maxlength-(step+1)*1024*1024)/sizeof(int); ++i) x += t1[i];
			for(int i=0; i<(maxlength-(step+1)*1024*1024)/sizeof(int); ++i) t1[i] = step;

			uint64_t time1 = get_utime();
			tcom.wait();
			uint64_t time2 = get_utime();

			printf("rank %d, get %d, %.9e %.9e\n", myrank, x, (time1-time0)*1e-9, (time2-time1)*1e-9); fflush(stdout);

			int h = handle0; handle0 = handle1; handle1 = h;
			char* p = buffer0; buffer0 = buffer1; buffer1 = p;
		}
		// DESTRUCT TofudComm HERE
	}
	free(buffer0);
	free(buffer1);

	
	MPI_Finalize();
}
#endif

