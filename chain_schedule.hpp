#ifndef CHAIN_SCHEDULE_HPP
#define CHAIN_SCHEDULE_HPP
// broadcast data by MPI send-recv communication
#include <mpi.h>
#include <cstdlib> // size_t
#include <cstdio>
#include "timer.hpp"

struct ChainSchedule {
	// a manger class for MPI send-recv based broadcast.
	// this uses very naiive algorithm, but works fine if you set good parameters.
	static int constexpr smax = 8; // maximum number of on-going communications
	static size_t constexpr minchunk_size = 1024;// * 1024; // minimum size of chunks; data are separed in chunks.
	int id, np;
	MPI_Comm comm;
	int prev, next;
	struct ReqBuf {
		static int constexpr nmax = 256; // maximum number of on-going MPI_Requests.
		MPI_Request reqs[nmax];
		int icomplete, isend, irecv;
		bool done() const { return icomplete == irecv; }
		bool full() const { return irecv - icomplete == nmax; }
		MPI_Request* recvtop() { return &reqs[irecv%nmax]; }
		MPI_Request* sendtop() { return &reqs[isend%nmax]; }
		MPI_Request* compltop() { return &reqs[icomplete%nmax]; }
		void clear() { icomplete = isend = irecv = 0; }
	} reqs;
	struct Data{
		static int constexpr maxnchunks = 100;
		char* ptr;
		size_t size;
		size_t chunk_size;
		int root;
		int nsend;
		int isend, irecv, nchunks;
		void set(char* ptr_, size_t size_, int root_, int nsend_) {
			ptr = ptr_;
			size = size_;
			root = root_;
			nsend = nsend_;
			isend = irecv = 0;
			if(size_<minchunk_size*maxnchunks){
				chunk_size = minchunk_size;
				nchunks = (size_+minchunk_size-1)/minchunk_size;
			}
			else {
				chunk_size = (size+maxnchunks-1)/maxnchunks;
				nchunks = (size+chunk_size-1)/chunk_size;
			}
			nchunks *= nsend;
		}
		bool recv(char*& start, size_t& cursize) {
			if(chunk_size*irecv >= size)
				start = NULL;
			else {
				start = ptr + chunk_size * irecv;
				cursize = chunk_size*irecv + chunk_size < size ? chunk_size: size - chunk_size*irecv;
			}
			++irecv;
			return irecv == nchunks;
		}
		bool send(char*& start, size_t& cursize) {
			int t = isend/nsend;
			if(chunk_size*t >= size)
				start = NULL;
			else {
				start = ptr + chunk_size*t;
				cursize = chunk_size*t + chunk_size < size ? chunk_size: size - chunk_size*t;
			}
			++isend;
			return isend == nchunks;
		}
	} data[smax];
	int isend, irecv, iend;

	ChainSchedule(MPI_Comm comm): comm(comm){
		MPI_Comm_rank(comm, &id);
		MPI_Comm_size(comm, &np);
		clear();
		prev = id==0 ? np-1: id-1;
		next = id==np-1 ? 0: id+1;
	}
	int idfrom(int id, int np, int root){
		int end = np/2;
		int tid = (id < root ? id + np - root: id - root);
		return (tid <= end? prev: next);
	}
	bool recv(){
		//if(irecv >= iend || iend >= smax) fprintf(stderr, "recv oor %d %d %d\n", isend, irecv ,iend);
		char* ptr;
		size_t size;
		bool completed = data[irecv].recv(ptr, size);
		//fprintf(stderr, "recv %d %d %d :: %d %d\n", isend, irecv ,iend, data[irecv].size, data[irecv].root);
		if(!ptr || id == data[irecv].root)
			*reqs.recvtop() = MPI_REQUEST_NULL;
		else {
			int from = idfrom(id, np, data[irecv].root);
			//fprintf(stderr, "recvc %d %d %d :: %d %d %d %d\n", isend, irecv ,iend, data[irecv].size, data[irecv].root, from, np);
			MPI_Irecv(ptr, size, MPI_BYTE, from, 0 , comm, reqs.recvtop());
		}
		++reqs.irecv;
		if(completed) {
			++irecv;
			return true;
		}
		else
			return false;
	}
	int idto(int id, int np, int root){
		int end = np/2;
		int tid = (id<root ? id+np-root: id-root);
		if(tid < end) return next;
		else if(tid > end+1)  return prev;
		else return -1;
	}
	bool send(){
		//if(isend >= iend || iend >= smax) fprintf(stderr, "send oor %d %d %d\n", isend, irecv ,iend);
		char* ptr;
		size_t size;
		bool completed = data[isend].send(ptr, size);
		//fprintf(stderr, "send %d %d %d :: %d %d\n", isend, irecv ,iend, data[isend].size, data[isend].root);
		if(!ptr) *reqs.sendtop() = MPI_REQUEST_NULL;
		else if(id == data[isend].root){
			int to = (data[isend].nsend==1 ? next: (data[isend].isend%2 ? next: prev));
			//fprintf(stderr, "sendr %d %d %d :: %d %d %d %d\n", isend, irecv ,iend, data[isend].size, data[isend].root, to, np);
			MPI_Isend(ptr, size, MPI_BYTE, to, 0, comm, reqs.sendtop());
		}
		else{
			int to = idto(id, np, data[isend].root);
			//fprintf(stderr, "sendc %d %d %d :: %d %d %d %d\n", isend, irecv ,iend, data[isend].size, data[isend].root, to, np);
			//if(to!=-1) MPI_Send(ptr, size, MPI_BYTE, to, 20000 + to, comm);
			//*reqs.sendtop() = MPI_REQUEST_NULL;
			if(to!=-1) MPI_Isend(ptr, size, MPI_BYTE, to, 0, comm, reqs.sendtop());
			else *reqs.sendtop() = MPI_REQUEST_NULL;
		}
		++reqs.isend;
		if(completed){
			++isend;
			return true;
		}
		else
			return false;
	}
	void schedule(char* ptr, size_t size, int root=-1){
		// broadcast (or receive) data in resion ptr[0:size-1] from root
		// the number of on-going broadcasts are limited ~ 3. Call force_wait() if the communications are done.
		if(np<=1) return;
		if(root == -1) root = id;
		if(size) {
			data[iend].set(ptr, size, root, (np>2 && id==root) ? 2: 1);
			++iend;
		}
		progress();
	}
	bool done() const {
		// is all the communication is done
		return isend == iend && reqs.done();
	}
	bool detached() const { return done(); }
	bool progress(bool timer=true){
		// progress communication. repeatedly call this method until !done().
		if(done()) return true;
		//fprintf(stderr, "progs %d %d %d :: %d %d %d\n", isend, irecv ,iend, reqs.icomplete, reqs.isend, reqs.irecv);
		if(timer) Timer::beg(Timer::TEST);
		bool changed = false;
		do{
			changed = false;
			while(reqs.isend < reqs.irecv){
				int flag;
				MPI_Test(reqs.sendtop(), &flag, MPI_STATUS_IGNORE);
				if(flag) {
					changed = true;
					send();
				}
				else break;
			}
			while(reqs.icomplete < reqs.isend) {
				int flag;
				MPI_Test(reqs.compltop(), &flag, MPI_STATUS_IGNORE);
				if(flag){
					changed = true;
					++reqs.icomplete;
				}
				else break;
			}
			while(irecv < iend && !reqs.full()){ changed=true; recv(); }
		} while(changed);
		if(timer) Timer::end(Timer::TEST);
		//fprintf(stderr, "proge %d %d %d :: %d %d %d\n", isend, irecv ,iend, reqs.icomplete, reqs.isend, reqs.irecv);
		return done();
	}
	void force_complete(){
		// call progress() untill !done()
		// XXX this can cause deadlock. Use wait_all() below to avoid it.
		//fprintf(stderr, "forcs %d %d %d :: %d %d %d\n", isend, irecv ,iend, reqs.icomplete, reqs.isend, reqs.irecv);
		Timer::beg(Timer::WAIT);
		while(!done()){
			if(reqs.isend < reqs.irecv){
				MPI_Wait(reqs.sendtop(), MPI_STATUS_IGNORE);
				send();
			}
			if(reqs.icomplete < reqs.isend) {
				MPI_Wait(reqs.compltop(), MPI_STATUS_IGNORE);
				++reqs.icomplete;
			}
			while(irecv < iend && !reqs.full())  recv();
		}
		//fprintf(stderr, "force %d %d %d :: %d %d %d\n", isend, irecv ,iend, reqs.icomplete, reqs.isend, reqs.irecv);
		Timer::end(Timer::WAIT);
		clear();
	}
	void clear() {
		// clear contexts before next schedule.
		isend = irecv = iend = 0;
		reqs.clear();
	}
};

void wait_all(ChainSchedule& lhs, ChainSchedule& rhs)
{
	// force wait two CSs. calling them separately will result in deadlock.
	Timer::beg(Timer::WAIT);
	while(!lhs.done() && !rhs.done()){
		if(lhs.progress(false)) break;
		if(rhs.progress(false)) break;
	}
	Timer::end(Timer::WAIT);
	lhs.force_complete();
	rhs.force_complete();
}

#endif

