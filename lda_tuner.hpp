#include "timer.hpp"
#include "schur_updator.hpp"
#include <cstdint>
#include <cstdio>
struct NullComm {
	bool detached() const { return true; }
	bool progress() const { return true; }
};
template<typename FHigh, typename FLow, bool du, bool dd, bool pack>
void tune_lda(Panels<FHigh>& p, LRPanels<FLow>* lr, bool verbose)
{
	if(p.is_tile || lr[0].is_tile) return; // no tune for tile
	GemmControl<FHigh, FLow, dd, pack> mmcon(p.b, p.nprow, p.npcol);
	size_t const clsize = 256 / sizeof(FHigh);
	size_t const maxlda = p.lda;
	size_t const minlda = (p.b * p.nprow + clsize-1)/clsize*clsize;
	size_t const flop = (size_t)p.b*p.nprow*p.b*p.npcol*p.b * 2;
	NullComm nullc;
	uint64_t besttime = UINT64_MAX;
	uint64_t bestlda = minlda;
	for(size_t lda = minlda; lda<maxlda; lda+=clsize) {
		if((lda*sizeof(FHigh)%4096) == 0) continue;
		mmcon.set(true, 0, 0, p, lr[0], lr[1]);
		// empty run
		mmcon.update_rest(nullc, nullc);
		uint64_t begin = get_utime();
		mmcon.update_rest(nullc, nullc);
		mmcon.update_rest(nullc, nullc);
		mmcon.update_rest(nullc, nullc);
		uint64_t time = get_utime() - begin;
		if(verbose){
			double sec = tick2second(time)/ 3;
			double flops = flop / sec * 1e-12;
			std::printf("lda_tuner:: %6ld %.5f sec %.5f Tflop/s\n", lda, sec, flops);

		}
		if(time < besttime) {
			besttime = time;
			bestlda = lda;
		}
	}
	if(verbose) std::printf("lda_tuner:: selected lda = %6ld\n", bestlda);
	p.lda = bestlda;
	p.ldpp = bestlda * p.b;
}
