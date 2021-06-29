MPICXX = mpicxx
CXXFP16 = mpicxx
DEFINES = -DEXTERNAL_CONV -DOTF_GEMV_OPTIMIZED
CFLAGS = -W -Wall -fopenmp #-lm
CFLAGS += -std=c++17 -O2 -Wextra -march=native $(DEFINES)
CXXFP16FLAGS = $(CFLAGS)
LDFLAGS = -lm
LDFLAGS += -L ../lib -lblas
OBJ = main.o lazy_init_omp.o otf_gemv.o avx2.o generic.o higham_mat_impl.o back_buffer.o

main.o: main.cpp fp16sim.hpp getrf_nopiv.hpp grid.hpp hpl_rand.hpp \
	iterative_refinement.hpp lazy_init.hpp matgen.hpp panel.hpp \
	panel_gemv.hpp panel_norm.hpp panel_trf.hpp panel_trsv.hpp \
	panel_check.hpp \
	schur_updator.hpp timer.hpp chain_schedule.hpp tofu.hpp kernels/kernel.h \
	highammgen.hpp  back_buffer.hpp
	$(MPICXX) -c -o $@ main.cpp $(CFLAGS)

lazy_init_omp.o : lazy_init_omp.cpp lazy_init.hpp panel.hpp hpl_rand.hpp \
	fp16sim.hpp timer.hpp fp16sim.hpp
	$(CXXFP16) -c -o $@ $< $(CXXFP16FLAGS)
otf_gemv.o: otf_gemv.cpp svesim.hpp hpl_rand.hpp
	$(CXXFP16) -c -o $@ $< $(CXXFP16FLAGS)
avx2.o: kernels/avx2.cpp fp16sim.hpp kernels/kernel.h
	$(CXXFP16) -c -o $@ $< $(CXXFP16FLAGS)
generic.o: kernels/generic.cpp fp16sim.hpp kernels/kernel.h
	$(MPICXX) -c -o $@ $< $(CFLAGS)
higham_mat_impl.o: higham_mat_impl.cpp
	$(CXXFP16) -c -o $@ $< $(CXXFP16FLAGS)
back_buffer.o: back_buffer.cpp back_buffer.hpp svesim.hpp fp16sim.hpp
	$(CXXFP16) -c -o $@ $< $(CXXFP16FLAGS)


driver.out: $(OBJ) 
	$(MPICXX) -o $@ $^ $(CFLAGS) $(LDFLAGS)

all: driver.out

clean:
	rm -f $(OBJ) driver.out
