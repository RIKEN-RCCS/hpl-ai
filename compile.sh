
# in both trad-mode and clang-mode
FCCpx -Wall -Wextra -Kfast,ocl,openmp dstrsv.cpp -c
FCCpx -Wall -Wextra -Kfast,openmp sgetrf_nopiv.cpp -c
FCCpx -Wall -Wextra -Kfast,openmp -c kernels/generic.cpp #-Koptmsg=2
#FCCpx -Wall -Wextra -Nclang -Kfast,ocl -fopenmp dstrsv.cpp -c -o dstrsv.clang.o
#FCCpx -Wall -Wextra -Nclang -Kfast -fopenmp sgetrf_nopiv.cpp -c -o sgetrf_nopiv.clang.o 
#FCCpx -Wall -Wextra -Nclang -Kfast -Wall -fopenmp -c kernels/generic.cpp -o generic.clang.o #-Koptmsg=2

# in clang-mode without openmp
FCCpx -Wall -Wextra -Nclang -Kfast -c lazy_init_omp.cpp #-Koptmsg=2
FCCpx -Wall -Wextra -Nclang -Kfast,ocl -c otf_gemv.cpp
FCCpx -Wall -Wextra -Nclang -Kfast -c higham_mat_impl.cpp
FCCpx -Wall -Wextra -Nclang -Kfast,ocl -fno-associative-math -c back_buffer.cpp #-Koptmsg=2
FCCpx -Wall -Wextra -Nclang -Kfast -c kernels/a64fx.cpp #-Koptmsg=2

# uses mpi
mpiFCCpx -Wall -Wextra -Kfast remap.cpp -c
mpiFCCpx -Wall -Wextra -Kfast,ocl,openmp -Nfjomplib -SSL2BLAMP -ltofucom -I/opt/FJSVtcs/pwrm/aarch64/include -L/opt/FJSVtcs/pwrm/aarch64/lib64 -lpwr \
	main.cpp lazy_init_omp.o otf_gemv.o sgetrf_nopiv.o higham_mat_impl.o back_buffer.o dstrsv.o a64fx.o generic.o remap.o -o hpl-mpi.trad 
#mpiFCCpx -Wall -Kfast -fopenmp -SSL2BLAMP main.cpp lazy_init_omp.o otf_gemv.o sgetrf_nopiv.o dstrsv.o generic.clang.o a64fx.o -o hpl-mpi.clang remap.o -Nclang,fjomplib -ltofucom

