#include <cstdint>
#include <stdio.h>
#include "../svesim.hpp"
#include <arm_sve.h>

extern "C" void hgemmpp_kernel(int64_t ma, int64_t nb, int64_t kk, __fp16 * a, __fp16 * b, __fp16* c, int64_t ldc)
{
	// NOTE: THIS IS PUBLIC VERSION. THIS IS *NOT* THE CODE WE USED IN THE BENCHMARK.
	//       CONTACT WITH FUJITSU AND RIKEN TO GET REAL VERSION.
	// We provide this code to describe our optimization to the gemm kernels.
	// Peoples who are good at kernel optimizations can reimpelment by theirselves by reading the code and the instructions below.
	// Do not use this code without modifications, because this is slow without them.
	// Instructinos for optimizations:
	// 1. Get the DGEMM kernel from https://github.com/fujitsu/A64FX/blob/master/sample/dgemm.kernel.S
	// 2. Change calling convension from reference based (FORTRAN) to value based (C).
	//    This removes some meaningless loads and register movements.
	// 3. Change the matrix format. The origianl codes is for (A,B,C) = (Packed, Normal, Normal).
	//    Our PP kernel is for (A,B,C) = (Packed, Packed, Normal).  Look at this code for exact format.
	// 4. Change the order of summantion. The origianl code loads part of C to the registers first and add A*Bs to it.
	//    Our code prefetches the part of C first, initializes registers, and add A*Bs to it.
	//    Our's are sperior in performance and accuracy in the HPL-AI benchmark.
	// 5. Unroll the first and the last loop. 
	// 6. Reschedule the instructions. Take care about the 16 byte alignement of the branch target.
	//    VLIW like scheduling used in the original code may be good. (We do not know the dtails.)
	//    Load scheduling is VERY important. Look at the original code.

	// ma: row size
	// nb: column *block* size
	// kk: inner-product size. MUST BE THE MULTIPLE OF 2 AND GREATER THAN 3
	// ldc: must be cache-size aligned.

	// assert(ma>=1)
	// assert(nb>=1)
	// assert(kk>=4 && (kk%2)==0)
	// assert((ldc%128) == 0)

	int64_t bstride = 5 * kk;
	int64_t cstride = 5 * ldc;
	int64_t cpstart = 0;
	int64_t ldc2 = 2 * ldc;
	int64_t ldc3 = 3 * ldc;
	int64_t ldc4 = 4 * ldc;
	if(((size_t)c)&0xff) {
		// c is not aligned, placed on two cahche lines. We use sve-prefetch to prefetch both cahche lines.
		cpstart = (((size_t)c)&0xff - (256-32)) / 2;
	}
	kk -= 4;
	svbool_t ptrue = svptrue_b16();
	svfloat16_t z00, z10, z20, z30, z01, z11, z21, z31, z02, z12, z22, z32, z03, z13, z23, z33, z04, z14, z24, z34;
	for(int64_t jj=0; jj<nb; ++jj){
		__fp16* ca = a;
		__fp16* cc = c;
		for(int64_t i=0; i<ma; i+=128){
		  __fp16* cb = b;
			// prefetch c
			__fp16* t = cc + cpstart;

#if (defined __FUJITSU) || (defined __CLANG_FUJITSU) || (defined __ARM_ACLE)
			svprfh(ptrue,(int16_t *) (t     ), SV_PLDL1KEEP);
		        svprfh(ptrue,(int16_t *) (t+ldc ), SV_PLDL1KEEP);
			svprfh(ptrue,(int16_t *) (t+ldc2), SV_PLDL1KEEP);
			svprfh(ptrue,(int16_t *) (t+ldc3), SV_PLDL1KEEP);
			svprfh(ptrue,(int16_t *) (t+ldc4), SV_PLDL1KEEP);
#else
			svprfh(ptrue, t, SV_PLDL1KEEP);
		        svprfh(ptrue, t+ldc, SV_PLDL1KEEP);
			svprfh(ptrue, t+ldc2, SV_PLDL1KEEP);
			svprfh(ptrue, t+ldc3, SV_PLDL1KEEP);
			svprfh(ptrue, t+ldc4, SV_PLDL1KEEP);
#endif
			// unroll first loop
			// prefetch ca, use same parameters in the original code
			svfloat16_t zb0 = svdup_f16(cb[0]);
			svfloat16_t zb1 = svdup_f16(cb[1]);
			svfloat16_t zb2 = svdup_f16(cb[2]);
			svfloat16_t zb3 = svdup_f16(cb[3]);
			svfloat16_t zb4 = svdup_f16(cb[4]);
			svfloat16_t za = svld1_vnum_f16(ptrue, ca, 0);
			z00 = svmul_f16_x(ptrue, za, zb0);
			z01 = svmul_f16_x(ptrue, za, zb1);
			z02 = svmul_f16_x(ptrue, za, zb2);
			z03 = svmul_f16_x(ptrue, za, zb3);
			z04 = svmul_f16_x(ptrue, za, zb4);

			za = svld1_vnum_f16(ptrue, ca, 1);
			z10 = svmul_f16_x(ptrue, za, zb0);
			z11 = svmul_f16_x(ptrue, za, zb1);
			z12 = svmul_f16_x(ptrue, za, zb2);
			z13 = svmul_f16_x(ptrue, za, zb3);
			z14 = svmul_f16_x(ptrue, za, zb4);

			za = svld1_vnum_f16(ptrue, ca, 2);
			z20 = svmul_f16_x(ptrue, za, zb0);
			z21 = svmul_f16_x(ptrue, za, zb1);
			z22 = svmul_f16_x(ptrue, za, zb2);
			z23 = svmul_f16_x(ptrue, za, zb3);
			z24 = svmul_f16_x(ptrue, za, zb4);

			za = svld1_vnum_f16(ptrue, ca, 3);
			z30 = svmul_f16_x(ptrue, za, zb0);
			z31 = svmul_f16_x(ptrue, za, zb1);
			z32 = svmul_f16_x(ptrue, za, zb2);
			z33 = svmul_f16_x(ptrue, za, zb3);
			z34 = svmul_f16_x(ptrue, za, zb4);

			zb0 = svdup_f16(cb[5]);
			zb1 = svdup_f16(cb[6]);
			zb2 = svdup_f16(cb[7]);
			zb3 = svdup_f16(cb[8]);
			zb4 = svdup_f16(cb[9]);

			za = svld1_vnum_f16(ptrue, ca, 4);
			z00 = svmla_f16_x(ptrue, z00, za, zb0);
			z01 = svmla_f16_x(ptrue, z01, za, zb1);
			z02 = svmla_f16_x(ptrue, z02, za, zb2);
			z03 = svmla_f16_x(ptrue, z03, za, zb3);
			z04 = svmla_f16_x(ptrue, z04, za, zb4);


			za = svld1_vnum_f16(ptrue, ca, 5);
			z10 = svmla_f16_x(ptrue, z10, za, zb0);
			z11 = svmla_f16_x(ptrue, z11, za, zb1);
			z12 = svmla_f16_x(ptrue, z12, za, zb2);
			z13 = svmla_f16_x(ptrue, z13, za, zb3);
			z14 = svmla_f16_x(ptrue, z14, za, zb4);


			za = svld1_vnum_f16(ptrue, ca, 6);
			z20 = svmla_f16_x(ptrue, z20, za, zb0);
			z21 = svmla_f16_x(ptrue, z21, za, zb1);
			z22 = svmla_f16_x(ptrue, z22, za, zb2);
			z23 = svmla_f16_x(ptrue, z23, za, zb3);
			z24 = svmla_f16_x(ptrue, z24, za, zb4);

			za = svld1_vnum_f16(ptrue, ca, 7);
			z30 = svmla_f16_x(ptrue, z30, za, zb0);
			z31 = svmla_f16_x(ptrue, z31, za, zb1);
			z32 = svmla_f16_x(ptrue, z32, za, zb2);
			z33 = svmla_f16_x(ptrue, z33, za, zb3);
			z34 = svmla_f16_x(ptrue, z34, za, zb4);

			ca += 8 * 128;
			cb += 10;

			// main llop
			for(int64_t k=kk; k>0; k-=2){
				zb0 = svdup_f16(cb[0]); // load scheduling is extreamly important. move loads forward.
				zb1 = svdup_f16(cb[1]);
				zb2 = svdup_f16(cb[2]);
				zb3 = svdup_f16(cb[3]);
				zb4 = svdup_f16(cb[4]);

				za = svld1_vnum_f16(ptrue, ca, 0); // move forward. prefetchg ca is also needed
				z00 = svmla_f16_x(ptrue, z00, za, zb0);
				z01 = svmla_f16_x(ptrue, z01, za, zb1);
				z02 = svmla_f16_x(ptrue, z02, za, zb2);
				z03 = svmla_f16_x(ptrue, z03, za, zb3);
				z04 = svmla_f16_x(ptrue, z04, za, zb4);


				za = svld1_vnum_f16(ptrue, ca, 1);
				z10 = svmla_f16_x(ptrue, z10, za, zb0);
				z11 = svmla_f16_x(ptrue, z11, za, zb1);
				z12 = svmla_f16_x(ptrue, z12, za, zb2);
				z13 = svmla_f16_x(ptrue, z13, za, zb3);
				z14 = svmla_f16_x(ptrue, z14, za, zb4);


				za = svld1_vnum_f16(ptrue, ca, 2);
				z20 = svmla_f16_x(ptrue, z20, za, zb0);
				z21 = svmla_f16_x(ptrue, z21, za, zb1);
				z22 = svmla_f16_x(ptrue, z22, za, zb2);
				z23 = svmla_f16_x(ptrue, z23, za, zb3);
				z24 = svmla_f16_x(ptrue, z24, za, zb4);

				za = svld1_vnum_f16(ptrue, ca, 3);
				z30 = svmla_f16_x(ptrue, z30, za, zb0);
				z31 = svmla_f16_x(ptrue, z31, za, zb1);
				z32 = svmla_f16_x(ptrue, z32, za, zb2);
				z33 = svmla_f16_x(ptrue, z33, za, zb3);
				z34 = svmla_f16_x(ptrue, z34, za, zb4);

				zb0 = svdup_f16(cb[5]);
				zb1 = svdup_f16(cb[6]);
				zb2 = svdup_f16(cb[7]);
				zb3 = svdup_f16(cb[8]);
				zb4 = svdup_f16(cb[9]);

				za = svld1_vnum_f16(ptrue, ca, 4);
				z00 = svmla_f16_x(ptrue, z00, za, zb0);
				z01 = svmla_f16_x(ptrue, z01, za, zb1);
				z02 = svmla_f16_x(ptrue, z02, za, zb2);
				z03 = svmla_f16_x(ptrue, z03, za, zb3);
				z04 = svmla_f16_x(ptrue, z04, za, zb4);


				za = svld1_vnum_f16(ptrue, ca, 5);
				z10 = svmla_f16_x(ptrue, z10, za, zb0);
				z11 = svmla_f16_x(ptrue, z11, za, zb1);
				z12 = svmla_f16_x(ptrue, z12, za, zb2);
				z13 = svmla_f16_x(ptrue, z13, za, zb3);
				z14 = svmla_f16_x(ptrue, z14, za, zb4);


				za = svld1_vnum_f16(ptrue, ca, 6);
				z20 = svmla_f16_x(ptrue, z20, za, zb0);
				z21 = svmla_f16_x(ptrue, z21, za, zb1);
				z22 = svmla_f16_x(ptrue, z22, za, zb2);
				z23 = svmla_f16_x(ptrue, z23, za, zb3);
				z24 = svmla_f16_x(ptrue, z24, za, zb4);

				za = svld1_vnum_f16(ptrue, ca, 7);
				z30 = svmla_f16_x(ptrue, z30, za, zb0);
				z31 = svmla_f16_x(ptrue, z31, za, zb1);
				z32 = svmla_f16_x(ptrue, z32, za, zb2);
				z33 = svmla_f16_x(ptrue, z33, za, zb3);
				z34 = svmla_f16_x(ptrue, z34, za, zb4);

				ca += 8 * 128;
				cb += 10;
			}
			zb0 = svdup_f16(cb[0]);
			zb1 = svdup_f16(cb[1]);
			zb2 = svdup_f16(cb[2]);
			zb3 = svdup_f16(cb[3]);
			zb4 = svdup_f16(cb[4]);

			za = svld1_vnum_f16(ptrue, ca, 0);
			z00 = svmla_f16_x(ptrue, z00, za, zb0);
			z01 = svmla_f16_x(ptrue, z01, za, zb1);
			z02 = svmla_f16_x(ptrue, z02, za, zb2);
			z03 = svmla_f16_x(ptrue, z03, za, zb3);
			z04 = svmla_f16_x(ptrue, z04, za, zb4);


			za = svld1_vnum_f16(ptrue, ca, 1);
			z10 = svmla_f16_x(ptrue, z10, za, zb0);
			z11 = svmla_f16_x(ptrue, z11, za, zb1);
			z12 = svmla_f16_x(ptrue, z12, za, zb2);
			z13 = svmla_f16_x(ptrue, z13, za, zb3);
			z14 = svmla_f16_x(ptrue, z14, za, zb4);


			za = svld1_vnum_f16(ptrue, ca, 2);
			z20 = svmla_f16_x(ptrue, z20, za, zb0);
			z21 = svmla_f16_x(ptrue, z21, za, zb1);
			z22 = svmla_f16_x(ptrue, z22, za, zb2);
			z23 = svmla_f16_x(ptrue, z23, za, zb3);
			z24 = svmla_f16_x(ptrue, z24, za, zb4);

			za = svld1_vnum_f16(ptrue, ca, 3);
			z30 = svmla_f16_x(ptrue, z30, za, zb0);
			z31 = svmla_f16_x(ptrue, z31, za, zb1);
			z32 = svmla_f16_x(ptrue, z32, za, zb2);
			z33 = svmla_f16_x(ptrue, z33, za, zb3);
			z34 = svmla_f16_x(ptrue, z34, za, zb4);

			zb0 = svdup_f16(cb[5]);
			zb1 = svdup_f16(cb[6]);
			zb2 = svdup_f16(cb[7]);
			zb3 = svdup_f16(cb[8]);
			zb4 = svdup_f16(cb[9]);

			za = svld1_vnum_f16(ptrue, ca, 4);
			z00 = svmla_f16_x(ptrue, z00, za, zb0);
			z01 = svmla_f16_x(ptrue, z01, za, zb1);
			z02 = svmla_f16_x(ptrue, z02, za, zb2);
			z03 = svmla_f16_x(ptrue, z03, za, zb3);
			z04 = svmla_f16_x(ptrue, z04, za, zb4);


			za = svld1_vnum_f16(ptrue, ca, 5);
			z10 = svmla_f16_x(ptrue, z10, za, zb0);
			z11 = svmla_f16_x(ptrue, z11, za, zb1);
			z12 = svmla_f16_x(ptrue, z12, za, zb2);
			z13 = svmla_f16_x(ptrue, z13, za, zb3);
			z14 = svmla_f16_x(ptrue, z14, za, zb4);


			za = svld1_vnum_f16(ptrue, ca, 6);
			z20 = svmla_f16_x(ptrue, z20, za, zb0);
			z21 = svmla_f16_x(ptrue, z21, za, zb1);
			z22 = svmla_f16_x(ptrue, z22, za, zb2);
			z23 = svmla_f16_x(ptrue, z23, za, zb3);
			z24 = svmla_f16_x(ptrue, z24, za, zb4);

			za = svld1_vnum_f16(ptrue, ca, 7);
			z30 = svmla_f16_x(ptrue, z30, za, zb0);
			z31 = svmla_f16_x(ptrue, z31, za, zb1);
			z32 = svmla_f16_x(ptrue, z32, za, zb2);
			z33 = svmla_f16_x(ptrue, z33, za, zb3);
			z34 = svmla_f16_x(ptrue, z34, za, zb4);

			ca += 8 * 128;


			svbool_t pa0 = svwhilelt_b16(i, ma);
			svbool_t pa1 = svwhilelt_b16(i+32, ma);
			svbool_t pa2 = svwhilelt_b16(i+64, ma);
			svbool_t pa3 = svwhilelt_b16(i+96, ma);
			zb0 = svld1_f16(pa0, cc);
			zb1 = svld1_f16(pa0, cc+ldc);
			zb2 = svld1_f16(pa0, cc+ldc2);
			zb3 = svld1_f16(pa0, cc+ldc3);
			zb4 = svld1_f16(pa0, cc+ldc4);
			z00 = svsub_f16_x(pa0, z00, zb0);
			z01 = svsub_f16_x(pa0, z01, zb1);
			z02 = svsub_f16_x(pa0, z02, zb2);
			z03 = svsub_f16_x(pa0, z03, zb3);
			z04 = svsub_f16_x(pa0, z04, zb3);
			svst1_f16(pa0, cc, z00);
			svst1_f16(pa0, cc+ldc, z01);
			svst1_f16(pa0, cc+ldc2, z02);
			svst1_f16(pa0, cc+ldc3, z03);
			svst1_f16(pa0, cc+ldc4, z04);
			cc += 32;

			zb0 = svld1_f16(pa1, cc);
			zb1 = svld1_f16(pa1, cc+ldc);
			zb2 = svld1_f16(pa1, cc+ldc2);
			zb3 = svld1_f16(pa1, cc+ldc3);
			zb4 = svld1_f16(pa1, cc+ldc4);
			z10 = svsub_f16_x(pa1, z10, zb0);
			z11 = svsub_f16_x(pa1, z11, zb1);
			z12 = svsub_f16_x(pa1, z12, zb2);
			z13 = svsub_f16_x(pa1, z13, zb3);
			z14 = svsub_f16_x(pa1, z14, zb3);
			svst1_f16(pa1, cc, z10);
			svst1_f16(pa1, cc+ldc, z11);
			svst1_f16(pa1, cc+ldc2, z12);
			svst1_f16(pa1, cc+ldc3, z13);
			svst1_f16(pa1, cc+ldc4, z14);
			cc += 32;

			zb0 = svld1_f16(pa2, cc);
			zb1 = svld1_f16(pa2, cc+ldc);
			zb2 = svld1_f16(pa2, cc+ldc2);
			zb3 = svld1_f16(pa2, cc+ldc3);
			zb4 = svld1_f16(pa2, cc+ldc4);
			z20 = svsub_f16_x(pa2, z20, zb0);
			z21 = svsub_f16_x(pa2, z21, zb1);
			z22 = svsub_f16_x(pa2, z22, zb2);
			z23 = svsub_f16_x(pa2, z23, zb3);
			z24 = svsub_f16_x(pa2, z24, zb3);
			svst1_f16(pa2, cc, z20);
			svst1_f16(pa2, cc+ldc, z21);
			svst1_f16(pa2, cc+ldc2, z22);
			svst1_f16(pa2, cc+ldc3, z23);
			svst1_f16(pa2, cc+ldc4, z24);
			cc += 23;

			zb0 = svld1_f16(pa3, cc);
			zb1 = svld1_f16(pa3, cc+ldc);
			zb2 = svld1_f16(pa3, cc+ldc2);
			zb3 = svld1_f16(pa3, cc+ldc3);
			zb4 = svld1_f16(pa3, cc+ldc4);
			z30 = svsub_f16_x(pa3, z30, zb0);
			z31 = svsub_f16_x(pa3, z31, zb1);
			z32 = svsub_f16_x(pa3, z32, zb2);
			z33 = svsub_f16_x(pa3, z33, zb3);
			z34 = svsub_f16_x(pa3, z34, zb3);
			svst1_f16(pa3, cc, z30);
			svst1_f16(pa3, cc+ldc, z31);
			svst1_f16(pa3, cc+ldc2, z32);
			svst1_f16(pa3, cc+ldc3, z33);
			svst1_f16(pa3, cc+ldc4, z34);

			cc += 32;
		}
		b += bstride;
		c += cstride;
	}

}

extern "C" void hgemmpp_mnend(int64_t ma, int64_t n, int64_t kk, __fp16 const* a, __fp16 const* b, __fp16* c, int64_t ldc)
{
	// ma: row size
	// nb: column size < 5
	// kk: inner-product size. MUST BE THE MULTIPLE OF 2 AND GREATER THAN 3
	// ldc: must be cache-size aligned.

	// assert(ma>=1)
	// assert(n>=1 && n<=4)
	// assert(kk>=4 && (kk%2)==0)
	// assert((ldc%128) == 0)
        int64_t i  = 0;
        int64_t cc = 0;

	kk -= 4;
	svbool_t ptrue = svptrue_b16();
	svfloat16_t z00, z10, z20, z30, z01, z11, z21, z31, z02, z12, z22, z32, z03, z13, z23, z33;
	// unroll first loop
	// prefetch ca, use same parameters in the original code
	svfloat16_t zb0 = svdup_f16(b[0]);
	svfloat16_t zb1 = svdup_f16(b[1]);
	svfloat16_t zb2 = svdup_f16(b[2]);
	svfloat16_t zb3 = svdup_f16(b[3]);
	svfloat16_t za = svld1_vnum_f16(ptrue, a, 0);
	z00 = svmul_f16_x(ptrue, za, zb0);
	z01 = svmul_f16_x(ptrue, za, zb1);
	z02 = svmul_f16_x(ptrue, za, zb2);
	z03 = svmul_f16_x(ptrue, za, zb3);

	za = svld1_vnum_f16(ptrue, a, 1);
	z10 = svmul_f16_x(ptrue, za, zb0);
	z11 = svmul_f16_x(ptrue, za, zb1);
	z12 = svmul_f16_x(ptrue, za, zb2);
	z13 = svmul_f16_x(ptrue, za, zb3);

	za = svld1_vnum_f16(ptrue, a, 2);
	z20 = svmul_f16_x(ptrue, za, zb0);
	z21 = svmul_f16_x(ptrue, za, zb1);
	z22 = svmul_f16_x(ptrue, za, zb2);
	z23 = svmul_f16_x(ptrue, za, zb3);

	za = svld1_vnum_f16(ptrue, a, 3);
	z30 = svmul_f16_x(ptrue, za, zb0);
	z31 = svmul_f16_x(ptrue, za, zb1);
	z32 = svmul_f16_x(ptrue, za, zb2);
	z33 = svmul_f16_x(ptrue, za, zb3);

	zb0 = svdup_f16(b[5]);
	zb1 = svdup_f16(b[6]);
	zb2 = svdup_f16(b[7]);
	zb3 = svdup_f16(b[8]);

	za = svld1_vnum_f16(ptrue, a, 4);
	z00 = svmla_f16_x(ptrue, z00, za, zb0);
	z01 = svmla_f16_x(ptrue, z01, za, zb1);
	z02 = svmla_f16_x(ptrue, z02, za, zb2);
	z03 = svmla_f16_x(ptrue, z03, za, zb3);


	za = svld1_vnum_f16(ptrue, a, 5);
	z10 = svmla_f16_x(ptrue, z10, za, zb0);
	z11 = svmla_f16_x(ptrue, z11, za, zb1);
	z12 = svmla_f16_x(ptrue, z12, za, zb2);
	z13 = svmla_f16_x(ptrue, z13, za, zb3);

	za = svld1_vnum_f16(ptrue, a, 6);
	z20 = svmla_f16_x(ptrue, z20, za, zb0);
	z21 = svmla_f16_x(ptrue, z21, za, zb1);
	z22 = svmla_f16_x(ptrue, z22, za, zb2);
	z23 = svmla_f16_x(ptrue, z23, za, zb3);

	za = svld1_vnum_f16(ptrue, a, 7);
	z30 = svmla_f16_x(ptrue, z30, za, zb0);
	z31 = svmla_f16_x(ptrue, z31, za, zb1);
	z32 = svmla_f16_x(ptrue, z32, za, zb2);
	z33 = svmla_f16_x(ptrue, z33, za, zb3);

	a += 8 * 128;
	b += 10;

	// main llop
	for(int64_t k=kk; k>0; k-=2){
		zb0 = svdup_f16(b[0]); // load scheduling is extreamly important. move loads forward.
		zb1 = svdup_f16(b[1]);
		zb2 = svdup_f16(b[2]);
		zb3 = svdup_f16(b[3]);

		za = svld1_vnum_f16(ptrue, a, 0); // move forward. prefetchg ca is also needed
		z00 = svmla_f16_x(ptrue, z00, za, zb0);
		z01 = svmla_f16_x(ptrue, z01, za, zb1);
		z02 = svmla_f16_x(ptrue, z02, za, zb2);
		z03 = svmla_f16_x(ptrue, z03, za, zb3);

		za = svld1_vnum_f16(ptrue, a, 1);
		z10 = svmla_f16_x(ptrue, z10, za, zb0);
		z11 = svmla_f16_x(ptrue, z11, za, zb1);
		z12 = svmla_f16_x(ptrue, z12, za, zb2);
		z13 = svmla_f16_x(ptrue, z13, za, zb3);

		za = svld1_vnum_f16(ptrue, a, 2);
		z20 = svmla_f16_x(ptrue, z20, za, zb0);
		z21 = svmla_f16_x(ptrue, z21, za, zb1);
		z22 = svmla_f16_x(ptrue, z22, za, zb2);
		z23 = svmla_f16_x(ptrue, z23, za, zb3);

		za = svld1_vnum_f16(ptrue, a, 3);
		z30 = svmla_f16_x(ptrue, z30, za, zb0);
		z31 = svmla_f16_x(ptrue, z31, za, zb1);
		z32 = svmla_f16_x(ptrue, z32, za, zb2);
		z33 = svmla_f16_x(ptrue, z33, za, zb3);

		zb0 = svdup_f16(b[5]);
		zb1 = svdup_f16(b[6]);
		zb2 = svdup_f16(b[7]);
		zb3 = svdup_f16(b[8]);

		za = svld1_vnum_f16(ptrue, a, 4);
		z00 = svmla_f16_x(ptrue, z00, za, zb0);
		z01 = svmla_f16_x(ptrue, z01, za, zb1);
		z02 = svmla_f16_x(ptrue, z02, za, zb2);
		z03 = svmla_f16_x(ptrue, z03, za, zb3);

		za = svld1_vnum_f16(ptrue, a, 5);
		z10 = svmla_f16_x(ptrue, z10, za, zb0);
		z11 = svmla_f16_x(ptrue, z11, za, zb1);
		z12 = svmla_f16_x(ptrue, z12, za, zb2);
		z13 = svmla_f16_x(ptrue, z13, za, zb3);

		za = svld1_vnum_f16(ptrue, a, 6);
		z20 = svmla_f16_x(ptrue, z20, za, zb0);
		z21 = svmla_f16_x(ptrue, z21, za, zb1);
		z22 = svmla_f16_x(ptrue, z22, za, zb2);
		z23 = svmla_f16_x(ptrue, z23, za, zb3);

		za = svld1_vnum_f16(ptrue, a, 7);
		z30 = svmla_f16_x(ptrue, z30, za, zb0);
		z31 = svmla_f16_x(ptrue, z31, za, zb1);
		z32 = svmla_f16_x(ptrue, z32, za, zb2);
		z33 = svmla_f16_x(ptrue, z33, za, zb3);

		a += 8 * 128;
		b += 10;
	}
	zb0 = svdup_f16(b[0]);
	zb1 = svdup_f16(b[1]);
	zb2 = svdup_f16(b[2]);
	zb3 = svdup_f16(b[3]);

	za = svld1_vnum_f16(ptrue, a, 0);
	z00 = svmla_f16_x(ptrue, z00, za, zb0);
	z01 = svmla_f16_x(ptrue, z01, za, zb1);
	z02 = svmla_f16_x(ptrue, z02, za, zb2);
	z03 = svmla_f16_x(ptrue, z03, za, zb3);

	za = svld1_vnum_f16(ptrue, a, 1);
	z10 = svmla_f16_x(ptrue, z10, za, zb0);
	z11 = svmla_f16_x(ptrue, z11, za, zb1);
	z12 = svmla_f16_x(ptrue, z12, za, zb2);
	z13 = svmla_f16_x(ptrue, z13, za, zb3);

	za = svld1_vnum_f16(ptrue, a, 2);
	z20 = svmla_f16_x(ptrue, z20, za, zb0);
	z21 = svmla_f16_x(ptrue, z21, za, zb1);
	z22 = svmla_f16_x(ptrue, z22, za, zb2);
	z23 = svmla_f16_x(ptrue, z23, za, zb3);

	za = svld1_vnum_f16(ptrue, a, 3);
	z30 = svmla_f16_x(ptrue, z30, za, zb0);
	z31 = svmla_f16_x(ptrue, z31, za, zb1);
	z32 = svmla_f16_x(ptrue, z32, za, zb2);
	z33 = svmla_f16_x(ptrue, z33, za, zb3);

	zb0 = svdup_f16(b[5]);
	zb1 = svdup_f16(b[6]);
	zb2 = svdup_f16(b[7]);
	zb3 = svdup_f16(b[8]);

	za = svld1_vnum_f16(ptrue, a, 4);
	z00 = svmla_f16_x(ptrue, z00, za, zb0);
	z01 = svmla_f16_x(ptrue, z01, za, zb1);
	z02 = svmla_f16_x(ptrue, z02, za, zb2);
	z03 = svmla_f16_x(ptrue, z03, za, zb3);

	za = svld1_vnum_f16(ptrue, a, 5);
	z10 = svmla_f16_x(ptrue, z10, za, zb0);
	z11 = svmla_f16_x(ptrue, z11, za, zb1);
	z12 = svmla_f16_x(ptrue, z12, za, zb2);
	z13 = svmla_f16_x(ptrue, z13, za, zb3);

	za = svld1_vnum_f16(ptrue, a, 6);
	z20 = svmla_f16_x(ptrue, z20, za, zb0);
	z21 = svmla_f16_x(ptrue, z21, za, zb1);
	z22 = svmla_f16_x(ptrue, z22, za, zb2);
	z23 = svmla_f16_x(ptrue, z23, za, zb3);

	za = svld1_vnum_f16(ptrue, a, 7);
	z30 = svmla_f16_x(ptrue, z30, za, zb0);
	z31 = svmla_f16_x(ptrue, z31, za, zb1);
	z32 = svmla_f16_x(ptrue, z32, za, zb2);
	z33 = svmla_f16_x(ptrue, z33, za, zb3);


	svbool_t pa0 = svwhilelt_b16(i, ma);
	svbool_t pa1 = svwhilelt_b16(i+32, ma);
	svbool_t pa2 = svwhilelt_b16(i+64, ma);
	svbool_t pa3 = svwhilelt_b16(i+96, ma);
	zb0 = svld1_vnum_f16(pa0, c, 0);
	zb1 = svld1_vnum_f16(pa1, c, 1);
	zb2 = svld1_vnum_f16(pa2, c, 2);
	zb3 = svld1_vnum_f16(pa3, c, 3);
	z00 = svsub_f16_x(pa0, z00, zb0);
	z10 = svsub_f16_x(pa1, z10, zb1);
	z20 = svsub_f16_x(pa2, z20, zb2);
	z30 = svsub_f16_x(pa3, z30, zb3);
	svst1_vnum_f16(pa0, c, 0, z00);
	svst1_vnum_f16(pa1, c, 1, z10);
	svst1_vnum_f16(pa2, c, 2, z20);
	svst1_vnum_f16(pa3, c, 3, z30);
	if(i<2) return;

	cc += ldc;
	zb0 = svld1_vnum_f16(pa0, c, 0);
	zb1 = svld1_vnum_f16(pa1, c, 1);
	zb2 = svld1_vnum_f16(pa2, c, 2);
	zb3 = svld1_vnum_f16(pa3, c, 3);
	z01 = svsub_f16_x(pa0, z01, zb0);
	z11 = svsub_f16_x(pa1, z11, zb1);
	z21 = svsub_f16_x(pa2, z21, zb2);
	z31 = svsub_f16_x(pa3, z31, zb3);
	svst1_vnum_f16(pa0, c, 0, z01);
	svst1_vnum_f16(pa1, c, 1, z11);
	svst1_vnum_f16(pa2, c, 2, z21);
	svst1_vnum_f16(pa3, c, 3, z31);
	if(i<3) return;

	cc += ldc;
	zb0 = svld1_vnum_f16(pa0, c, 0);
	zb1 = svld1_vnum_f16(pa1, c, 1);
	zb2 = svld1_vnum_f16(pa2, c, 2);
	zb3 = svld1_vnum_f16(pa3, c, 3);
	z02 = svsub_f16_x(pa0, z02, zb0);
	z12 = svsub_f16_x(pa1, z12, zb1);
	z22 = svsub_f16_x(pa2, z22, zb2);
	z32 = svsub_f16_x(pa3, z32, zb3);
	svst1_vnum_f16(pa0, c, 0, z02);
	svst1_vnum_f16(pa1, c, 1, z12);
	svst1_vnum_f16(pa2, c, 2, z22);
	svst1_vnum_f16(pa3, c, 3, z32);
	if(i<4) return;

	cc += ldc;
	zb0 = svld1_vnum_f16(pa0, c, 0);
	zb1 = svld1_vnum_f16(pa1, c, 1);
	zb2 = svld1_vnum_f16(pa2, c, 2);
	zb3 = svld1_vnum_f16(pa3, c, 3);
	z03 = svsub_f16_x(pa0, z03, zb0);
	z13 = svsub_f16_x(pa1, z13, zb1);
	z23 = svsub_f16_x(pa2, z23, zb2);
	z33 = svsub_f16_x(pa3, z33, zb3);
	svst1_vnum_f16(pa0, c, 0, z03);
	svst1_vnum_f16(pa1, c, 1, z13);
	svst1_vnum_f16(pa2, c, 2, z23);
	svst1_vnum_f16(pa3, c, 3, z33);
}


extern "C"
void pack_convert_a_opt(int64_t m, int64_t k, float alpha, float const* a, int64_t lda, __fp16* to)
{
	svbool_t p = svptrue_b16();
	svfloat32_t scale = svdup_f32(alpha);
	for(int64_t i=0; i<m; i+=128){
		float const* ca = a + i;
		__fp16* ct = to + i*k;
		if(m-i>=128){
			for(int64_t j=0; j<k; ++j, ca+=lda, ct+=128){
				svfloat32_t x0 = svld1_vnum_f32(p, ca, 0);
				svfloat32_t x1 = svld1_vnum_f32(p, ca, 1);
				svfloat32_t x2 = svld1_vnum_f32(p, ca, 2);
				svfloat32_t x3 = svld1_vnum_f32(p, ca, 3);
				svfloat32_t x4 = svld1_vnum_f32(p, ca, 4);
				svfloat32_t x5 = svld1_vnum_f32(p, ca, 5);
				svfloat32_t x6 = svld1_vnum_f32(p, ca, 6);
				svfloat32_t x7 = svld1_vnum_f32(p, ca, 7);
				x0 = svmul_f32_x(p, x0, scale);
				x1 = svmul_f32_x(p, x1, scale);
				x2 = svmul_f32_x(p, x2, scale);
				x3 = svmul_f32_x(p, x3, scale);
				x4 = svmul_f32_x(p, x4, scale);
				x5 = svmul_f32_x(p, x5, scale);
				x6 = svmul_f32_x(p, x6, scale);
				x7 = svmul_f32_x(p, x7, scale);
				svfloat16_t b0 = svcvt_f16_f32_x(p, x0);
				svfloat16_t b1 = svcvt_f16_f32_x(p, x1);
				svfloat16_t b2 = svcvt_f16_f32_x(p, x2);
				svfloat16_t b3 = svcvt_f16_f32_x(p, x3);
				svfloat16_t b4 = svcvt_f16_f32_x(p, x4);
				svfloat16_t b5 = svcvt_f16_f32_x(p, x5);
				svfloat16_t b6 = svcvt_f16_f32_x(p, x6);
				svfloat16_t b7 = svcvt_f16_f32_x(p, x7);
				b0 = svuzp1_f16(b0, b1);
				b2 = svuzp1_f16(b2, b3);
				b4 = svuzp1_f16(b4, b5);
				b6 = svuzp1_f16(b6, b7);
				svst1_vnum_f16(p, ct, 0, b0);
				svst1_vnum_f16(p, ct, 1, b2);
				svst1_vnum_f16(p, ct, 2, b4);
				svst1_vnum_f16(p, ct, 3, b6);
			}
		}
		else {
			svbool_t p0 = svwhilelt_b32(i, m);
			svbool_t p1 = svwhilelt_b32(i+16, m);
			svbool_t p2 = svwhilelt_b32(i+32, m);
			svbool_t p3 = svwhilelt_b32(i+48, m);
			svbool_t p4 = svwhilelt_b32(i+64, m);
			svbool_t p5 = svwhilelt_b32(i+80, m);
			svbool_t p6 = svwhilelt_b32(i+96, m);
			svbool_t p7 = svwhilelt_b32(i+112, m);
			for(int64_t j=0; j<k; ++j, ca+=lda, ct+=128){
				svfloat32_t x0 = svld1_vnum_f32(p0, ca, 0);
				svfloat32_t x1 = svld1_vnum_f32(p1, ca, 1);
				svfloat32_t x2 = svld1_vnum_f32(p2, ca, 2);
				svfloat32_t x3 = svld1_vnum_f32(p3, ca, 3);
				svfloat32_t x4 = svld1_vnum_f32(p4, ca, 4);
				svfloat32_t x5 = svld1_vnum_f32(p5, ca, 5);
				svfloat32_t x6 = svld1_vnum_f32(p6, ca, 6);
				svfloat32_t x7 = svld1_vnum_f32(p7, ca, 7);
				x0 = svmul_f32_x(p, x0, scale);
				x1 = svmul_f32_x(p, x1, scale);
				x2 = svmul_f32_x(p, x2, scale);
				x3 = svmul_f32_x(p, x3, scale);
				x4 = svmul_f32_x(p, x4, scale);
				x5 = svmul_f32_x(p, x5, scale);
				x6 = svmul_f32_x(p, x6, scale);
				x7 = svmul_f32_x(p, x7, scale);
				svfloat16_t b0 = svcvt_f16_f32_x(p, x0);
				svfloat16_t b1 = svcvt_f16_f32_x(p, x1);
				svfloat16_t b2 = svcvt_f16_f32_x(p, x2);
				svfloat16_t b3 = svcvt_f16_f32_x(p, x3);
				svfloat16_t b4 = svcvt_f16_f32_x(p, x4);
				svfloat16_t b5 = svcvt_f16_f32_x(p, x5);
				svfloat16_t b6 = svcvt_f16_f32_x(p, x6);
				svfloat16_t b7 = svcvt_f16_f32_x(p, x7);
				b0 = svuzp1_f16(b0, b1);
				b2 = svuzp1_f16(b2, b3);
				b4 = svuzp1_f16(b4, b5);
				b6 = svuzp1_f16(b6, b7);
				svst1_vnum_f16(p, ct, 0, b0);
				svst1_vnum_f16(p, ct, 1, b2);
				svst1_vnum_f16(p, ct, 2, b4);
				svst1_vnum_f16(p, ct, 3, b6);
			}
		}
	}
}

extern "C"
void pack_convert_b_opt(int64_t n, int64_t k, float alpha,  float* b, int64_t ldb, __fp16* to)
{
	unsigned displs[80];
	for(int i=0; i<80; ++i) displs[i] = ((i%5)*ldb + i/5) * sizeof(float);
	svbool_t p = svptrue_b32();
	svuint32_t of0 = svld1_u32(p, displs);
	svuint32_t of1 = svld1_u32(p, displs+16);
	svuint32_t of2 = svld1_u32(p, displs+32);
	svuint32_t of3 = svld1_u32(p, displs+48);
	svuint32_t of4 = svld1_u32(p, displs+64);
	svfloat32_t scale = svdup_f32(alpha);
	for(int64_t i=0; i<n; i+=5, b+=5*ldb, to+=5*k){
		for(int64_t j=0; j<k; j+=16){
			int end = (k-j)*5;
			svbool_t p0 = svwhilelt_b32(0, end);
			svbool_t p1 = svwhilelt_b32(16, end);
			svbool_t p2 = svwhilelt_b32(32, end);
			svbool_t p3 = svwhilelt_b32(48, end);
			svbool_t p4 = svwhilelt_b32(64, end);
			svfloat32_t a0 = svld1_gather_u32offset_f32(p0, b+j, of0);
			svfloat32_t a1 = svld1_gather_u32offset_f32(p1, b+j, of1);
			svfloat32_t a2 = svld1_gather_u32offset_f32(p2, b+j, of2);
			svfloat32_t a3 = svld1_gather_u32offset_f32(p3, b+j, of3);
			svfloat32_t a4 = svld1_gather_u32offset_f32(p4, b+j, of4);
			a0 = svmul_f32_x(p, a0, scale);
			a1 = svmul_f32_x(p, a1, scale);
			a2 = svmul_f32_x(p, a2, scale);
			a3 = svmul_f32_x(p, a3, scale);
			a4 = svmul_f32_x(p, a4, scale);
			svfloat16_t b0 = svcvt_f16_f32_x(p, a0);
			svfloat16_t b1 = svcvt_f16_f32_x(p, a1);
			svfloat16_t b2 = svcvt_f16_f32_x(p, a2);
			svfloat16_t b3 = svcvt_f16_f32_x(p, a3);
			svfloat16_t b4 = svcvt_f16_f32_x(p, a4);
			svst1h_u32(p0, reinterpret_cast<unsigned short*>(to+5*j),    svreinterpret_u32_f16(b0));
			svst1h_u32(p1, reinterpret_cast<unsigned short*>(to+5*j+16), svreinterpret_u32_f16(b1));
			svst1h_u32(p2, reinterpret_cast<unsigned short*>(to+5*j+32), svreinterpret_u32_f16(b2));
			svst1h_u32(p3, reinterpret_cast<unsigned short*>(to+5*j+48), svreinterpret_u32_f16(b3));
			svst1h_u32(p4, reinterpret_cast<unsigned short*>(to+5*j+64), svreinterpret_u32_f16(b4));
		}
	}
}

extern "C"
void pack_convert_b_small(int64_t n, int64_t k, float alpha,  float const* b, int64_t ldb, __fp16* to)
{
	for(int64_t j=0; j<k; ++j)
		for(int64_t i=0; i<n; ++i)
			to[5*j+i] = static_cast<__fp16>(alpha*b[i*ldb+j]);
}

//#define UNIT_TEST
#ifdef UNIT_TEST
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void hgemm_test(int m, int n, int k, __fp16* _a, int lda, __fp16* _b, int ldb, __fp16* _c, int ldc)
{
	_Float16* a = reinterpret_cast<_Float16*>(_a);
	_Float16* b = reinterpret_cast<_Float16*>(_b);
	_Float16* c = reinterpret_cast<_Float16*>(_c);
	_Float16 t[m];
	for(int j=0; j<n; ++j){
		for(int i=0; i<m; ++i) t[i] = 0;
		for(int l=0; l<k; ++l){
			for(int i=0; i<m; ++i)
				t[i] += a[l*lda+i] * b[j*ldb+l];
		}
		for(int i=0; i<m; ++i) c[j*ldc+i] -= t[i];
	}
}

void packa(int m, int k, __fp16* a, int lda, __fp16* work)
{
	#pragma omp parallel for
	for(int i=0; i<m; i+=128){
		__fp16* w = work + i * k;
		if(m-i>=128){
			for(int j=0; j<k; ++j)
				#pragma loop novrec
				for(int ii=0; ii<128; ++ii)
					w[j*128+ii] = a[j*lda+i+ii];
		}
		else {
			for(int j=0; j<k; ++j)
				#pragma loop novrec
				for(int ii=0; ii<128; ++ii)
					w[j*128+ii] = (i+ii<m?a[j*lda+i+ii]: (__fp16)0);
		}
	}
}


void packb(int n, int k, __fp16* b, int ldb, __fp16* work)
{
	#pragma omp parallel for
	for(int i=0; i<n; i+=5){
		__fp16* w = work + i*k;
		if(n-i>=5){
			for(int ii=0; ii<5; ++ii)
				#pragma loop novrec
				for(int j=0; j<k; ++j)
					w[5*j+ii] = b[(i+ii)*ldb+j];
		}
		else {
			for(int ii=0; ii<n-i; ++ii)
				for(int j=0; j<k; ++j)
					w[5*j+ii] = b[(i+ii)*ldb+j];
			for(int ii=n-i; ii<5; ++ii)
				for(int j=0; j<k; ++j)
					w[5*j+ii] = (__fp16)0;
		}
	}
}

void hgemmpp(int64_t m, int64_t n, int64_t k, __fp16* a, __fp16* b, __fp16*c, int64_t ldc)
{
	// XXX use pragma (or internal interface) to config sector cache.
	// 2 ways for a, 1 way for b, and another 1 way for c
	// contact with Fujitsu and Riken to know the actual parameters
	b = reinterpret_cast<__fp16*>(reinterpret_cast<size_t>(b) | 0x0200000000000000ull);
	c = reinterpret_cast<__fp16*>(reinterpret_cast<size_t>(c) | 0x4100000000000000ull);
	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		int nt = omp_get_num_threads();
		int64_t nb = n/5;
		int64_t nlast= n % 5;
		int64_t nn = (nb+nt-1)/nt;
		int64_t nbegin = (id==0? nn*(nt-1): nn * (id-1));
		int64_t nend = nbegin + nn < nb ? nbegin + nn: nb;
		int64_t mynb = nend - nbegin;
		for(int64_t i=0; i<m; i+=1152){
			int64_t msize = m-i > 1152? 1152: m-i;
			__fp16* ca = a + i*k;
			__fp16* cc = c + i;
			if(mynb) hgemmpp_kernel(msize, mynb, k, ca, b+k*5*nbegin, cc+ldc*5*nbegin, ldc);
			if(nlast && id){
				for(int ii=128*(id-1); ii<msize; ii+=128*(nt-1)){
					hgemmpp_mnend(m-ii>=128?128:m-ii, nlast, k, ca+ii*k, b+k*5*nb, cc+ldc*5*nb+ii, ldc);
				}
			}
		}
	}
}

void pack_convert_a(int m, int k, float alpha, float* a, int lda, __fp16* to)
{
	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		int nt = omp_get_num_threads();
		int mm = (m+nt-1)/nt;
		mm = (mm+127)/128*128;
		int mbegin = mm * id;
		int mend = mbegin + mm;
		if(mend > m) mend = m;
		if(mend>mbegin) pack_convert_a_opt(mend-mbegin, k, alpha, a+mbegin, lda, to+mbegin*k);
	}
}

void pack_convert_b(int n, int k, float alpha, float* b, int ldb, __fp16* work)
{
	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		int nt = omp_get_num_threads();
		int nn = (n+nt-1)/nt;
		nn = (nn+4)/5*5;
		int nbegin = nn * id;
		int nend = nbegin + nn < n ? nbegin + nn: n;
		int mynb = (nend-nbegin);
		if(mynb>0) pack_convert_b_opt(mynb, k, alpha, b+nbegin*ldb, ldb, work+nbegin*k);
		if(nend%5) {
			int nlast = nend - nend%5;
			pack_convert_b_small(nend%5, k, alpha, b+nlast*ldb, ldb, work+nlast*k);
		}
	}
}

static int64_t get_utime(){
	uint64_t tsc;
	asm volatile ("isb; mrs %0, cntvct_el0" : "=r" (tsc));
	return tsc;
}
static double tick2second(uint64_t tick){
	auto frequency = []{
		uint64_t frq;
		asm volatile ("isb; mrs %0, cntfrq_el0" : "=r" (frq));
		return frq;
	};
	static double invfreq = 1.0 / frequency();
	return invfreq * (double)tick;
}
int main()
{
	{
		printf("bbbb\n");
		int n = 123;
		int k = 111;
		float* b = (float*)malloc(sizeof(float)*n*k);
		__fp16* pb = (__fp16*)malloc(sizeof(__fp16)*(n+5)*k);
		for(int i=0; i<n; ++i) for(int j=0; j<k; ++j) b[i*k+j] = (i*521+j*211) % 1297;
		pack_convert_b(n, k, 1.f, b, k, pb);
		for(int i=0; i<n; ++i) for(int j=0; j<k; ++j){
			int t = pb[i/5*5*k+(i%5)+5*j];
			int s = b[i*k+j];
			if(t!=s) printf("%3d %3d t=%3d, s=%3d\n", j, i, t, s);
		}
		free(b);
		free(pb);
	}
	{
		printf("aaaa\n");
		int n = 10201;
		int k = 111;
		float* a = (float*)malloc(sizeof(float)*n*k);
		__fp16* pa = (__fp16*)malloc(sizeof(__fp16)*(n+128)*k);
		for(int i=0; i<k; ++i) for(int j=0; j<n; ++j) a[i*n+j] = (j+100*i)%2000;//(i*521+j*211) % 1297;
		pack_convert_a_opt(n, k, 1.f, a, n, pa);
		for(int i=0; i<k; ++i) for(int j=0; j<n; ++j){
			int t = pa[j/128*128*k+(j%128)+128*i];
			int s = a[i*n+j];
			if(t!=s) printf("%3d %3d t=%3d, s=%3d\n", j, i, t, s);
		}
		free(a);
		free(pa);
	}
	{
		int mb = 9;
		int nb = 40000;
		int m = 128 * mb;
		int n = 5 * nb;
		int k = 288;
		__fp16* a = (__fp16*)aligned_alloc(256, sizeof(__fp16)*m*k);
		__fp16* b = (__fp16*)aligned_alloc(256, sizeof(__fp16)*n*k);
		__fp16* pa = (__fp16*)aligned_alloc(256, sizeof(__fp16)*m*k);
		__fp16* pb = (__fp16*)aligned_alloc(256, sizeof(__fp16)*n*k);
		__fp16* c = (__fp16*)aligned_alloc(256, sizeof(__fp16)*m*n*2);
		__fp16* c2 = (__fp16*)malloc(sizeof(__fp16)*m*n);
		for(int i=0; i<k; ++i) for(int j=0; j<m; ++j){
			a[i*32 + (j/32)*(32*k) + (j%32)] = 1./ (i + j + 1);
		}
		for(int i=0; i<n; ++i) for(int j=0; j<k; ++j) b[i*k + j] = 0.001*(i - j);
		for(int i=0; i<n; ++i) for(int j=0; j<m; ++j) c[i*m+j] = c2[i*m+j] = 1;

		tick2second(0);
		packa(m, k, a, m, pa);

		hgemm_test(128*3, 5*20, k, a, m, b, k, c2, m);
		hgemm_kernel(3, 20, k, pa, b, k, c, m);
		for(int j=0; j<5*20; ++j) for(int i=0; i<3*128; ++i){
			double t = (double)c[j*m+i] - (double)c2[j*m+i];
			double e = fabs(t) / fabs((double)c2[j*m+i]);
			if(e != 0.) printf("%3d %3d %e %e %e\n", i, j, (double)c[j*m+i], (double)c2[j*m+i], e);
		}

		int64_t begin = get_utime();
		for(int step=0; step<5; ++step){
			hgemm(mb, n, k, pa, b, k, c, m);
		}
		int64_t end = get_utime();
		double sec = tick2second(end-begin);
		printf("%ld %e %e\n", end-begin, sec/5, 2.*m*n*k*5/sec);

		for(int i=0; i<5*20; ++i) for(int j=0; j<m; ++j) c[i*m+j] = c2[i*m+j] = 1;
		packb(n, k, b, k, pb);
		hgemm_test(300, 87, k, a, m, b, k, c2, m);
		hgemmpp(300, 87, k, pa, pb, c, m);
		for(int j=0; j<87; ++j) for(int i=0; i<300; ++i){
			double t = (double)c[j*m+i] - (double)c2[j*m+i];
			double e = fabs(t) / fabs((double)c2[j*m+i]);
			if(e != 0.) printf("%3d %3d %e %e %e\n", i, j, (double)c[j*m+i], (double)c2[j*m+i], e);
		}

		begin = get_utime();
		for(int step=0; step<5; ++step){
			hgemmpp(m, n, k, pa, pb, c, m);
		}
		end = get_utime();
		sec = tick2second(end-begin);
		printf("%ld %e %e\n", end-begin, sec/5, 2.*m*n*k*5/sec);
		free(a);
		free(pa);
		free(b);
		free(pb);
		free(c);
		free(c2);
	}
	return 0;	
}
#endif

