#include "back_buffer.hpp"
#include "fp16sim.hpp"
#include "svesim.hpp"
#include <math.h>

extern "C" void bb_init_piv(int n, int* a, int* idx)
{
	// initialize piv.
	// The location of the blocks are quite complicated.
	// With double-decker layout, the upper half part of the buffer are free to use.
	// We place the back-buffer to there, thus, we don't need extra storage for back-buffer.
	// As the decomposition progresses, the top part of the storage is replaced with the higher-precision data.
	// Therefore, we need to move the back-buffer not to be overwritten.
	// To reduce the data-movement cost, we used non-natural ordering of the blocks.
	// This function build the ordering in reversed way.

	// 1) initialize pivot with the last position of the blocks
	for(int i=0; i<n; ++i) {a[2*i] = i; a[2*i+i] = -1;} // a[2*i] contains i-th block
	for(int i=0; i<n; ++i) idx[i] = 2*i; // reversed index
	int iempty = 2*n - 3;
	int iend = 2*n - 2;
	// 2) apply inverse of pop(row) from row=nprow-2 to 0
	while(iempty > 0){
		a[iempty] = a[iend];
		idx[a[iend]] = iempty;
		iempty -= 2;
		--iend;
	}
}

#if 0 //ndef __ARM_FEATURE_SVE
extern "C" void bb_writeback_b_impl_fp16(int b, fp16* __restrict__ x, fp16* __restrict__ y)
{
	// we need good compile options
	// for intel compiler, use /fp-model:precies
	// back-buffer write-back blocks with compensated sum. This gives extra accuracy.

	fp16 const scaledown = fp16(1.f/BB_NCYCLE);
	fp16 const scaleup = fp16(BB_NCYCLE * 1.f);
	for(int i=0; i<b; ++i){
		// compensated two_sum
		fp16 t0 = x[i];
		fp16 t1 = y[i] * scaledown;
		fp16 t2 = t0 + t1;
		fp16 t3 = t2 - t0;
		fp16 t4 = t1 - t3;
		x[i] = t2;
		y[i] = t4 * scaleup;
		
	}

}
#else
extern "C" void bb_writeback_b_impl_fp16(int b, fp16* __restrict__ x, fp16* __restrict__ y)
{
	fp16 const scaledown = 1.f/BB_NCYCLE;
	fp16 const scaleup = BB_NCYCLE * 1.f;
	svfloat16_t svsd = svdup_f16(scaledown);
	svfloat16_t svsu = svdup_f16(scaleup);
	#if 0
	int nn = svcnth();
	for(int i=0; i<b; i+=nn){
		svbool_t t = svwhilelt_b16(i, b);
		svfloat16_t t0 = svld1_vnum_f16(t, x+i, 0);
		svfloat16_t t1 = svld1_vnum_f16(t, y+i, 0);
		svfloat16_t t2 = svmla_f16_x(t, t0, t1, svsd);
		svfloat16_t t3 = svsub_f16_x(t, t2, t0);
		svfloat16_t t4 = svnmls_f16_x(t, t3, t1, svsd);
		svfloat16_t t5 = svmul_f16_x(t, t4, svsu);
		svst1_vnum_f16(t, x+i, 0, t2);
		svst1_vnum_f16(t, y+i, 0, t5);
	}
	#else
	if((b%32) == 0){
		svbool_t t = svptrue_b16();
		//#pragma clang loop unroll(enable)
		for(int i=0; i<b; i+=32){
			svfloat16_t t0 = svld1_vnum_f16(t, x+i, 0);
			svfloat16_t t1 = svld1_vnum_f16(t, y+i, 0);
			svfloat16_t t2 = svmla_f16_x(t, t0, t1, svsd);
			svfloat16_t t3 = svsub_f16_x(t, t2, t0);
			svfloat16_t t4 = svnmls_f16_x(t, t3, t1, svsd);
			svfloat16_t t5 = svmul_f16_x(t, t4, svsu);
			svst1_vnum_f16(t, x+i, 0, t2);
			svst1_vnum_f16(t, y+i, 0, t5);
		}
	}
	else {
		for(int i=0; i<b; i+=svcnth()){
			svbool_t t = svwhilelt_b16(i, b);
			svfloat16_t t0 = svld1_vnum_f16(t, x+i, 0);
			svfloat16_t t1 = svld1_vnum_f16(t, y+i, 0);
			svfloat16_t t2 = svmla_f16_x(t, t0, t1, svsd);
			svfloat16_t t3 = svsub_f16_x(t, t2, t0);
			svfloat16_t t4 = svnmls_f16_x(t, t3, t1, svsd);
			svfloat16_t t5 = svmul_f16_x(t, t4, svsu);
			svst1_vnum_f16(t, x+i, 0, t2);
			svst1_vnum_f16(t, y+i, 0, t5);
		}
	}
	#endif
}
#endif

#if 1 //ndef __ARM_FEATURE_SVE
extern "C" void bb_writeback_b_impl_fp32(int b, float* __restrict__ x, float* __restrict__ y)
{

	float const scaledown = 1.f/BB_NCYCLE;
	float const scaleup = BB_NCYCLE;
	for(int i=0; i<b; ++i){
		float t0 = x[i];
		float t1 = y[i] * scaledown;
		float t2 = t0 + t1;
		float t3 = t2 - t0;
		float t4 = t1 - t3;
		x[i] = t2;
		y[i] = t4 * scaleup;
	}
}
#else
extern "C" void bb_writeback_b_impl_fp32(int b, float* __restrict__ x, float* __restrict__ y)
{
	float const scaledown = 1.f/BB_NCYCLE;
	float const scaleup = BB_NCYCLE * 1.f;
	svfloat32_t svsd = svdup_f32(scaledown);
	svfloat32_t svsu = svdup_f32(scaleup);
	int nn = svcntw();
	for(int i=0; i<b; i+=nn){
		// compensated two_sum
		svbool_t t = svwhilelt_b32(i, b);
		svfloat32_t t0 = svld1_vnum_f32(t, x+i, 0);
		svfloat32_t t1 = svld1_vnum_f32(t, y+i, 0);
		svfloat32_t t2 = svmla_f32_x(t, t0, t1, svsd);
		svfloat32_t t3 = svsub_f32_x(t, t2, t0);
		svfloat32_t t4 = svnmls_f32_x(t, t3, t1, svsd);
		svfloat32_t t5 = svmul_f32_x(t, t4, svsu);
		svst1_vnum_f32(t, x+i, 0, t2);
		svst1_vnum_f32(t, y+i, 0, t5);
	}

}
#endif
extern "C" void bb_writeback_b_impl_int_fp16(int b, fp16* x, fp16* y)
{
	// using 16bit integer for back-buffer is another idea. 
	// this is better the the above in terms of the value range.
	// you need to scale the input so well to avoid overflows.
	short* xi = reinterpret_cast<short*>(x);
	fp16 const scaledown = 1.f/BB_NCYCLE;
	fp16 const scaleup = BB_NCYCLE * 1.f;
	int stride = svcnth();
	svfloat16_t svdw = svdup_f16(scaledown);
	svfloat16_t svup = svdup_f16(scaleup);
	for(int i=0; i<b; i+=stride){
		svbool_t t  = svwhilelt_b16(i, b);
		svint16_t t0 = svld1_vnum_s16(t, xi+i, 0);
		svfloat16_t t1 = svld1_vnum_f16(t, y+i, 0);
		t1 = svmul_f16_x(t, t1, svdw);
		svfloat16_t t2 = svrintn_f16_x(t, t1); // round to int. the rsult should be ~ 1
		svfloat16_t t3 = svsub_f16_x(t, t1, t2);
		t3 = svmul_f16_x(t, t3, svup);
		svint16_t t4 = svcvt_s16_f16_x(t, t2);
		t0 = svqadd_s16(t0, t4);
		svst1_vnum_s16(t, xi+i, 0, t0);
		svst1_vnum_f16(t, y+i, 0, t3);
		
	}
}
extern "C" void bb_writeback_b_impl_int_fp32(int b, float* x, float* y)
{
	int32_t* xi = reinterpret_cast<int32_t*>(x);
	float const scaledown = 1.f/BB_NCYCLE;
	float const scaleup = BB_NCYCLE * 1.f;
	#pragma omp simd
	for(int i=0; i<b; ++i){
		float t1 = y[i] * scaledown;
		float t2 = roundf(t1);
		float t3 = t1 - t2;
		xi[i] += (int)t2;
		y[i] = t3 * scaleup;
	}
}
extern "C" void bb_writeback_impl_fp16(int b, int rowstart, int nprow, int const* rpiv, fp16* c)
{
	// write-back all blocks in a column
	for(int i=rowstart; i<nprow; ++i){
		// blocks for back-buffers are distributed over a column. we need reverse index (rpiv) to address.
		bb_writeback_b_impl_fp16(b, c+b*rpiv[i], c+b*(i+nprow));
	}
}
extern "C" void bb_writeback_impl_fp32(int b, int rowstart, int nprow, int const* rpiv, float* c)
{
	for(int i=rowstart; i<nprow; ++i)
		bb_writeback_b_impl_fp32(b, c+b*rpiv[i], c+b*(i+nprow));
}
extern "C" void bb_writeback_impl_int_fp16(int b, int rowstart, int nprow, int const* rpiv, fp16* c)
{
	for(int i=rowstart; i<nprow; ++i)
		bb_writeback_b_impl_int_fp16(b, c+b*rpiv[i], c+b*(i+nprow));
}
extern "C" void bb_writeback_impl_int_fp32(int b, int rowstart, int nprow, int const* rpiv, float* c)
{
	for(int i=rowstart; i<nprow; ++i)
		bb_writeback_b_impl_int_fp32(b, c+b*rpiv[i], c+b*(i+nprow));
}

#ifdef UNIT_TEST
#include <stdlib.h>
#include <stdio.h>
int main()
{
	for(int b=1; b<300; ++b){
		double* dx, *dy;
		fp16* x, *y;
		dx = (double*)std::malloc(sizeof(double)*b);
		dy = (double*)std::malloc(sizeof(double)*b);
		x = (fp16*)std::malloc(sizeof(fp16) * 3 * b);
		y = (fp16*)std::malloc(sizeof(fp16) * 3 * b);

		for(int i=0; i<3*b; ++i) x[i] = y[i] = (float)-1234;
		for(int i=0; i<b; ++i) dx[i] = (int)(std::rand() * (32./RAND_MAX));
		for(int i=0; i<b; ++i) dy[i] = std::rand() * (64./RAND_MAX);

		fp16* xx = x + b;
		short* xi = reinterpret_cast<short*>(xx);
		fp16* yy = y + b;

		for(int i=0; i<b; ++i) xi[i] = (int)dx[i];
		for(int i=0; i<b; ++i) yy[i] = (float)dy[i];


		bb_writeback_b_impl_int_fp16(b, xx, yy);

		double er = 0.;
		for(int i=0; i<b; ++i){
			double t = 32 * (int)xi[i] + (double)yy[i];
			double r = 32 * dx[i] + dy[i];
			double e = fabs(t-r)/fabs(r);
			printf("#  %d %f %f %e\n", i, t, r, e);
			er = er > e ? er: e;
		}
		printf("%d %e\n", b, er);

		free(dx);
		free(dy);
		free(x);
		free(y);

	}
}
#endif
