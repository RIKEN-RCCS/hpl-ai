#ifndef SVESIM_HPP
#define SVESIM_HPP

// ARM sve wrapper.
// Reimpelment with SSE, AVX, or sometihng if performance is important.
// (basically, we do not use this wrapper in time-consuming parts.)
// Add functions if needed.

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#else
#include <cstdint>
#include <math.h>
#include "fp16sim.hpp"
#define SVE_VLEN 64
#define svcntd() (SVE_VLEN/8)
#define svcntw() (SVE_VLEN/4)
#define svcnth() (SVE_VLEN/2)
struct svbool_t {
	bool x[SVE_VLEN];
};
static svbool_t svptrue_b64()
{
	svbool_t r;
	for(int i=0; i<SVE_VLEN/8; ++i) r.x[8*i] = true;
	return r;
}
static svbool_t svwhilelt_b64(int64_t begin, int64_t end)
{
	svbool_t r;
	for(int i=0; i<SVE_VLEN/8; ++i) r.x[8*i] = i+begin < end;
	return r;
}
static svbool_t svptrue_b32()
{
	svbool_t r;
	for(int i=0; i<SVE_VLEN/4; ++i) r.x[4*i] = true;
	return r;
}
static svbool_t svwhilelt_b32(int64_t begin, int64_t end)
{
	svbool_t r;
	for(int i=0; i<SVE_VLEN/4; ++i) r.x[4*i] = i+begin < end;
	return r;
}
static svbool_t svptrue_b16()
{
	svbool_t r;
	for(int i=0; i<SVE_VLEN/2; ++i) r.x[2*i] = true;
	return r;
}
static svbool_t svwhilelt_b16(int64_t begin, int64_t end)
{
	svbool_t r;
	for(int i=0; i<SVE_VLEN/2; ++i) r.x[2*i] = i+begin < end;
	return r;
}

struct svint64_t {
	int64_t x[SVE_VLEN/8];
};
static svint64_t svdup_s64(int64_t x)
{
	svint64_t r;
	for(int i=0; i<SVE_VLEN/8; ++i) r.x[i] = x;
	return r;
}
static svint64_t svld1_s64(svbool_t t, int64_t const* x)
{
	svint64_t r;
	for(int i=0; i<SVE_VLEN/8; ++i) r.x[i] = (t.x[8*i] ? x[i]: 0ll);
	return r;
}
static svint64_t svmad_s64_x(svbool_t t, svint64_t a, svint64_t b, svint64_t c)
{
	svint64_t r;
	for(int i=0; i<SVE_VLEN/8; ++i)
		r.x[i] = (t.x[8*i] ? a.x[i] * b.x[i] + c.x[i]: a.x[i]);
	return r;
}
static svint64_t svindex_s64(int64_t base, int64_t step)
{
	svint64_t r;
	for(int i=0; i<SVE_VLEN/8; ++i) r.x[i] = base + i*step;
	return r;
}
struct svfloat64_t {
	double x[SVE_VLEN/8];
};
static svfloat64_t svdup_f64(double x)
{
	svfloat64_t r;
	for(int i=0; i<SVE_VLEN/8; ++i) r.x[i] = x;
	return r;
}
static svfloat64_t svld1_vnum_f64(svbool_t t, double const* x, int vnum)
{
	svfloat64_t r;
	for(int i=0; i<SVE_VLEN/8; ++i) r.x[i] = (t.x[8*i] ? x[vnum*SVE_VLEN/8+i]: 0.);
	return r;
}
static void svst1_vnum_f64(svbool_t t, double* x, int vnum, svfloat64_t r)
{
	for(int i=0; i<SVE_VLEN/8; ++i){
		if(t.x[8*i])
			x[vnum*SVE_VLEN/8+i] = r.x[i];
	}
}
static svfloat64_t svadd_f64_x(svbool_t t, svfloat64_t a, svfloat64_t b)
{
	svfloat64_t r;
	for(int i=0; i<SVE_VLEN/8; ++i)
		r.x[i] = (t.x[8*i] ? a.x[i] + b.x[i]: a.x[i]);
	return r;
}
static svfloat64_t svmla_f64_x(svbool_t t, svfloat64_t a, svfloat64_t b, svfloat64_t c)
{
	svfloat64_t r;
	for(int i=0; i<SVE_VLEN/8; ++i)
		r.x[i] = (t.x[8*i] ? a.x[i] + b.x[i] * c.x[i]: a.x[i]);
	return r;
}
static svfloat64_t svmla_n_f64_x(svbool_t t, svfloat64_t a, svfloat64_t b, float c)
{
	svfloat64_t r;
	for(int i=0; i<SVE_VLEN/8; ++i)
		r.x[i] = (t.x[8*i] ? a.x[i] + b.x[i] * c: a.x[i]);
	return r;
}
static svfloat64_t svcvt_f64_s64_x(svbool_t t, svint64_t x){
	svfloat64_t r;
	for(int i=0; i<SVE_VLEN/8; ++i)
		r.x[i] = (t.x[8*i] ? x.x[i]: 0.);
	return r;
}
struct svfloat32_t {
	float x[SVE_VLEN/4];
};
static svfloat32_t svdup_f32(float x)
{
	svfloat32_t r;
	for(int i=0; i<SVE_VLEN/4; ++i) r.x[i] = x;
	return r;
}
static svfloat32_t svld1_vnum_f32(svbool_t t, float const* x, int vnum)
{
	svfloat32_t r;
	for(int i=0; i<SVE_VLEN/4; ++i) r.x[i] = (t.x[4*i] ? x[vnum*SVE_VLEN/4+i]: 0.f);
	return r;
}
static void svst1_vnum_f32(svbool_t t, float* x, int vnum, svfloat32_t r)
{
	for(int i=0; i<SVE_VLEN/4; ++i){
		if(t.x[4*i])
			x[vnum*SVE_VLEN/4+i] = r.x[i];
	}
}
static svfloat32_t svadd_f32_x(svbool_t t, svfloat32_t a, svfloat32_t b)
{
	svfloat32_t r;
	for(int i=0; i<SVE_VLEN/4; ++i)
		r.x[i] = (t.x[4*i] ? a.x[i] + b.x[i]: a.x[i]);
	return r;
}
static svfloat32_t svmla_f32_x(svbool_t t, svfloat32_t a, svfloat32_t b, svfloat32_t c)
{
	svfloat32_t r;
	for(int i=0; i<SVE_VLEN/4; ++i)
		r.x[i] = (t.x[4*i] ? a.x[i] + b.x[i] * c.x[i]: a.x[i]);
	return r;
}
static svfloat32_t svmls_f32_x(svbool_t t, svfloat32_t a, svfloat32_t b, svfloat32_t c)
{
	svfloat32_t r;
	for(int i=0; i<SVE_VLEN/4; ++i)
		r.x[i] = (t.x[4*i] ? a.x[i] - b.x[i] * c.x[i]: a.x[i]);
	return r;
}


static svfloat32_t svnmls_f32_x(svbool_t t, svfloat32_t a, svfloat32_t b, svfloat32_t c)
{
	svfloat32_t r;
	for(int i=0; i<SVE_VLEN/4; ++i)
		r.x[i] = (t.x[4*i] ? b.x[i] * c.x[i] - a.x[i]: a.x[i]);
	return r;
}


struct svfloat16_t {
	fp16 x[SVE_VLEN/2];
};
struct svint16_t {
	short x[SVE_VLEN/2];
};
static svfloat16_t svdup_f16(fp16 x)
{
	svfloat16_t r;
	for(int i=0; i<SVE_VLEN/2; ++i) r.x[i] = x;
	return r;
}
static svfloat16_t svld1_vnum_f16(svbool_t t, fp16 const* x, int vnum)
{
	svfloat16_t r;
	for(int i=0; i<SVE_VLEN/2; ++i) r.x[i] = (t.x[2*i] ? x[vnum*SVE_VLEN/2+i]: fp16(0.f));
	return r;
}
static void svst1_vnum_f16(svbool_t t, fp16* x, int vnum, svfloat16_t r)
{
	for(int i=0; i<SVE_VLEN/2; ++i){
		if(t.x[2*i])
			x[vnum*SVE_VLEN/2+i] = r.x[i];
	}
}
static svfloat16_t svadd_f16_x(svbool_t t, svfloat16_t a, svfloat16_t b)
{
	svfloat16_t r;
	for(int i=0; i<SVE_VLEN/2; ++i)
		r.x[i] = (t.x[2*i] ? a.x[i] + b.x[i]: a.x[i]);
	return r;
}
static svfloat16_t svsub_f16_x(svbool_t t, svfloat16_t a, svfloat16_t b)
{
	svfloat16_t r;
	for(int i=0; i<SVE_VLEN/2; ++i)
		r.x[i] = (t.x[2*i] ? a.x[i] - b.x[i]: a.x[i]);
	return r;
}
static svfloat16_t svmul_f16_x(svbool_t t, svfloat16_t a, svfloat16_t b)
{
	svfloat16_t r;
	for(int i=0; i<SVE_VLEN/2; ++i)
		r.x[i] = (t.x[2*i] ? a.x[i] * b.x[i]: a.x[i]);
	return r;
}
static svfloat16_t svmla_f16_x(svbool_t t, svfloat16_t a, svfloat16_t b, svfloat16_t c)
{
	svfloat16_t r;
	for(int i=0; i<SVE_VLEN/2; ++i)
		r.x[i] = (t.x[2*i] ? a.x[i] + b.x[i] * c.x[i]: a.x[i]);
	return r;
}
static svfloat16_t svmls_f16_x(svbool_t t, svfloat16_t a, svfloat16_t b, svfloat16_t c)
{
	svfloat16_t r;
	for(int i=0; i<SVE_VLEN/2; ++i)
		r.x[i] = (t.x[2*i] ? a.x[i] - b.x[i] * c.x[i]: a.x[i]);
	return r;
}
static svfloat16_t svnmls_f16_x(svbool_t t, svfloat16_t a, svfloat16_t b, svfloat16_t c)
{
	svfloat16_t r;
	for(int i=0; i<SVE_VLEN/2; ++i)
		r.x[i] = (t.x[2*i] ? b.x[i] * c.x[i] - a.x[i]: a.x[i]);
	return r;
}
static svfloat16_t svrintn_f16_x(svbool_t t, svfloat16_t a)
{
	svfloat16_t r;
	for(int i=0; i<SVE_VLEN/2; ++i)
		r.x[i] = (t.x[2*i] ? fp16(roundf((float)a.x[i])): a.x[i]);
	return r;
}
static svint16_t svcvt_s16_f16_x(svbool_t t, svfloat16_t a)
{
	svint16_t r;
	for(int i=0; i<SVE_VLEN/2; ++i)
		r.x[i] = (t.x[2*i] ? (short)(float)a.x[i]: (short)0);
	return r;
}

static svint16_t svld1_vnum_s16(svbool_t t, short const* x, int vnum)
{
	svint16_t r;
	for(int i=0; i<SVE_VLEN/2; ++i) r.x[i] = (t.x[2*i] ? x[vnum*SVE_VLEN/2+i]: (short)0);
	return r;
}
static void svst1_vnum_s16(svbool_t t, short* x, int vnum, svint16_t r)
{
	for(int i=0; i<SVE_VLEN/2; ++i){
		if(t.x[2*i])
			x[vnum*SVE_VLEN/2+i] = r.x[i];
	}
}
static svint16_t svqadd_s16(svint16_t a, svint16_t b)
{
	int const max = (1<<15)-1;
	int const min = -(1<<15);
	svint16_t r;
	for(int i=0; i<SVE_VLEN/2; ++i){
		int x = a.x[i];
		int y = b.x[i];
		if(x+y >= max) r.x[i] = max;
		else if(x+y<=min) r.x[i] = min;
		else r.x[i] = x + y;
	}
	return r;
}
#endif
#endif
