#include <stdint.h>
#include "../fp16sim.hpp"
#ifdef __ARM_FEATURE_SVE
#define HGEMM_PACK
#define HGEMM_PACK_MUNIT 128
#define HGEMM_PACK_NUNIT 5
#define HGEMM_PACK_MB 1152
#define HGEMM_PACK_NB 100
#endif

#ifdef __AVX2__
#define HGEMM_PACK
#define HGEMM_PACK_MUNIT 16
#define HGEMM_PACK_NUNIT 4
#define HGEMM_PACK_MB 64
#define HGEMM_PACK_NB 32
#endif

#ifdef HGEMM_PACK
extern "C" {
void hgemmpp_kernel(int64_t m, int64_t nb, int64_t k, fp16 const* __restrict__ a, fp16 const* __restrict__ b, fp16* __restrict__ c, int64_t ldc);
void hgemmpp_mnend(int64_t m, int64_t n, int64_t k, fp16 const* __restrict__ a, fp16 const* __restrict__ b, fp16* __restrict__ c, int64_t ldc);
void pack_convert_a_opt(int64_t m, int64_t k, float alpha, float const* a, int64_t lda, fp16* to);
void pack_convert_b_opt(int64_t n, int64_t k, float alpha, float const* b, int64_t ldb, fp16* to);
void pack_convert_b_small(int64_t n, int64_t k, float alpha, float const* b, int64_t ldb, fp16* to);
};
void pack_convert_a(int m, int k, float alpha, float const* a, int64_t lda, fp16* to);
void pack_convert_b(int n, int k, float alpha, float const* b, int64_t ldb, fp16* to);
#endif
