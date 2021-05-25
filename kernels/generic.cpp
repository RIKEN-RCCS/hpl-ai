#include <omp.h>
#include "kernel.h"

#ifdef HGEMM_PACK
void pack_convert_a(int m, int k, float alpha, float const* a, int64_t lda, fp16* to)
{
	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		int nt = omp_get_num_threads();
		int mm = (m+nt-1)/nt;
		mm = (mm+HGEMM_PACK_MUNIT-1)/HGEMM_PACK_MUNIT*HGEMM_PACK_MUNIT;
		int mbegin = mm * id;
		int mend = mbegin + mm;
		if(mend > m) mend = m;
		if(mend>mbegin) pack_convert_a_opt(mend-mbegin, k, alpha, a+mbegin, lda, to+(int64_t)mbegin*k);
	}
}
void pack_convert_b(int n, int k, float alpha, float const* b, int64_t ldb, fp16* work)
{
	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		int nt = omp_get_num_threads();
		int nn = (n+nt-1)/nt;
		nn = (nn+HGEMM_PACK_NUNIT-1)/HGEMM_PACK_NUNIT*HGEMM_PACK_NUNIT;
		int nbegin = nn * id;
		int nend = nbegin + nn < n ? nbegin + nn: n;
		int nr = nend % HGEMM_PACK_NUNIT;
		int mynb = (nend-nbegin);
		if(mynb>0) pack_convert_b_opt(mynb, k, alpha, b+nbegin*ldb, ldb, work+(int64_t)nbegin*k);
		if(nr){
			int64_t nlast = nend - nr;
			pack_convert_b_small(nr, k, alpha, b+nlast*ldb, ldb, work+nlast*k);
		}
	}
}
#endif
