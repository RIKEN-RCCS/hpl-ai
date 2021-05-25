#ifndef HPL_RAND_HPP
#define HPL_RAND_HPP

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

struct RandCoeff {
	uint64_t a;
	uint64_t c;

	static RandCoeff default_vals(){
		return {6364136223846793005, 1};
	}

	RandCoeff operator*(const RandCoeff &rhs) const {
		return {a * rhs.a, a * rhs.c + c};
	}

	RandCoeff pow(const uint64_t n) const {
		if (0 == n){
			return RandCoeff{1, 0};
		} else 	{
			RandCoeff tmp = pow(n/2);
			tmp = tmp * tmp;
			if(n%2){
				return  tmp * (*this);
			}else{
				return tmp;
			}
		}
	}
};

struct RandStat{
	uint64_t x;

	static RandStat initialize(
			uint64_t  seed, 
			RandCoeff coef = RandCoeff::default_vals())
	{
		return coef * RandStat{seed};
	}

	friend RandStat operator*(RandCoeff coef, RandStat stat){
		return { coef.a * stat.x + coef.c };
	}

	// returns [-0.5:0.5]
	operator double() const {
		return static_cast<int64_t>(x) * 0x1.fffffffffffffP-65;
	}
	operator float() const {
		float tmp = static_cast<double>(*this);
		return tmp;
	}
};

// fill subumat (i0:nrow-1,  j0:ncol-1) of fullmat (0:nrow-1, 0:ncol-1)
template<typename F>
static void fill_one_panel_with_rand(
		int const       n,
		int const       i0,
		int const       j0,
		int const       nrow,
		int const       ncol,
		F              *a,
		size_t const    lda,
		uint64_t const  seed,
		bool const      calc_diag = true)
{
	RandStat stat_00 = RandStat::initialize(seed);

	RandCoeff inc1 = RandCoeff::default_vals();
	RandCoeff jump_one_col = inc1.pow(n);
	RandCoeff jump_ij = inc1.pow(i0 + n * static_cast<uint64_t>(j0));

	RandStat stat_ij = jump_ij * stat_00;

	RandStat at_0j = stat_ij;
	for(int j=0; j<ncol; j++){
		RandStat at_ij = at_0j;
		for(int i=0; i<nrow; i++){
			double t = static_cast<double>(at_ij);
			a[j*lda + i] = static_cast<F>(t);
			at_ij = inc1 * at_ij;
		}
		at_0j = jump_one_col * at_0j;
	}
	if(calc_diag && (i0 == j0) && (nrow==ncol)){
		RandCoeff jump_i0 = inc1.pow(i0);

		RandStat stat_i0 = jump_i0 * stat_00;
		for(int i=0; i<(nrow<ncol?nrow:ncol); i++){
			RandStat stat_ij = stat_i0;
			double sum = 0.0;
			for(int j=0; j<n; j++){
				if(i0+i!=j) 
					sum += fabs(double(stat_ij));
				stat_ij = jump_one_col * stat_ij;
			}
			a[lda*i + i] = static_cast<F>(sum);
			stat_i0 = inc1 * stat_i0;
		}
	}
}

static inline 
double calc_diag(int const i, int const n, uint64_t const seed){
	RandStat stat_00 = RandStat::initialize(seed);
	RandCoeff inc1 = RandCoeff::default_vals();
	RandCoeff jump_one_col = inc1.pow(n);
	RandCoeff jump_i0 = inc1.pow(i);
	RandStat stat_ij = jump_i0 * stat_00;

	double sum = 0.0;
	for(int j=0; j<n; j++){
		if(i!=j) 
			sum += fabs(double(stat_ij));
		stat_ij = jump_one_col * stat_ij;
	}
	return sum;
}

// debug
static inline 
double mat_elem(int n, int i, int j, int seed){
	RandStat stat_00 = RandStat::initialize(seed);
	RandCoeff inc1 = RandCoeff::default_vals();
	RandCoeff jump_ij = inc1.pow(i + static_cast<uint64_t>(n) * j);
	return double(jump_ij * stat_00);
}

template<typename F>
struct Matgen{
	uint64_t seed;
	int n;
	F const* diag;
	RandCoeff incl1, jumpn, jumpi, jumpj;

	enum {NUM_POWERS = 16};
	RandCoeff powers[NUM_POWERS];

	double scalea, scaleb;

	Matgen(uint64_t seed, int n, int iskip, int jskip, F const* diag): seed(seed), n(n), diag(diag) {
		incl1 = RandCoeff::default_vals();
		jumpn = incl1.pow(n);
		jumpi = incl1.pow(iskip);
		jumpj = incl1.pow(n * static_cast<uint64_t>(jskip));
		for(int i=0; i<NUM_POWERS; i++){
			powers[i] = incl1.pow(i);
		}
		scalea = sqrt(n*sqrt(n));
		scaleb = 1;
	}
	RandCoeff jump(int i, int j) const {
		return incl1.pow(i + n * static_cast<uint64_t>(j));
	}
};

#endif
