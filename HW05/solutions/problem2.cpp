#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <vector>
#include "stopwatch.hpp"
#include "math.hpp"
#include <omp.h>

/*
 *
 * 	See the Intel Intrinsics Guide for details on each intrinisc function
 * 	https://software.intel.com/sites/landingpage/IntrinsicsGuide/
 *
 */

double f(double x) {
	return std::exp(std::sin(x)) * std::cos(x / 40.0);
}
__m256d f(__m256d x) {
	auto div = _mm256_div_pd(x, _mm256_set1_pd(40.0));
	auto s = gromacs::sin(x);
	auto c = gromacs::cos(div);
	auto e = gromacs::exp(s);
	return _mm256_mul_pd(e, c);
}

/*
 *  Add up the elements of the vector.
 *
 *  NOTE: This could also be accomplished via the use the horizontal add intrinsic _mm256_hadd_pd,
 *  but that would require more code. The compiler gives us "array-like" access to the individual
 *  elements of the vector and then translates this to vector instructions for us.
 */
double horizontal_sum(__m256d x) {
	return x[0] + x[1] + x[2] + x[3];
}

int main(int argc, char *argv[]) {
	if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " N\n";
		return -1;
	}

	const auto num_threads = std::atoi(argv[1]);
	omp_set_num_threads(num_threads);

	constexpr auto N = 1000000UL;
	constexpr auto h = 100.0 / N;
	constexpr auto num_iters = 1UL;

	std::vector<float> timings;
	timings.reserve(num_iters);

	stopwatch<std::milli, float> sw;

	__m256d sum = _mm256_setzero_pd();
	double integral_value { };
	constexpr auto vec_width = sizeof(__m256d) / sizeof(double);

	// Do it a few times to get rid of timing jitter
	for (size_t m = 0; m < num_iters; ++m) {
		sw.start();

		// Start by packing the coefficients of the four "leading" terms in Simpson's rule
		auto coeffs = _mm256_set_pd(17.0, 59.0, 43.0, 49.0);

		// Make a vector of the corresponding x values: x = N*h for N={0,1,2,3}
		auto x = _mm256_mul_pd(_mm256_set_pd(0.0, 1.0, 2.0, 3.0), _mm256_set1_pd(h));

		// Multiply the coefficients by the value of 'f' for each x (note: this is a _packed_ instruction)
		const auto left = _mm256_mul_pd(coeffs, f(x));

		// Do the same again for the "trailing" four terms in Simpson's rule
		coeffs = _mm256_set_pd(49.0, 43.0, 59.0, 17.0);
		const auto tmp = _mm256_sub_pd(_mm256_set1_pd(static_cast<double>(N)), _mm256_set_pd(3.0, 2.0, 1.0, 0.0));
		x = _mm256_mul_pd(tmp, _mm256_set1_pd(h));
		const auto right = _mm256_mul_pd(coeffs, f(x));

		sum = _mm256_setzero_pd();
		integral_value = 0.0;

		/*
		 * In C and C++, you can multiply together a double and an int to get a double like so
		 *
		 * 		double x = 1.5; int i = 3; double y = x * i;
		 *
		 * 	and it "just works." Because we are using intrinsics, we have to do the manual
		 * 	int->double conversion ourselves. It really makes you appreciate having even a
		 * 	rudimentary type system like that in C!
		 */
		const auto fi_offset = _mm256_cvtepi32_pd(_mm_set_epi32(0, 1, 2, 3));

		/*
		 * 	This is a user-defined reduction operation. For details, see
		 * 	section 2.16 "declare reduction Directive" of the OpenMP specification
		 * 	http://www.openmp.org/wp-content/uploads/openmp-4.5.pdf
		 */
		#pragma omp declare reduction(mm256d_sum : __m256d : omp_out = _mm256_add_pd(omp_out,omp_in)) initializer(omp_priv = _mm256_setzero_pd())

		// How many vector elements are processed by each iteration?
		// This is needed for doing loop unrolling correctly
		constexpr auto vec_inc = 1;

		// How many iterations do we need to cover the half-open interval [4, N - 4)?
		constexpr auto num_iters = (N - 4 - 4) / (vec_inc * vec_width);

		// Here is the main work loop that actually calculates the integral.
		#pragma omp parallel for reduction(mm256d_sum : sum)
		for (auto i = 4UL; i <= num_iters * vec_inc * vec_width; i += vec_inc * vec_width) {
			const auto fi = _mm256_cvtepi32_pd(_mm_set1_epi32(i));
			const auto t = _mm256_add_pd(fi, fi_offset);
			const auto indices = _mm256_mul_pd(_mm256_set1_pd(h), t);
			sum = _mm256_add_pd(sum, f(indices));
		}

		/*
		 * 	Now we do a scalar (i.e., not vectorized) sum of the terms
		 * 	that were not covered by the vectors
		 */
		double extra_sum {};
		for (auto i = num_iters * vec_inc * vec_width + 4; i <= N - 4; i++) {
			extra_sum += f(h * i);
		}

		integral_value = h / 48.0 * (horizontal_sum(left) + horizontal_sum(right)) + horizontal_sum(_mm256_mul_pd(_mm256_set1_pd(h), sum)) + extra_sum * h;

		sw.stop();
		timings.push_back(sw.count());
	}

	const auto min_time = *std::min_element(timings.begin(), timings.end());
	std::cout << num_threads << " " << min_time << " " << std::setprecision(15) << integral_value << "\n";

	// Calculate the percent error
	auto perror = [](auto x) {return 100.0 * std::fabs(x-32.121040688226245) / 32.121040688226245;};

	// For debugging only. Don't include this in your final results
	std::cout << "% error = " << perror(integral_value) << '\n';
}
