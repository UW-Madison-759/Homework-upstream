#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <array>
#include <algorithm>
#include "stopwatch.hpp"
#include <omp.h>

template <typename T>
auto read_image(char const* filename) {
	std::ifstream fin { filename };

	/*
	 * 	See the iterator pair interface for std::vector in the docs
	 * 	http://en.cppreference.com/w/cpp/container/vector/vector
	 *
	 * 	Recall that the second operand '{}' default-constructs a
	 * 	std::istream_iterator<T> that tells the std::vector constructor
	 * 	to read until it reaches the EOF marker
	 */
	return std::vector<T> { std::istream_iterator<T> { fin }, { } };
}

int main(int argc, char* argv[]) {
	if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " N\n";
		return -1;
	}
	const auto num_threads = atoi(argv[1]);
	omp_set_num_threads(num_threads);

	using hist_val_t = size_t;
	auto image = read_image<hist_val_t>("picture.inp");

	std::array<hist_val_t, 7> histogram;
	stopwatch<std::milli, float> sw;
	std::vector<float> timings;

	// Do 10 iterations to remove jitter
	// NOTE: These timings do _not_ include the startup overhead of the team of threads
	for (int i = 0; i < 10; i++) {
		// reset the histogram
		histogram.fill(0);

		/*
		 * 	An alternative to this method is to use an array section as
		 * 	the reduction target. This would work like so
		 *
		 * 		int hist[7] = {0};
		 * 		#pragma omp parallel for reduction(+:hist[:7])
		 * 		for (auto j = 0UL; j < image.size(); j++) {
		 * 			hist[image[j]]++;
		 * 		}
		 *
		 * 	However, this does not seem to be working on Euler at the moment.
		 */
		#pragma omp parallel
		{
			std::array<hist_val_t, 7> local_hist;
			local_hist.fill(0);

			#pragma omp single
			sw.start();

			#pragma omp for
			for (auto j = 0UL; j < image.size(); j++) {
				local_hist[image[j]]++;
			}

			#pragma omp critical
			for (auto i = 0UL; i < histogram.size(); i++) {
				histogram[i] += local_hist[i];
			}

			#pragma omp single
			{
				sw.stop();
				timings.push_back(sw.count());
			}
		}
	}

	/*
	 * 	The homework says to put each item on a separate line. However,
	 * 	the grading system ignores all whitespace, and having four items
	 * 	per line makes it easier to handle in Python, so that's what we
	 * 	do here.
	 */
	for (auto x : histogram)
		std::cout << x << ' ';

	std::cout << num_threads << ' '
			  << *std::min_element(timings.begin(), timings.end()) << '\n';
}
