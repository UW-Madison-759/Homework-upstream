#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <cmath>
#include <iomanip>
#include <vector>
#include <algorithm>
#include "stopwatch.hpp"

struct integration_bound { double lb, ub; };

template <typename UnaryFunc>
double simpson(UnaryFunc f, const integration_bound bnds, const double h) {
	const auto N = static_cast<int>((bnds.ub-bnds.lb)/h);

	// The "leading" terms
	const auto leading = 17.0 * f(0.0) + 59.0 * f(h) + 43.0 * f(2.0 * h) + 49.0 * f(3.0 * h);

	// The "trailing" terms
	const auto trailing = 49.0 * f(h * (N - 3)) + 43.0 * f(h * (N - 2)) + 59.0 * f(h * (N - 1)) + 17.0 * f(h * N);

	double sum{};

	#pragma omp parallel for reduction(+:sum)
	for (int i = 4; i < N - 4; i++) {
		sum += f(i * h);
	}

	// The summation scaling factor
	sum *= 48.0;

	// Add the leading and trailing terms
	sum += leading + trailing;

	// And the final prefactor
	sum *= h / 48.0;

	return sum;
}

int main(int argc, char *argv[]) {
	/*
	 * 	The problem statement has only one argument, but passing the name via
	 * 	the command line is a more portable way of handling this. This program
	 * 	would then be run from the terminal or an sbatch script like so
	 *
	 * 		./problem1 1 $(uname -n)
	 *
	 * 	For now, we just have a placeholder for the server name.
	 */
	if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " N\n";
		return -1;
	}

	const auto num_threads = std::atoi(argv[1]);
	omp_set_num_threads(num_threads);

	// Dummy placeholder
	const auto server_name = "foo";

	auto f = [](auto x) {
		return std::exp(std::sin(x)) * std::cos(x / 40.0);
	};

	stopwatch<std::milli, float> sw;
	std::vector<float> timings;

	// Do 10 iterations to remove jitter
	// NOTE: These timings will include the startup overhead of the team of threads
	double sum{};
	for (int i=0; i<10; i++) {
		sw.start();
		sum = simpson(f, {0.0, 100.0}, 1e-4);
		sw.stop();
		timings.push_back(sw.count());
	}

	/*
	 * 	The homework says to put each item on a separate line. However,
	 * 	the grading system ignores all whitespace, and having four items
	 * 	per line makes it easier to handle in Python, so that's what we
	 * 	do here.
	 */
	std::cout << num_threads << ' '
			  << *std::min_element(timings.begin(), timings.end()) << ' '
			  << server_name << ' '
			  << std::fixed << std::setprecision(12) << sum << '\n';
}
