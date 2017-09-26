#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include "stopwatch.hpp"

struct point3D {
	float x, y, z;
};

int main() {
	std::ofstream fout{"problem2.out"};

	/*
	 * This a lambda (or anonymous function) that prints the minimum timing value to the file.
	 *
	 * 	[&fout] - this captures the output file stream (by reference) so we can write to it
	 * 	(array_view<float> const& x) - this is the parameter list. This is just like a parameter
	 * 								   for any function. The array_view type is defined in stopwatch.hpp.
	 * 								   It contains the timing values returned by the time_it function (see below).
	 */
	auto min_time = [&fout](array_view<float> const& x) {
		fout << *std::min_element(x.begin(), x.end()) << '\n';
	};

	constexpr auto size = 1'000'000UL;

	stopwatch<std::milli, float> sw;
	std::vector<point3D> x(size);

	/*
	 * Note that this is really just a dummy variable we are using
	 * to prevent the compiler from optimizing away our calculations.
	 *
	 * At high levels of optimization (e.g., -O3), the compiler would be able to tell
	 * that we aren't actually doing anything with 'sum' and would throw away all of
	 * our work- giving us timings of zero!
	 */
	volatile float sum{};

	/*
	 * This is another lambda that calculates the sum of the x values of each Point3D
	 *
	 * 	&x - This captures the vector (by reference) so we can calculate with its values
	 * 	&sum - This captures the sum (by reference) so we can write to it directly.
	 */
	auto f = [&x, &sum]() {
		/*
		 * See the documentation for std::accumulate
		 * http://en.cppreference.com/w/cpp/algorithm/accumulate
		 */
		sum = std::accumulate(x.begin(), x.end(), 0.0f,
				/*
				 * This is yet another lambda.
				 * It gets called once for each element of the vector x
				 *
				 * The first parameter is the running total (or sum)
				 * The second parameter is the current element of x (a point3D object)
				 *
				 * It adds the current x value to the running total and returns
				 * that part of the sum.
				 */
				[](float total, point3D const& p){ return total + p.x;});

		/*
		 * This is identical to using
		 *
		 * float sum = 0.0;
		 * for (size_t i=0; i<x.size(); i++)
		 *     sum += x[i].x;
		 */
	};

	/*
	 * The time_it function will execute the function 'f' 10 times and then call 'min_time'
	 * with the 10 timing results wrapped in a array_view<float>.
	 */
	sw.time_it(10UL, f, min_time);
}
