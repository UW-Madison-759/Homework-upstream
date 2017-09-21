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

	auto min_time = [&fout](array_view<float> const& x) {
		fout << *std::min_element(x.begin(), x.end()) << '\n';
	};

	constexpr auto size = 1'000'000UL;

	stopwatch<std::milli, float> sw;
	std::vector<point3D> x(size);

	/*
	 * Note that this is really just a dummy variable we are using
	 * to prevent the compiler from optimizing away our calculations.
	 */
	volatile float sum{};
	auto f = [&x, &sum]() {
		sum = std::accumulate(x.begin(), x.end(), 0.0f, [](float total, point3D const& p){ return total + p.x;});
	};
	sw.time_it(10UL, f, min_time);
}
