#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include "stopwatch.hpp"

struct point3D {
	float x, y, z;
};

struct point3D_c {
	float x, y, z;
	char c;
};

struct point3D_ca {
	float x, y, z;
	char c[52];
};

/*
 * Use a template function to handle the three different
 * versions of the point3D struct.
 */
template <typename T, typename UnaryFunc>
void time_it(size_t size, UnaryFunc min_time) {
	stopwatch<std::milli, float> sw;
	std::vector<T> x(size);
	volatile float sum{};
	auto f = [&x, &sum]() {
		sum = std::accumulate(x.begin(), x.end(), 0.0f, [](float total, T const& p){ return total + p.x;});
	};
	sw.time_it(10UL, f, min_time);
}

int main() {
	std::ofstream fout{"problem2.out"};

	/*
	 * A lambda to store the minimum time in the output file.
	 * 'array_view<T>' is defined in 'stopwatch.hpp'. It is
	 * just a simple array-like container that holds the
	 * timing values returned by the stopwatch::time_it function.
	 */
	auto min_time = [&fout](array_view<float> const& x) {
		fout << *std::min_element(x.begin(), x.end()) << '\n';
	};

	constexpr auto size = 1'000'000UL;
	std::cout << "sizeof(point3D) = " << sizeof(point3D) << '\n';
	time_it<point3D>(size, min_time);

	std::cout << "sizeof(point3D_c) = " << sizeof(point3D_c) << '\n';
	time_it<point3D_c>(size, min_time);

	std::cout << "sizeof(point3D_ca) = " << sizeof(point3D_ca) << '\n';
	time_it<point3D_ca>(size, min_time);
}
