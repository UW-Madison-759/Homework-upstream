#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <omp.h>
#include "stopwatch.hpp"

template<typename T>
auto read_matrix(char const* filename) {
	std::ifstream fin { filename };
	return std::vector<T>(std::istream_iterator<T>(fin), {});
}

int main(int argc, char *argv[]) {
	if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " N\n";
		return -1;
	}

	const auto num_threads = std::atoi(argv[1]);
	omp_set_num_threads(num_threads);

	using matrix_t = double;
	auto A = read_matrix<matrix_t>("inputA.inp");
	auto B = read_matrix<matrix_t>("inputB.inp");

	constexpr auto len = 1024UL;
	stopwatch<std::milli, float> sw;

	std::vector<matrix_t> C(len * len);
	std::fill(C.begin(), C.end(), matrix_t{});

	sw.start();
#pragma omp parallel for
	for (auto i = 0UL; i < len; i++) {
		for (auto k = 0UL; k < len; k++) {
			for (auto j = 0UL; j < len; j++) {
				C[i * len + j] += A[i * len + k] * B[k * len + j];
			}
		}
	}
	sw.stop();
	std::cout << C.back() << ' ' << sw.count() << ' ' << num_threads << '\n';
}
