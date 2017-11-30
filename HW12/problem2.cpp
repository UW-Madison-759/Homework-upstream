#include <iostream>
#include <vector>
#include <fstream>
#include <numeric>
#include <iterator>
#include <algorithm>
#include "problem2.hu"

template <typename T>
auto read_file(char const* name, int size) {
	std::ifstream fin{name};
	std::vector<T> x;
	x.reserve(static_cast<size_t>(size));
	std::copy_n(std::istream_iterator<T>(fin), size, std::back_inserter(x));
	return x;
}

int main(int argc, char const* argv[]) {
	if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " N\n";
		return -1;
	}

	const auto N = std::atoi(argv[1]);
	auto x = read_file<float>("problem2.inp", N);

	std::vector<float> sum(static_cast<size_t>(N));
	prefix_scan(x.data(), sum.data(), N);
}
