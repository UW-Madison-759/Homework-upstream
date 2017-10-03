#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iterator>
#include "stopwatch.hpp"

template<typename Container>
auto exclusive_scan(Container const& c) {
	std::vector<typename Container::value_type> x;
	x.reserve(c.size());

	// Calculate the inclusive scan.
	std::partial_sum(std::begin(c), std::end(c), std::back_inserter(x));

	// Convert to exclusive scan by taking the N-1 element
	// Note: end() is one past the last element
	return *(x.end() - 2);
}

auto read_data(char const* filename) {
	std::ifstream fin { filename };
	int dummy{};
	fin >> dummy;
	std::vector<int> x;
	x.reserve(static_cast<size_t>(dummy));
	std::copy(std::istream_iterator<int>(fin), {}, std::back_inserter(x));
	return x;
}

int main(int argc, char *argv[]) {
	if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " file\n";
		return -1;
	}

	auto numbers = read_data(argv[1]);

	stopwatch<std::milli, float> sw;
	sw.start();
	const auto last = exclusive_scan(numbers);
	sw.stop();

	std::cout << numbers.size() << ' ' << last << ' ' << sw.count() << '\n';
}
