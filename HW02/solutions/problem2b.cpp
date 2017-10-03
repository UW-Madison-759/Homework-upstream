#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
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

auto generate(int N) {
	std::vector<int> numbers;
	numbers.reserve(static_cast<size_t>(N));
	std::mt19937 dev { std::random_device{}() };
	std::uniform_int_distribution<int> gen(-N, N);
	std::generate_n(std::back_inserter(numbers), N, [&gen,&dev]() {return gen(dev);});
	return numbers;
}

int main() {
	stopwatch<std::milli, float> sw;

	std::ofstream fout { "problem2b.out" };

	for (int i = 10; i < 20; i++) {
		auto numbers = generate(1 << i);

		sw.start();
		const auto last = exclusive_scan(numbers);
		sw.stop();
		fout << numbers.size() << ' ' << sw.count() << ' ' << last << '\n';
	}
}
