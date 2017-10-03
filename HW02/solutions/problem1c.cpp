#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include "stopwatch.hpp"
#include "mysort.hpp"

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

	std::ofstream fout { "problem1c.out" };

	for (int i = 10; i < 20; i++) {
		auto numbers = generate(1 << i);

		/*
		 * 	Because all sorts are done inplace, we need a copy of
		 * 	the numbers so that each sort sees the same values to
		 * 	be sorted.
		 */
		auto numbers_copy{numbers};

		sw.start();
		my_sort(numbers);
		sw.stop();
		const auto my_sort = sw.count();

		sw.start();
		std::sort(numbers_copy.begin(), numbers_copy.end());
		sw.stop();
		const auto std_sort = sw.count();

		fout << numbers.size() << ' ' << my_sort << ' ' << std_sort << ' ' << '\n';
	}
}
