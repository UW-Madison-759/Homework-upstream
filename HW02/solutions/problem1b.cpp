#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <random>
#include "stopwatch.hpp"

int main(int argc, char *argv[]) {
	if (argc > 2) {
		std::cerr << "Usage: " << argv[0] << "[N]\n";
		return -1;
	}

	std::vector<int> numbers;

	// No number specified
	if (argc == 1) {
		std::ifstream fin { "problem1.in" };

		/*
		 * In C++, we don't need to know how many values are in the
		 * file, so we would normally just ignore this value. However,
		 * we can use it as an optimization by using std::vector::reserve
		 * so that we only do one memory allocation- thereby speeding up
		 * the read process (because 'numbers' won't have to reallocate
		 * as it grows).
		 */
		int dummy;
		fin >> dummy;
		numbers.reserve(static_cast<size_t>(dummy));

		/*
		 * Read the rest of the numbers from the file
		 *
		 * Note: The second argument '{}' invokes the default constructor
		 * 		 for 'std::istream_iterator<int>' which tells 'std::copy'
		 * 		 to read from the file until it reaches the EOF marker.
		 *
		 * 		 The documentation for std::back_inserter is below
		 * 		 http://en.cppreference.com/w/cpp/iterator/back_inserter
		 */
		std::copy(std::istream_iterator<int>(fin), { }, std::back_inserter(numbers));
	} else {
		// Generate N numbers randomly
		const auto N = std::atoi(argv[1]);

		// apply the same optimization discussed above for the argc == 1 case
		numbers.reserve(static_cast<size_t>(N));

		/*
		 * C++11 introduced substantially enhanced facilities for generating
		 * pseudo-random numbers (http://en.cppreference.com/w/cpp/numeric/random).
		 *
		 * 	std::mt19937 -	this is a version of the Mersenne-Twister algorithm
		 * 				   	http://en.cppreference.com/w/cpp/numeric/random/mersenne_twister_engine
		 *
		 * 	std::uniform_int_distributino<int> gen(1,N) - a uniform distribution of integers between 1 and N.
		 * 					http://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
		 *
		 * 	std::generate_n(C, N, G) - Insert N values into C using the generator G
		 * 					http://en.cppreference.com/w/cpp/algorithm/generate_n
		 */
		std::mt19937 dev { std::random_device()() };
		std::uniform_int_distribution<int> gen(-N, N);
		std::generate_n(std::back_inserter(numbers), N, [&gen,&dev]() {return gen(dev);});
	}

	stopwatch<std::milli, float> sw;
	sw.start();
	std::sort(numbers.begin(), numbers.end());
	sw.stop();

	std::ofstream fout { "problem1.out" };
	for (auto x : numbers)
		fout << x << '\n';

	std::cout << numbers.size() << ' ' << sw.count() << '\n';
}

