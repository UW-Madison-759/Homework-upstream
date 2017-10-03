#include <fstream>
#include <vector>
#include <list>
#include <algorithm>
#include "stopwatch.hpp"
#include <functional>

template<typename Container, typename UnaryFunc>
float time_it(size_t count, Container container, UnaryFunc f) {
	std::vector<float> timings;
	stopwatch<std::milli, float> sw;

	// Do 10 iterations to get rid of jitter
	for (size_t iter = 0; iter < 10; iter++) {
		sw.start();
		for (size_t i = 0; i < count; i++) {
			f(i);
		}
		sw.stop();
		timings.push_back(sw.count());

		// Reset the input
		container.clear();
	}
	return *std::min_element(timings.begin(), timings.end());
}

int main() {
	// The number of integers in each test
	constexpr static size_t sizes[] = { 1<<8, 1<<9, 1<<10, 1<<11, 1<<12, 1<<13};

	{
		std::ofstream fout { "problem1_array.out" };
		std::vector<int> data;

		/*
		 *  This is a collection of function objects. This is constructed
		 *  so that we can iterate over them just like they were "normal"
		 *  values in a container (see below)
		 */
		auto funcs = std::initializer_list<std::function<void(int)>> {
			[&data](int n) {
				// Insert at the front
				data.insert(data.begin(), n);
			},
			[&data](int n) {
				// Insert in the middle
				auto pos = data.begin();
				std::advance(pos, data.size() / 2UL);
				data.insert(pos, n);
			},
			[&data](int n) {
				// Insert at the end
				data.push_back(n);
			}
		};

		// For each function in 'funcs'
		for (auto f : funcs) {
			// for each size in 'sizes'
			for (auto size : sizes) {
				// Time the insertion of size elements into data at the position specifed in 'f'.
				fout << size << " " << time_it(size, data, f) << '\n';
			}
		}
	}

	{
		std::ofstream fout { "problem1_list.out" };
		std::list<int> data;
		auto funcs = std::initializer_list<std::function<void(int)>> {
			[&data](int n) {
				// Note that a std::list has a 'push_front', but a std::vector does not
				data.push_front(n);
			},
			[&data](int n) {
				auto pos = data.begin();
				std::advance(pos, data.size() / 2UL);
				data.insert(pos, n);
			},
			[&data](int n) {
				data.push_back(n);
			}
		};

		for (auto f : funcs) {
			for (auto size : sizes) {
				fout << size << " " << time_it(size, data, f) << '\n';
			}
		}
	}
}
