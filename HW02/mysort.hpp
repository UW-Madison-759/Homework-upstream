#pragma once

template<typename Container>
void my_sort(Container &c) {
	auto n = c.size();
	bool swapped = false;

	do {
		swapped = false;
		for (auto i = 1UL; i < n; i++) {
			if (c[i - 1] > c[i]) {
				std::swap(c[i - 1], c[i]);
				swapped = true;
			}
		}
		n--;
	} while (swapped);
}
