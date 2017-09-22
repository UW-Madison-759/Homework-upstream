#include <fstream>
#include <iostream>
#include <string>

int main() {
	std::ifstream fin{"problem1.txt"};
	std::string id;
	fin >> id;
	std::cout << "Hello! I'm student " << id.substr(id.length() - 4, id.length()) << ".\n";
}
