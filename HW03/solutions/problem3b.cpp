#include <algorithm>
#include <functional>

int main() {
	constexpr auto nrows = 96UL, ncols = 209UL;

	// This is the C++11 way of writing a typedef
	using matrix = float[nrows][ncols];

	/*
	 * 	In problem3a, you should have found that 2D arrays in C or C++
	 * 	are stored sequentially. This means that we can "flatten" a matrix
	 * 	(represented by a 2D array) into a 1D array by taking a pointer
	 * 	to the first element of the matrix.
	 */
	matrix A, B, C;
	auto *pA = &A[0][0],
		 *pB = &B[0][0],
		 *pC = &C[0][0];

	// The size is now the total number of elements in the matrix
	constexpr auto size = nrows * ncols;

	/*
	 *  std::transform is the C++ way of writing
	 *
	 *  for(int i=0; i<size; i++)
	 *  	pC[i] = pA[i] + pB[i];
	 */
	std::transform(pA, pA + size, pB, pC, std::plus<>());

	/*
	 * The question is then why this version is vectorizable and the "traditional"
	 * double-nested loop version isn't.
	 *
	 *		for(int i=0; i<nrows; i++)
	 *			for(int j=0; j<ncols; j++)
	 *				C[i][j] = A[i][j] + B[i][j];
	 *
	 *
	 * It turns out that modern compilers like gcc and clang _will_ vectorize the
	 * nested loop version. However, they have to put forth a lot of effort to do it.
	 * For the purposes of this homework, though, we are dealing with hypothetical
	 * dumb compiler that doesn't know the tricks gcc and clang can use.
	 *
	 * Let's look at the calculation for just the first row (i.e., i=0). We have
	 *
	 *		for(int j=0; j<ncols; j++)
	 *			C[0][j] = A[0][j] + B[0][j];
	 *
	 *	Each row contains ncols*sizeof(float)=209*4=836 bytes. Our vector width is 8 bytes.
	 *	Therefore, we would need 836/8 = 104.5 vector additions to do the first row. But how
	 *	does the CPU work with half a vector? Well, it can't. In this situation, our dumb
	 *	compiler gives up and won't vectorize the loop.
	 *
	 *	Now let's look at the 1D version.
	 *
	 *	 for(int i=0; i<size; i++)
	 *  	pC[i] = pA[i] + pB[i];
	 *
	 *  Now we have ncols*nrows*sizeof(float)=96*209*4=80256 bytes that are _sequential_. This
	 *  is the key that let's our dumb compiler vectorize the loop. Now we have 80256/8 = 10032
	 *  vector additions to add the two matrices.
	 *
	 *  The general idea here is that a matrix is an abstract type that is actually implemented
	 *  as a 1D sequence in memory. As programmers, we can exploit this fact to get the compiler
	 *  to generate better code for us.
	 */
}
