/*
* Compute reference result for C = al*A + be*B operation
*/

#include <stdlib.h>
////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void computeGold( float*, const float*, const float, const float*, const float, unsigned int, unsigned int);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! C = al*A + be*B
//! @param C          reference data, computed herein but preallocated somewhere else
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void
computeGold(float* C, const float* A, const float al, const float* B, const float be, unsigned int hA, unsigned int wA)
{
   for (unsigned int i = 0; i < hA*wA; ++i){
      C[i] = (float)(al*A[i] + be*B[i]);
   }
}
