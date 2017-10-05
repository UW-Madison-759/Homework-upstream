/*
 * This code was adapted from the GROMACS molecular simulation package.
 * Modified by Tim Haines, November 2015
 *
 * Copyright (c) 2014,2015, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 */

#ifndef GROMACS_MATH_HPP
#define GROMACS_MATH_HPP

#include <immintrin.h>

namespace gromacs {

constexpr double GMX_DOUBLE_NEGZERO = -0.0;
constexpr double PI = 3.14159265358979323846;

inline __m256d fmadd_d(__m256d a, __m256d b, __m256d c){
	return _mm256_add_pd(_mm256_mul_pd(a, b), c);
}

inline __m256d fmsub_d(__m256d a, __m256d b, __m256d c){
	return _mm256_sub_pd(_mm256_mul_pd(a, b), c);
}

inline __m256d gmx_simd_cvt_dib2db_avx_256(__m128i a) {
    __m128i a1 = _mm_shuffle_epi32(a, _MM_SHUFFLE(3, 3, 2, 2));
    __m128i a0 = _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 1, 0, 0));
    return _mm256_castsi256_pd(_mm256_insertf128_si256(_mm256_castsi128_si256(a0), a1, 0x1));
}

inline __m256d set_exponent(__m256d x) {
	const __m128i expbias = _mm_set1_epi32(1023);
	__m128i iexp128a, iexp128b;

	iexp128a = _mm256_cvtpd_epi32(x);
	iexp128a = _mm_add_epi32(iexp128a, expbias);
	iexp128b = _mm_shuffle_epi32(iexp128a, _MM_SHUFFLE(3, 3, 2, 2));
	iexp128a = _mm_shuffle_epi32(iexp128a, _MM_SHUFFLE(1, 1, 0, 0));
	iexp128b = _mm_slli_epi64(iexp128b, 52);
	iexp128a = _mm_slli_epi64(iexp128a, 52);
	return _mm256_castsi256_pd(_mm256_insertf128_si256(_mm256_castsi128_si256(iexp128a), iexp128b, 0x1));
}

inline __m256d exp(__m256d x) {
	const auto argscale = _mm256_set1_pd(1.44269504088896340735992468100);
	const auto arglimit = _mm256_set1_pd(1022.0);
	const auto invargscale0 = _mm256_set1_pd(-0.69314718055966295651160180568695068359375);
	const auto invargscale1 = _mm256_set1_pd(-2.8235290563031577122588448175013436025525412068e-13);
	const auto CE12 = _mm256_set1_pd(2.078375306791423699350304e-09);
	const auto CE11 = _mm256_set1_pd(2.518173854179933105218635e-08);
	const auto CE10 = _mm256_set1_pd(2.755842049600488770111608e-07);
	const auto CE9 = _mm256_set1_pd(2.755691815216689746619849e-06);
	const auto CE8 = _mm256_set1_pd(2.480158383706245033920920e-05);
	const auto CE7 = _mm256_set1_pd(0.0001984127043518048611841321);
	const auto CE6 = _mm256_set1_pd(0.001388888889360258341755930);
	const auto CE5 = _mm256_set1_pd(0.008333333332907368102819109);
	const auto CE4 = _mm256_set1_pd(0.04166666666663836745814631);
	const auto CE3 = _mm256_set1_pd(0.1666666666666796929434570);
	const auto CE2 = _mm256_set1_pd(0.5);
	const auto one = _mm256_set1_pd(1.0);
	__m256d fexppart;
	__m256d intpart;
	__m256d y, p;
	__m256d valuemask;

	y = _mm256_mul_pd(x, argscale);
	fexppart = set_exponent(y); /* rounds to nearest int internally */
	intpart = _mm256_round_pd(y, _MM_FROUND_NINT); /* use same rounding mode here */
	valuemask = _mm256_cmp_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.0),y), arglimit, _CMP_LE_OQ);
	fexppart = _mm256_and_pd(fexppart, valuemask);

	/* Extended precision arithmetics */
	x = fmadd_d(invargscale0, intpart, x);
	x = fmadd_d(invargscale1, intpart, x);

	p = fmadd_d(CE12, x, CE11);
	p = fmadd_d(p, x, CE10);
	p = fmadd_d(p, x, CE9);
	p = fmadd_d(p, x, CE8);
	p = fmadd_d(p, x, CE7);
	p = fmadd_d(p, x, CE6);
	p = fmadd_d(p, x, CE5);
	p = fmadd_d(p, x, CE4);
	p = fmadd_d(p, x, CE3);
	p = fmadd_d(p, x, CE2);
	p = fmadd_d(p, _mm256_mul_pd(x, x), _mm256_add_pd(x, one));
	x = _mm256_mul_pd(p, fexppart);
	return x;
}

inline void sincos(__m256d x, __m256d *sinval, __m256d *cosval) {
	/* Constants to subtract Pi/4*x from y while minimizing precision loss */
	const auto argred0 = _mm256_set1_pd(-2 * 0.78539816290140151978);
	const auto argred1 = _mm256_set1_pd(-2 * 4.9604678871439933374e-10);
	const auto argred2 = _mm256_set1_pd(-2 * 1.1258708853173288931e-18);
	const auto argred3 = _mm256_set1_pd(-2 * 1.7607799325916000908e-27);
	const auto two_over_pi = _mm256_set1_pd(2.0 / PI);
	const auto const_sin5 = _mm256_set1_pd(1.58938307283228937328511e-10);
	const auto const_sin4 = _mm256_set1_pd(-2.50506943502539773349318e-08);
	const auto const_sin3 = _mm256_set1_pd(2.75573131776846360512547e-06);
	const auto const_sin2 = _mm256_set1_pd(-0.000198412698278911770864914);
	const auto const_sin1 = _mm256_set1_pd(0.0083333333333191845961746);
	const auto const_sin0 = _mm256_set1_pd(-0.166666666666666130709393);

	const auto const_cos7 = _mm256_set1_pd(-1.13615350239097429531523e-11);
	const auto const_cos6 = _mm256_set1_pd(2.08757471207040055479366e-09);
	const auto const_cos5 = _mm256_set1_pd(-2.75573144028847567498567e-07);
	const auto const_cos4 = _mm256_set1_pd(2.48015872890001867311915e-05);
	const auto const_cos3 = _mm256_set1_pd(-0.00138888888888714019282329);
	const auto const_cos2 = _mm256_set1_pd(0.0416666666666665519592062);
	const auto half = _mm256_set1_pd(0.5);
	const auto one = _mm256_set1_pd(1.0);
	__m256d ssign, csign;
	__m256d x2, y, z, psin, pcos, sss, ccc;
	__m256d mask;

	const __m128i ione = _mm_set1_epi32(1);
	const __m128i itwo = _mm_set1_epi32(2);
	__m128i iy;

	z = _mm256_mul_pd(x, two_over_pi);
	iy = _mm256_cvtpd_epi32(z);
	y = _mm256_round_pd(z, _MM_FROUND_NINT);

	mask = gmx_simd_cvt_dib2db_avx_256(_mm_cmpeq_epi32(_mm_and_si128(iy, ione), _mm_setzero_si128()));
	ssign = _mm256_and_pd(_mm256_set1_pd(GMX_DOUBLE_NEGZERO),
	gmx_simd_cvt_dib2db_avx_256(_mm_cmpeq_epi32(_mm_and_si128(iy, itwo), itwo)));
	csign = _mm256_and_pd(_mm256_set1_pd(GMX_DOUBLE_NEGZERO),
	gmx_simd_cvt_dib2db_avx_256(_mm_cmpeq_epi32(_mm_and_si128(_mm_add_epi32(iy, ione), itwo), itwo)));

	x = fmadd_d(y, argred0, x);
	x = fmadd_d(y, argred1, x);
	x = fmadd_d(y, argred2, x);
	x = fmadd_d(y, argred3, x);
	x2 = _mm256_mul_pd(x, x);

	psin = fmadd_d(const_sin5, x2, const_sin4);
	psin = fmadd_d(psin, x2, const_sin3);
	psin = fmadd_d(psin, x2, const_sin2);
	psin = fmadd_d(psin, x2, const_sin1);
	psin = fmadd_d(psin, x2, const_sin0);
	psin = fmadd_d(psin, _mm256_mul_pd(x2, x), x);

	pcos = fmadd_d(const_cos7, x2, const_cos6);
	pcos = fmadd_d(pcos, x2, const_cos5);
	pcos = fmadd_d(pcos, x2, const_cos4);
	pcos = fmadd_d(pcos, x2, const_cos3);
	pcos = fmadd_d(pcos, x2, const_cos2);
	pcos = fmsub_d(pcos, x2, half);
	pcos = fmadd_d(pcos, x2, one);

	sss = _mm256_blendv_pd(pcos, psin, mask);
	ccc = _mm256_blendv_pd(psin, pcos, mask);

	*sinval = _mm256_xor_pd(sss, ssign);
	*cosval = _mm256_xor_pd(ccc, csign);
}

__m256d sin(__m256d x) {
	__m256d s, c;
	gromacs::sincos(x, &s, &c);
	return s;
}

__m256d cos(__m256d x) {
	__m256d s, c;
	gromacs::sincos(x, &s, &c);
	return c;
}

}
#endif //GROMACS_MATH_HPP
