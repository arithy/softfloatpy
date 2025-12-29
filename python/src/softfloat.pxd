# SoftFloatPy: A Python binding of Berkeley SoftFloat.
#
# Copyright (c) 2024-2025 Arihiro Yoshida. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# cython: language_level=3

from libc.stdint cimport (
    uint16_t, uint32_t, uint64_t,
    int32_t, int64_t,
    uint_fast8_t, uint_fast32_t, uint_fast64_t,
    int_fast32_t, int_fast64_t
)


cdef extern from "softfloat.h":

    ctypedef struct bfloat16_t:
        uint16_t v

    ctypedef struct float16_t:
        uint16_t v

    ctypedef struct float32_t:
        uint32_t v

    ctypedef struct float64_t:
        uint64_t v

    ctypedef struct float128_t:
        uint64_t[2] v

    enum:
        softfloat_tininess_beforeRounding,
        softfloat_tininess_afterRounding

    enum:
        softfloat_round_near_even,
        softfloat_round_minMag,
        softfloat_round_min,
        softfloat_round_max,
        softfloat_round_near_maxMag,
        softfloat_round_odd

    enum:
        softfloat_flag_inexact,
        softfloat_flag_underflow,
        softfloat_flag_overflow,
        softfloat_flag_infinite,
        softfloat_flag_invalid

    cdef uint_fast8_t softfloat_detectTininess
    cdef uint_fast8_t softfloat_roundingMode
    cdef uint_fast8_t softfloat_exceptionFlags

    float16_t ui32_to_f16(uint32_t)
    float32_t ui32_to_f32(uint32_t)
    float64_t ui32_to_f64(uint32_t)
    float128_t ui32_to_f128(uint32_t)

    float16_t ui64_to_f16(uint64_t)
    float32_t ui64_to_f32(uint64_t)
    float64_t ui64_to_f64(uint64_t)
    float128_t ui64_to_f128(uint64_t)

    float16_t i32_to_f16(int32_t)
    float32_t i32_to_f32(int32_t)
    float64_t i32_to_f64(int32_t)
    float128_t i32_to_f128(int32_t)

    float16_t i64_to_f16(int64_t)
    float32_t i64_to_f32(int64_t)
    float64_t i64_to_f64(int64_t)
    float128_t i64_to_f128(int64_t)

    uint_fast32_t f16_to_ui32(float16_t, uint_fast8_t, bint)
    uint_fast64_t f16_to_ui64(float16_t, uint_fast8_t, bint)
    int_fast32_t f16_to_i32(float16_t, uint_fast8_t, bint)
    int_fast64_t f16_to_i64(float16_t, uint_fast8_t, bint)
    float32_t f16_to_f32(float16_t)
    float64_t f16_to_f64(float16_t)
    float128_t f16_to_f128(float16_t)
    float16_t f16_roundToInt(float16_t, uint_fast8_t, bint)
    float16_t f16_add(float16_t, float16_t)
    float16_t f16_sub(float16_t, float16_t)
    float16_t f16_mul(float16_t, float16_t)
    float16_t f16_mulAdd(float16_t, float16_t, float16_t)
    float16_t f16_div(float16_t, float16_t)
    float16_t f16_rem(float16_t, float16_t)
    float16_t f16_sqrt(float16_t)
    bint f16_eq(float16_t, float16_t)
    bint f16_le(float16_t, float16_t)
    bint f16_lt(float16_t, float16_t)
    bint f16_eq_signaling(float16_t, float16_t)
    bint f16_le_quiet(float16_t, float16_t)
    bint f16_lt_quiet(float16_t, float16_t)
    bint f16_isSignalingNaN(float16_t)

    float32_t bf16_to_f32(bfloat16_t)
    bfloat16_t f32_to_bf16(float32_t)
    bint bf16_isSignalingNaN(bfloat16_t)

    uint_fast32_t f32_to_ui32(float32_t, uint_fast8_t, bint)
    uint_fast64_t f32_to_ui64(float32_t, uint_fast8_t, bint)
    int_fast32_t f32_to_i32(float32_t, uint_fast8_t, bint)
    int_fast64_t f32_to_i64(float32_t, uint_fast8_t, bint)
    float16_t f32_to_f16(float32_t)
    float64_t f32_to_f64(float32_t)
    float128_t f32_to_f128(float32_t)
    float32_t f32_roundToInt(float32_t, uint_fast8_t, bint)
    float32_t f32_add(float32_t, float32_t)
    float32_t f32_sub(float32_t, float32_t)
    float32_t f32_mul(float32_t, float32_t)
    float32_t f32_mulAdd(float32_t, float32_t, float32_t)
    float32_t f32_div(float32_t, float32_t)
    float32_t f32_rem(float32_t, float32_t)
    float32_t f32_sqrt(float32_t)
    bint f32_eq(float32_t, float32_t)
    bint f32_le(float32_t, float32_t)
    bint f32_lt(float32_t, float32_t)
    bint f32_eq_signaling(float32_t, float32_t)
    bint f32_le_quiet(float32_t, float32_t)
    bint f32_lt_quiet(float32_t, float32_t)
    bint f32_isSignalingNaN(float32_t)

    uint_fast32_t f64_to_ui32(float64_t, uint_fast8_t, bint)
    uint_fast64_t f64_to_ui64(float64_t, uint_fast8_t, bint)
    int_fast32_t f64_to_i32(float64_t, uint_fast8_t, bint)
    int_fast64_t f64_to_i64(float64_t, uint_fast8_t, bint)
    float16_t f64_to_f16(float64_t)
    float32_t f64_to_f32(float64_t)
    float128_t f64_to_f128(float64_t)
    float64_t f64_roundToInt(float64_t, uint_fast8_t, bint)
    float64_t f64_add(float64_t, float64_t)
    float64_t f64_sub(float64_t, float64_t)
    float64_t f64_mul(float64_t, float64_t)
    float64_t f64_mulAdd(float64_t, float64_t, float64_t)
    float64_t f64_div(float64_t, float64_t)
    float64_t f64_rem(float64_t, float64_t)
    float64_t f64_sqrt(float64_t)
    bint f64_eq(float64_t, float64_t)
    bint f64_le(float64_t, float64_t)
    bint f64_lt(float64_t, float64_t)
    bint f64_eq_signaling(float64_t, float64_t)
    bint f64_le_quiet(float64_t, float64_t)
    bint f64_lt_quiet(float64_t, float64_t)
    bint f64_isSignalingNaN(float64_t)

    uint_fast32_t f128_to_ui32(float128_t, uint_fast8_t, bint)
    uint_fast64_t f128_to_ui64(float128_t, uint_fast8_t, bint)
    int_fast32_t f128_to_i32(float128_t, uint_fast8_t, bint)
    int_fast64_t f128_to_i64(float128_t, uint_fast8_t, bint)
    float16_t f128_to_f16(float128_t)
    float32_t f128_to_f32(float128_t)
    float64_t f128_to_f64(float128_t)
    float128_t f128_roundToInt(float128_t, uint_fast8_t, bint)
    float128_t f128_add(float128_t, float128_t)
    float128_t f128_sub(float128_t, float128_t)
    float128_t f128_mul(float128_t, float128_t)
    float128_t f128_mulAdd(float128_t, float128_t, float128_t)
    float128_t f128_div(float128_t, float128_t)
    float128_t f128_rem(float128_t, float128_t)
    float128_t f128_sqrt(float128_t)
    bint f128_eq(float128_t, float128_t)
    bint f128_le(float128_t, float128_t)
    bint f128_lt(float128_t, float128_t)
    bint f128_eq_signaling(float128_t, float128_t)
    bint f128_le_quiet(float128_t, float128_t)
    bint f128_lt_quiet(float128_t, float128_t)
    bint f128_isSignalingNaN(float128_t)
