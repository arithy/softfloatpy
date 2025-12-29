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

import softfloatpy as sf


def test_tininess_mode() -> None:
    for m in sf.TininessMode:
        sf.set_tininess_mode(m)
        assert sf.get_tininess_mode() == m


def test_rounding_mode() -> None:
    for m in sf.RoundingMode:
        sf.set_rounding_mode(m)
        assert sf.get_rounding_mode() == m


def test_exception_flags() -> None:
    for f in [0x19, 0x03]:
        sf.set_exception_flags(f)
        assert sf.get_exception_flags() == f


def test_exception_flags_inexact() -> None:
    sf.set_exception_flags(0)
    sf.f32_round_to_int(sf.Float32.from_float(1e30), sf.RoundingMode.MIN, exact=True)
    assert sf.test_exception_flags(sf.ExceptionFlag.INEXACT)


def test_exception_flags_underflow() -> None:
    sf.set_exception_flags(0)
    sf.f32_mul(sf.Float32.from_float(1e-30), sf.Float32.from_float(1e-30))
    assert sf.test_exception_flags(sf.ExceptionFlag.UNDERFLOW)


def test_exception_flags_overflow() -> None:
    sf.set_exception_flags(0)
    sf.f32_mul(sf.Float32.from_float(1e30), sf.Float32.from_float(1e30))
    assert sf.test_exception_flags(sf.ExceptionFlag.OVERFLOW)


def test_exception_flags_infinite() -> None:
    sf.set_exception_flags(0)
    sf.f32_div(sf.Float32.from_float(1.0), sf.Float32.from_float(0.0))
    assert sf.test_exception_flags(sf.ExceptionFlag.INFINITE)


def test_exception_flags_invalid() -> None:
    sf.set_exception_flags(0)
    sf.f32_div(sf.Float32.from_float(0.0), sf.Float32.from_float(0.0))
    assert sf.test_exception_flags(sf.ExceptionFlag.INVALID)
