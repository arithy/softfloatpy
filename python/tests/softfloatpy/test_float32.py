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

import math

import softfloatpy as sf  # type: ignore

_SIGNALING_NAN: bytes = b'\xff\x80\x00\x01'


def test_f32_bytes() -> None:
    b: bytes = b'\xed\xcb\x43\x21'
    assert sf.Float32.from_bytes(b).to_bytes() == b


def test_f32_float() -> None:
    f: float = -12.5
    assert sf.Float32.from_float(f).to_float() == f
    assert str(sf.Float32.from_float(f)) == str(f)


def test_f32_to_ui32() -> None:
    f: float = 12.3
    assert sf.f32_to_ui32(sf.Float32.from_float(f), sf.RoundingMode.MIN).to_int() == math.floor(f)


def test_f32_to_ui64() -> None:
    f: float = 12.3
    assert sf.f32_to_ui64(sf.Float32.from_float(f), sf.RoundingMode.MIN).to_int() == math.floor(f)


def test_f32_to_i32() -> None:
    f: float = -12.3
    assert sf.f32_to_i32(sf.Float32.from_float(f), sf.RoundingMode.MIN).to_int() == math.floor(f)


def test_f32_to_i64() -> None:
    f: float = -12.3
    assert sf.f32_to_i64(sf.Float32.from_float(f), sf.RoundingMode.MIN).to_int() == math.floor(f)


def test_f32_to_f16() -> None:
    f: float = -12.5
    assert sf.f32_to_f16(sf.Float32.from_float(f)).to_float() == f


def test_f32_to_f64() -> None:
    f: float = -12.5
    assert sf.f32_to_f64(sf.Float32.from_float(f)).to_float() == f


def test_f32_to_f128() -> None:
    f: float = -12.5
    assert sf.f32_to_f128(sf.Float32.from_float(f)).to_float() == f


def test_f32_round_to_int() -> None:
    f: float = -12.3
    assert sf.f32_round_to_int(sf.Float32.from_float(f), sf.RoundingMode.MIN).to_float() == math.floor(f)


def test_f32_add() -> None:
    x: float = -12.5
    y: float = 3.25
    assert sf.f32_add(sf.Float32.from_float(x), sf.Float32.from_float(y)).to_float() == x + y


def test_f32_sub() -> None:
    x: float = -12.5
    y: float = 3.25
    assert sf.f32_sub(sf.Float32.from_float(x), sf.Float32.from_float(y)).to_float() == x - y


def test_f32_mul() -> None:
    x: float = -12.5
    y: float = 3.25
    assert sf.f32_mul(sf.Float32.from_float(x), sf.Float32.from_float(y)).to_float() == x * y


def test_f32_mul_add() -> None:
    x: float = -12.5
    y: float = 3.25
    z: float = -0.125
    assert sf.f32_mul_add(
        sf.Float32.from_float(x), sf.Float32.from_float(y), sf.Float32.from_float(z)
    ).to_float() == x * y + z


def test_f32_div() -> None:
    for x, y in [(8.75, 3.5), (-8.75, 3.5), (8.75, -3.5), (-8.75, -3.5)]:
        assert sf.f32_div(sf.Float32.from_float(x), sf.Float32.from_float(y)).to_float() == x / y


def test_f32_rem() -> None:
    for x, y in [(8.75, 3.5), (-8.75, 3.5), (8.75, -3.5), (-8.75, -3.5)]:
        z: float = x / y
        assert sf.f32_rem(sf.Float32.from_float(x), sf.Float32.from_float(y)).to_float() == x - y * (math.floor(z) if z >= 0 else math.ceil(z))


def test_f32_sqrt() -> None:
    x: float = 2.25
    assert sf.f32_sqrt(sf.Float32.from_float(x)).to_float() == math.sqrt(x)


def test_f32_eq() -> None:
    x: float = -12.5
    y: float = 3.25
    assert sf.f32_eq(sf.Float32.from_float(x), sf.Float32.from_float(x))
    assert not sf.f32_eq(sf.Float32.from_float(x), sf.Float32.from_float(y))


def test_f32_le() -> None:
    x: float = -12.5
    y: float = 3.25
    assert sf.f32_le(sf.Float32.from_float(x), sf.Float32.from_float(x))
    assert sf.f32_le(sf.Float32.from_float(x), sf.Float32.from_float(y))
    assert not sf.f32_le(sf.Float32.from_float(y), sf.Float32.from_float(x))


def test_f32_lt() -> None:
    x: float = -12.5
    y: float = 3.25
    assert not sf.f32_lt(sf.Float32.from_float(x), sf.Float32.from_float(x))
    assert sf.f32_lt(sf.Float32.from_float(x), sf.Float32.from_float(y))
    assert not sf.f32_lt(sf.Float32.from_float(y), sf.Float32.from_float(x))


def test_f32_eq_signaling() -> None:
    x: float = -12.5
    sf.set_exception_flags(0)
    assert sf.f32_eq_signaling(sf.Float32.from_float(x), sf.Float32.from_float(x))
    assert not sf.test_exception_flags(sf.ExceptionFlag.INVALID)
    sf.set_exception_flags(0)
    assert not sf.f32_eq_signaling(sf.Float32.from_float(x), sf.Float32.from_bytes(_SIGNALING_NAN))
    assert sf.test_exception_flags(sf.ExceptionFlag.INVALID)


def test_f32_le_quiet() -> None:
    x: float = -12.5
    sf.set_exception_flags(0)
    assert sf.f32_le_quiet(sf.Float32.from_float(x), sf.Float32.from_float(x))
    assert not sf.test_exception_flags(sf.ExceptionFlag.INVALID)
    sf.set_exception_flags(0)
    assert not sf.f32_le_quiet(sf.Float32.from_float(x), sf.Float32.from_bytes(_SIGNALING_NAN))
    assert sf.test_exception_flags(sf.ExceptionFlag.INVALID)


def test_f32_lt_quiet() -> None:
    x: float = -12.5
    sf.set_exception_flags(0)
    assert not sf.f32_lt_quiet(sf.Float32.from_float(x), sf.Float32.from_float(x))
    assert not sf.test_exception_flags(sf.ExceptionFlag.INVALID)
    sf.set_exception_flags(0)
    assert not sf.f32_lt_quiet(sf.Float32.from_float(x), sf.Float32.from_bytes(_SIGNALING_NAN))
    assert sf.test_exception_flags(sf.ExceptionFlag.INVALID)


def test_f32_is_signaling_nan() -> None:
    assert sf.f32_is_signaling_nan(sf.Float32.from_bytes(_SIGNALING_NAN))


def test_operators() -> None:
    ix: sf.Float32
    iy: sf.Float32
    iz: sf.Float32
    for x in [8.75, -8.75]:
        ix = sf.Float32.from_float(x)
        assert +ix is not ix
        assert (+ix).to_float() == +x
        assert (-ix).to_float() == -x
    for x, y in [(8.75, 3.5), (-8.75, 3.5), (8.75, -3.5), (-8.75, -3.5)]:
        ix = sf.Float32.from_float(x)
        iy = sf.Float32.from_float(y)
        assert (ix + iy).to_float() == x + y
        assert (ix - iy).to_float() == x - y
        assert (ix * iy).to_float() == x * y
        assert (ix / iy).to_float() == x / y
        assert (ix // iy).to_float() == x // y
        assert (ix % iy).to_float() == x % y
        assert (ix < iy) == (x < y)
        assert (ix <= iy) == (x <= y)
        assert (ix > iy) == (x > y)
        assert (ix >= iy) == (x >= y)
        assert (ix == iy) == (x == y)
        assert (ix != iy) == (x != y)
        iz = ix
        iz += iy
        assert iz is not ix
        assert iz.to_float() == x + y
        iz = ix
        iz -= iy
        assert iz is not ix
        assert iz.to_float() == x - y
        iz = ix
        iz *= iy
        assert iz is not ix
        assert iz.to_float() == x * y
        iz = ix
        iz /= iy
        assert iz is not ix
        assert iz.to_float() == x / y
        iz = ix
        iz //= iy
        assert iz is not ix
        assert iz.to_float() == x // y
        iz = ix
        iz %= iy
        assert iz is not ix
        assert iz.to_float() == x % y
