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

_SIGNALING_NAN: bytes = b'\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01'


def test_f128_bytes() -> None:
    b: bytes = b'\xed\xcb\xaf\x09\x90\xfa\xbc\xde\x12\x34\x56\x78\x87\x65\x43\x21'
    assert sf.Float128.from_bytes(b).to_bytes() == b


def test_f128_float() -> None:
    f: float = -12.5
    assert sf.Float128.from_float(f).to_float() == f
    assert str(sf.Float128.from_float(f)) == str(f)


def test_f128_to_ui32() -> None:
    f: float = 12.3
    assert sf.f128_to_ui32(sf.Float128.from_float(f), sf.RoundingMode.MIN).to_int() == math.floor(f)


def test_f128_to_ui64() -> None:
    f: float = 12.3
    assert sf.f128_to_ui64(sf.Float128.from_float(f), sf.RoundingMode.MIN).to_int() == math.floor(f)


def test_f128_to_i32() -> None:
    f: float = -12.3
    assert sf.f128_to_i32(sf.Float128.from_float(f), sf.RoundingMode.MIN).to_int() == math.floor(f)


def test_f128_to_i64() -> None:
    f: float = -12.3
    assert sf.f128_to_i64(sf.Float128.from_float(f), sf.RoundingMode.MIN).to_int() == math.floor(f)


def test_f128_to_f16() -> None:
    f: float = -12.5
    assert sf.f128_to_f16(sf.Float128.from_float(f)).to_float() == f


def test_f128_to_f32() -> None:
    f: float = -12.5
    assert sf.f128_to_f32(sf.Float128.from_float(f)).to_float() == f


def test_f128_to_f64() -> None:
    f: float = -12.5
    assert sf.f128_to_f64(sf.Float128.from_float(f)).to_float() == f


def test_f128_round_to_int() -> None:
    f: float = -12.3
    assert sf.f128_round_to_int(sf.Float128.from_float(f), sf.RoundingMode.MIN).to_float() == math.floor(f)


def test_f128_add() -> None:
    x: float = -12.5
    y: float = 3.25
    assert sf.f128_add(sf.Float128.from_float(x), sf.Float128.from_float(y)).to_float() == x + y


def test_f128_sub() -> None:
    x: float = -12.5
    y: float = 3.25
    assert sf.f128_sub(sf.Float128.from_float(x), sf.Float128.from_float(y)).to_float() == x - y


def test_f128_mul() -> None:
    x: float = -12.5
    y: float = 3.25
    assert sf.f128_mul(sf.Float128.from_float(x), sf.Float128.from_float(y)).to_float() == x * y


def test_f128_mul_add() -> None:
    x: float = -12.5
    y: float = 3.25
    z: float = -0.125
    assert sf.f128_mul_add(
        sf.Float128.from_float(x), sf.Float128.from_float(y), sf.Float128.from_float(z)
    ).to_float() == x * y + z


def test_f128_div() -> None:
    for x, y in [(8.75, 3.5), (-8.75, 3.5), (8.75, -3.5), (-8.75, -3.5)]:
        assert sf.f128_div(sf.Float128.from_float(x), sf.Float128.from_float(y)).to_float() == x / y


def test_f128_rem() -> None:
    for x, y in [(8.75, 3.5), (-8.75, 3.5), (8.75, -3.5), (-8.75, -3.5)]:
        z: float = x / y
        assert sf.f128_rem(sf.Float128.from_float(x), sf.Float128.from_float(y)).to_float() == x - y * (math.floor(z) if z >= 0 else math.ceil(z))


def test_f128_sqrt() -> None:
    x: float = 2.25
    assert sf.f128_sqrt(sf.Float128.from_float(x)).to_float() == math.sqrt(x)


def test_f128_eq() -> None:
    x: float = -12.5
    y: float = 3.25
    assert sf.f128_eq(sf.Float128.from_float(x), sf.Float128.from_float(x))
    assert not sf.f128_eq(sf.Float128.from_float(x), sf.Float128.from_float(y))


def test_f128_le() -> None:
    x: float = -12.5
    y: float = 3.25
    assert sf.f128_le(sf.Float128.from_float(x), sf.Float128.from_float(x))
    assert sf.f128_le(sf.Float128.from_float(x), sf.Float128.from_float(y))
    assert not sf.f128_le(sf.Float128.from_float(y), sf.Float128.from_float(x))


def test_f128_lt() -> None:
    x: float = -12.5
    y: float = 3.25
    assert not sf.f128_lt(sf.Float128.from_float(x), sf.Float128.from_float(x))
    assert sf.f128_lt(sf.Float128.from_float(x), sf.Float128.from_float(y))
    assert not sf.f128_lt(sf.Float128.from_float(y), sf.Float128.from_float(x))


def test_f128_eq_signaling() -> None:
    x: float = -12.5
    sf.set_exception_flags(0)
    assert sf.f128_eq_signaling(sf.Float128.from_float(x), sf.Float128.from_float(x))
    assert not sf.test_exception_flags(sf.ExceptionFlag.INVALID)
    sf.set_exception_flags(0)
    assert not sf.f128_eq_signaling(sf.Float128.from_float(x), sf.Float128.from_bytes(_SIGNALING_NAN))
    assert sf.test_exception_flags(sf.ExceptionFlag.INVALID)


def test_f128_le_quiet() -> None:
    x: float = -12.5
    sf.set_exception_flags(0)
    assert sf.f128_le_quiet(sf.Float128.from_float(x), sf.Float128.from_float(x))
    assert not sf.test_exception_flags(sf.ExceptionFlag.INVALID)
    sf.set_exception_flags(0)
    assert not sf.f128_le_quiet(sf.Float128.from_float(x), sf.Float128.from_bytes(_SIGNALING_NAN))
    assert sf.test_exception_flags(sf.ExceptionFlag.INVALID)


def test_f128_lt_quiet() -> None:
    x: float = -12.5
    sf.set_exception_flags(0)
    assert not sf.f128_lt_quiet(sf.Float128.from_float(x), sf.Float128.from_float(x))
    assert not sf.test_exception_flags(sf.ExceptionFlag.INVALID)
    sf.set_exception_flags(0)
    assert not sf.f128_lt_quiet(sf.Float128.from_float(x), sf.Float128.from_bytes(_SIGNALING_NAN))
    assert sf.test_exception_flags(sf.ExceptionFlag.INVALID)


def test_f128_is_signaling_nan() -> None:
    assert sf.f128_is_signaling_nan(sf.Float128.from_bytes(_SIGNALING_NAN))


def test_operators() -> None:
    ix: sf.Float128
    iy: sf.Float128
    iz: sf.Float128
    for x in [8.75, -8.75]:
        ix = sf.Float128.from_float(x)
        assert +ix is not ix
        assert (+ix).to_float() == +x
        assert (-ix).to_float() == -x
    for x, y in [(8.75, 3.5), (-8.75, 3.5), (8.75, -3.5), (-8.75, -3.5)]:
        ix = sf.Float128.from_float(x)
        iy = sf.Float128.from_float(y)
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
