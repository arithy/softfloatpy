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

import softfloatpy as sf

_SIGNALING_NAN: bytes = b'\xfc\x01'


def test_f16_size() -> None:
    assert sf.Float16.size() == 16


def test_f16_bytes() -> None:
    b: bytes = b'\xed\xcb'
    assert sf.Float16.from_bytes(b).to_bytes() == b


def test_f16_float() -> None:
    f: float = -12.5
    o: sf.Float16 = sf.Float16.from_float(f)
    assert o.to_float() == f
    assert str(o) == str(f)


def test_f16_to_ui32() -> None:
    f: float = 12.3
    o: sf.Float16 = sf.Float16.from_float(f)
    assert sf.f16_to_ui32(o, sf.RoundingMode.MIN).to_int() == math.floor(f)
    assert sf.f16_to_ui32(o, sf.RoundingMode.MIN).to_bytes() == o.to_ui32(sf.RoundingMode.MIN).to_bytes()


def test_f16_to_ui64() -> None:
    f: float = 12.3
    o: sf.Float16 = sf.Float16.from_float(f)
    assert sf.f16_to_ui64(o, sf.RoundingMode.MIN).to_int() == math.floor(f)
    assert sf.f16_to_ui64(o, sf.RoundingMode.MIN).to_bytes() == o.to_ui64(sf.RoundingMode.MIN).to_bytes()


def test_f16_to_i32() -> None:
    f: float = -12.3
    o: sf.Float16 = sf.Float16.from_float(f)
    assert sf.f16_to_i32(o, sf.RoundingMode.MIN).to_int() == math.floor(f)
    assert sf.f16_to_i32(o, sf.RoundingMode.MIN).to_bytes() == o.to_i32(sf.RoundingMode.MIN).to_bytes()


def test_f16_to_i64() -> None:
    f: float = -12.3
    o: sf.Float16 = sf.Float16.from_float(f)
    assert sf.f16_to_i64(o, sf.RoundingMode.MIN).to_int() == math.floor(f)
    assert sf.f16_to_i64(o, sf.RoundingMode.MIN).to_bytes() == o.to_i64(sf.RoundingMode.MIN).to_bytes()


def test_f16_to_f32() -> None:
    f: float = -12.5
    o: sf.Float16 = sf.Float16.from_float(f)
    assert sf.f16_to_f32(o).to_float() == f
    assert sf.f16_to_f32(o).to_bytes() == o.to_f32().to_bytes()


def test_f16_to_f64() -> None:
    f: float = -12.5
    o: sf.Float16 = sf.Float16.from_float(f)
    assert sf.f16_to_f64(o).to_float() == f
    assert sf.f16_to_f64(o).to_bytes() == o.to_f64().to_bytes()


def test_f16_to_f128() -> None:
    f: float = -12.5
    o: sf.Float16 = sf.Float16.from_float(f)
    assert sf.f16_to_f128(o).to_float() == f
    assert sf.f16_to_f128(o).to_bytes() == o.to_f128().to_bytes()


def test_f16_round_to_int() -> None:
    f: float = -12.3
    o: sf.Float16 = sf.Float16.from_float(f)
    assert sf.f16_round_to_int(o, sf.RoundingMode.MIN).to_float() == math.floor(f)
    assert sf.f16_round_to_int(o, sf.RoundingMode.MIN).to_bytes() == o.round_to_int(sf.RoundingMode.MIN).to_bytes()


def test_f16_add() -> None:
    x: float = -12.5
    y: float = 3.25
    o: sf.Float16 = sf.Float16.from_float(x)
    p: sf.Float16 = sf.Float16.from_float(y)
    assert sf.f16_add(o, p).to_float() == x + y
    assert sf.f16_add(o, p).to_bytes() == sf.Float16.add(o, p).to_bytes()


def test_f16_sub() -> None:
    x: float = -12.5
    y: float = 3.25
    o: sf.Float16 = sf.Float16.from_float(x)
    p: sf.Float16 = sf.Float16.from_float(y)
    assert sf.f16_sub(o, p).to_float() == x - y
    assert sf.f16_sub(o, p).to_bytes() == sf.Float16.sub(o, p).to_bytes()


def test_f16_mul() -> None:
    x: float = -12.5
    y: float = 3.25
    o: sf.Float16 = sf.Float16.from_float(x)
    p: sf.Float16 = sf.Float16.from_float(y)
    assert sf.f16_mul(o, p).to_float() == x * y
    assert sf.f16_mul(o, p).to_bytes() == sf.Float16.mul(o, p).to_bytes()


def test_f16_mul_add() -> None:
    x: float = -12.5
    y: float = 3.25
    z: float = -0.125
    o: sf.Float16 = sf.Float16.from_float(x)
    p: sf.Float16 = sf.Float16.from_float(y)
    q: sf.Float16 = sf.Float16.from_float(z)
    assert sf.f16_mul_add(o, p, q).to_float() == x * y + z
    assert sf.f16_mul_add(o, p, q).to_bytes() == sf.Float16.mul_add(o, p, q).to_bytes()


def test_f16_div() -> None:
    for x, y in [(8.75, 3.5), (-8.75, 3.5), (8.75, -3.5), (-8.75, -3.5)]:
        o: sf.Float16 = sf.Float16.from_float(x)
        p: sf.Float16 = sf.Float16.from_float(y)
        assert sf.f16_div(o, p).to_float() == x / y
        assert sf.f16_div(o, p).to_bytes() == sf.Float16.div(o, p).to_bytes()


def test_f16_rem() -> None:
    for x, y in [(8.75, 3.5), (-8.75, 3.5), (8.75, -3.5), (-8.75, -3.5)]:
        z: float = x / y
        o: sf.Float16 = sf.Float16.from_float(x)
        p: sf.Float16 = sf.Float16.from_float(y)
        assert sf.f16_rem(o, p).to_float() == x - y * (math.floor(z) if z >= 0 else math.ceil(z))
        assert sf.f16_rem(o, p).to_bytes() == sf.Float16.rem(o, p).to_bytes()


def test_f16_sqrt() -> None:
    x: float = 2.25
    o: sf.Float16 = sf.Float16.from_float(x)
    assert sf.f16_sqrt(o).to_float() == math.sqrt(x)
    assert sf.f16_sqrt(o).to_bytes() == sf.Float16.sqrt(o).to_bytes()


def test_f16_eq() -> None:
    x: float = -12.5
    y: float = 3.25
    o: sf.Float16 = sf.Float16.from_float(x)
    p: sf.Float16 = sf.Float16.from_float(y)
    assert sf.f16_eq(o, o)
    assert not sf.f16_eq(o, p)
    assert sf.Float16.eq(o, o)
    assert not sf.Float16.eq(o, p)


def test_f16_le() -> None:
    x: float = -12.5
    y: float = 3.25
    o: sf.Float16 = sf.Float16.from_float(x)
    p: sf.Float16 = sf.Float16.from_float(y)
    assert sf.f16_le(o, o)
    assert sf.f16_le(o, p)
    assert not sf.f16_le(p, o)
    assert sf.Float16.le(o, o)
    assert sf.Float16.le(o, p)
    assert not sf.Float16.le(p, o)


def test_f16_lt() -> None:
    x: float = -12.5
    y: float = 3.25
    o: sf.Float16 = sf.Float16.from_float(x)
    p: sf.Float16 = sf.Float16.from_float(y)
    assert not sf.f16_lt(o, o)
    assert sf.f16_lt(o, p)
    assert not sf.f16_lt(p, o)
    assert not sf.Float16.lt(o, o)
    assert sf.Float16.lt(o, p)
    assert not sf.Float16.lt(p, o)


def test_f16_eq_signaling() -> None:
    x: float = -12.5
    o: sf.Float16 = sf.Float16.from_float(x)
    p: sf.Float16 = sf.Float16.from_bytes(_SIGNALING_NAN)
    sf.set_exception_flags(0)
    assert sf.f16_eq_signaling(o, o)
    assert not sf.test_exception_flags(sf.ExceptionFlag.INVALID)
    sf.set_exception_flags(0)
    assert not sf.f16_eq_signaling(o, p)
    assert sf.test_exception_flags(sf.ExceptionFlag.INVALID)
    sf.set_exception_flags(0)
    assert sf.Float16.eq_signaling(o, o)
    assert not sf.test_exception_flags(sf.ExceptionFlag.INVALID)
    sf.set_exception_flags(0)
    assert not sf.Float16.eq_signaling(o, p)
    assert sf.test_exception_flags(sf.ExceptionFlag.INVALID)


def test_f16_le_quiet() -> None:
    x: float = -12.5
    o: sf.Float16 = sf.Float16.from_float(x)
    p: sf.Float16 = sf.Float16.from_bytes(_SIGNALING_NAN)
    sf.set_exception_flags(0)
    assert sf.f16_le_quiet(o, o)
    assert not sf.test_exception_flags(sf.ExceptionFlag.INVALID)
    sf.set_exception_flags(0)
    assert not sf.f16_le_quiet(o, p)
    assert sf.test_exception_flags(sf.ExceptionFlag.INVALID)
    sf.set_exception_flags(0)
    assert sf.Float16.le_quiet(o, o)
    assert not sf.test_exception_flags(sf.ExceptionFlag.INVALID)
    sf.set_exception_flags(0)
    assert not sf.Float16.le_quiet(o, p)
    assert sf.test_exception_flags(sf.ExceptionFlag.INVALID)


def test_f16_lt_quiet() -> None:
    x: float = -12.5
    o: sf.Float16 = sf.Float16.from_float(x)
    p: sf.Float16 = sf.Float16.from_bytes(_SIGNALING_NAN)
    sf.set_exception_flags(0)
    assert not sf.f16_lt_quiet(o, o)
    assert not sf.test_exception_flags(sf.ExceptionFlag.INVALID)
    sf.set_exception_flags(0)
    assert not sf.f16_lt_quiet(o, p)
    assert sf.test_exception_flags(sf.ExceptionFlag.INVALID)
    sf.set_exception_flags(0)
    assert not sf.Float16.lt_quiet(o, o)
    assert not sf.test_exception_flags(sf.ExceptionFlag.INVALID)
    sf.set_exception_flags(0)
    assert not sf.Float16.lt_quiet(o, p)
    assert sf.test_exception_flags(sf.ExceptionFlag.INVALID)


def test_f16_is_signaling_nan() -> None:
    o: sf.Float16 = sf.Float16.from_bytes(_SIGNALING_NAN)
    assert sf.f16_is_signaling_nan(o)
    assert o.is_signaling_nan()


def test_operators() -> None:
    ix: sf.Float16
    iy: sf.Float16
    iz: sf.Float16
    for x in [8.75, -8.75]:
        ix = sf.Float16.from_float(x)
        assert +ix is not ix
        assert (+ix).to_float() == +x
        assert (-ix).to_float() == -x
    for x, y in [(8.75, 3.5), (-8.75, 3.5), (8.75, -3.5), (-8.75, -3.5)]:
        ix = sf.Float16.from_float(x)
        iy = sf.Float16.from_float(y)
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
