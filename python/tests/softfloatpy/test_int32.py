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

import softfloatpy as sf  # type: ignore


def test_i32_bytes() -> None:
    b: bytes = b'\xed\xcb\x43\x21'
    assert sf.Int32.from_bytes(b).to_bytes() == b


def test_i32_int() -> None:
    i: int = -0x6dcb4321
    assert sf.Int32.from_int(i).to_int() == i
    assert str(sf.Int32.from_int(i)) == str(i)


def test_i32_to_f16() -> None:
    i: int = -123
    assert sf.i32_to_f16(sf.Int32.from_int(i)).to_float() == float(i)


def test_i32_to_f32() -> None:
    i: int = -123
    assert sf.i32_to_f32(sf.Int32.from_int(i)).to_float() == float(i)


def test_i32_to_f64() -> None:
    i: int = -123
    assert sf.i32_to_f64(sf.Int32.from_int(i)).to_float() == float(i)


def test_i32_to_f128() -> None:
    i: int = -123
    assert sf.i32_to_f128(sf.Int32.from_int(i)).to_float() == float(i)


def test_operators() -> None:
    ix: sf.Int32
    iy: sf.Int32
    iz: sf.Int32
    for x in [8, -8]:
        ix = sf.Int32.from_int(x)
        assert +ix is not ix
        assert (+ix).to_int() == +x
        assert (-ix).to_int() == -x
        assert (~ix).to_int() == ~x
    for x, y in [(8, 3), (-8, 3), (8, -3), (-8, -3)]:
        ix = sf.Int32.from_int(x)
        iy = sf.Int32.from_int(y)
        assert (ix + iy).to_int() == x + y
        assert (ix - iy).to_int() == x - y
        assert (ix * iy).to_int() == x * y
        assert (ix // iy).to_int() == x // y
        assert (ix % iy).to_int() == x % y
        if y >= 0:
            assert (ix << iy).to_int() == x << y
            assert (ix >> iy).to_int() == x >> y
        assert (ix & iy).to_int() == x & y
        assert (ix | iy).to_int() == x | y
        assert (ix ^ iy).to_int() == x ^ y
        assert (ix < iy) == (x < y)
        assert (ix <= iy) == (x <= y)
        assert (ix > iy) == (x > y)
        assert (ix >= iy) == (x >= y)
        assert (ix == iy) == (x == y)
        assert (ix != iy) == (x != y)
        iz = ix
        iz += iy
        assert iz is not ix
        assert iz.to_int() == x + y
        iz = ix
        iz -= iy
        assert iz is not ix
        assert iz.to_int() == x - y
        iz = ix
        iz *= iy
        assert iz is not ix
        assert iz.to_int() == x * y
        iz = ix
        iz //= iy
        assert iz is not ix
        assert iz.to_int() == x // y
        iz = ix
        iz %= iy
        assert iz is not ix
        assert iz.to_int() == x % y
        if y >= 0:
            iz = ix
            iz <<= iy
            assert iz is not ix
            assert iz.to_int() == x << y
            iz = ix
            iz >>= iy
            assert iz is not ix
            assert iz.to_int() == x >> y
        iz = ix
        iz &= iy
        assert iz is not ix
        assert iz.to_int() == x & y
        iz = ix
        iz |= iy
        assert iz is not ix
        assert iz.to_int() == x | y
        iz = ix
        iz ^= iy
        assert iz is not ix
        assert iz.to_int() == x ^ y
