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

_SIGNALING_NAN: bytes = b'\xff\x81'


def test_bf16_bytes() -> None:
    b: bytes = b'\xed\xcb'
    assert sf.BFloat16.from_bytes(b).to_bytes() == b


def test_bf16_float() -> None:
    f: float = -12.5
    assert sf.BFloat16.from_float(f).to_float() == f
    assert str(sf.BFloat16.from_float(f)) == str(f)


def test_bf16_to_f32() -> None:
    f: float = -12.5
    assert sf.bf16_to_f32(sf.BFloat16.from_float(f)).to_float() == f


def test_f32_to_bf16() -> None:
    f: float = -12.5
    assert sf.f32_to_bf16(sf.Float32.from_float(f)).to_float() == f


def test_bf16_is_signaling_nan() -> None:
    assert sf.bf16_is_signaling_nan(sf.BFloat16.from_bytes(_SIGNALING_NAN))
