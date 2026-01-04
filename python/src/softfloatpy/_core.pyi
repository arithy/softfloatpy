# SoftFloatPy: A Python binding of Berkeley SoftFloat.
#
# Copyright (c) 2024-2026 Arihiro Yoshida. All rights reserved.
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

from typing import Self
from enum import IntEnum, IntFlag


class TininessMode(IntEnum):
    """The tininess detection modes.

    - ``BEFORE_ROUNDING``: Detecting tininess before rounding.
    - ``AFTER_ROUNDING``: Detecting tininess after rounding.

    """
    BEFORE_ROUNDING = ...
    AFTER_ROUNDING = ...


class RoundingMode(IntEnum):
    """The rounding modes.

    - ``NEAR_EVEN``: Rounding to nearest, with ties to even.
    - ``NEAR_MAX_MAG``: Rounding to nearest, with ties to maximum magnitude (away from zero).
    - ``MIN_MAG``: Rounding to minimum magnitude (toward zero).
    - ``MIN``: Rounding to minimum (down).
    - ``MAX``: Rounding to maximum (up).

    """
    NEAR_EVEN = ...
    MIN_MAG = ...
    MIN = ...
    MAX = ...
    NEAR_MAX_MAG = ...


class ExceptionFlag(IntFlag):
    """The floating-point exception flags.

    - ``INEXACT``: The exception set if the rounded value is different from the mathematically exact result of the operation.
    - ``UNDERFLOW``: The exception set if the rounded value is tiny and inexact.
    - ``OVERFLOW``: The exception set if the absolute value of the rounded value is too large to be represented.
    - ``INFINITE``: The exception set if the result is infinite given finite operands.
    - ``INVALID``: The exception set if a finite or infinite result cannot be returned.

    """
    INEXACT = ...
    UNDERFLOW = ...
    OVERFLOW = ...
    INFINITE = ...
    INVALID = ...


class UInt32:
    """A 32-bit unsigned integer.

    The object is immutable.

    The following operators are supported:

    - unary operators: ``+``, ``~``.
    - binary operators: ``+``, ``-``, ``*``, ``//``, ``%``,
      ``<<``, ``>>``, ``&``, ``|``, ``^``, ``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``,
      ``+=``, ``-=``, ``*=``, ``//=``, ``%=``, ``<<=``, ``>>=``, ``&=``, ``|=``, ``^=``.

    The following operators are unsupported:

    - unary operators: ``-``.
    - binary operators: ``**``, ``/``, ``**=``, ``/=``.

    """

    @classmethod
    def size(cls) -> int:
        """Returns the native data size in bits.

        Returns:
            The native data size in bits, i.e. 32.

        """
        ...

    @classmethod
    def from_bytes(cls, src: bytes) -> Self:
        """Creates a new instance from the specified byte sequence.

        Args:
            src: The byte sequence representing the native data with big endian.
                 The length must be 4.

        Returns:
            A new instance created from the specified byte sequence.

        Raises:
            ValueError: If the length of bytes is not 4.

        """
        ...

    def to_bytes(self) -> bytes:
        """Returns the native data as a byte sequence.

        Returns:
            A byte sequence representing the native data with big endian.
            The length is 4.

        """
        ...

    @classmethod
    def from_int(cls, src: int) -> Self:
        """Creates a new instance from the specified integer.

        Args:
            src: The integer from which a new instance is created.

        Returns:
            A new instance created from the specified integer.

        """
        ...

    def to_int(self) -> int:
        """Returns the native data as an integer.

        Returns:
            An integer that represents the native data.

        """
        ...

    @classmethod
    def from_f16(cls, src: Float16, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True) -> Self:
        """Creates a new instance from the specified IEEE 754 binary16 floating point.

        The result is the same as that of :func:`f16_to_ui32()`.

        Args:
            src: The IEEE 754 binary16 floating point from which a new instance is created.
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            A new instance created from the specified IEEE 754 binary16 floating point.

        """
        ...

    def to_f16(self) -> Float16:
        """Converts the 32-bit unsigned integer to an IEEE 754 binary16 floating point.

        The result is the same as that of :func:`ui32_to_f16()`.

        Returns:
            The IEEE 754 binary16 floating point.

        """
        ...

    @classmethod
    def from_f32(cls, src: Float32, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True) -> Self:
        """Creates a new instance from the specified IEEE 754 binary32 floating point.

        The result is the same as that of :func:`f32_to_ui32()`.

        Args:
            src: The IEEE 754 binary32 floating point from which a new instance is created.
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            A new instance created from the specified IEEE 754 binary32 floating point.

        """
        ...

    def to_f32(self) -> Float32:
        """Converts the 32-bit unsigned integer to an IEEE 754 binary32 floating point.

        The result is the same as that of :func:`ui32_to_f32()`.

        Returns:
            The IEEE 754 binary32 floating point.

        """
        ...

    @classmethod
    def from_f64(cls, src: Float64, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True) -> Self:
        """Creates a new instance from the specified IEEE 754 binary64 floating point.

        The result is the same as that of :func:`f64_to_ui32()`.

        Args:
            src: The IEEE 754 binary64 floating point from which a new instance is created.
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            A new instance created from the specified IEEE 754 binary64 floating point.

        """
        ...

    def to_f64(self) -> Float64:
        """Converts the 32-bit unsigned integer to an IEEE 754 binary64 floating point.

        The result is the same as that of :func:`ui32_to_f64()`.

        Returns:
            The IEEE 754 binary64 floating point.

        """
        ...

    @classmethod
    def from_f128(cls, src: Float128, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True) -> Self:
        """Creates a new instance from the specified IEEE 754 binary128 floating point.

        The result is the same as that of :func:`f128_to_ui32()`.

        Args:
            src: The IEEE 754 binary128 floating point from which a new instance is created.
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            A new instance created from the specified IEEE 754 binary128 floating point.

        """
        ...

    def to_f128(self) -> Float128:
        """Converts the 32-bit unsigned integer to an IEEE 754 binary128 floating point.

        The result is the same as that of :func:`ui32_to_f128()`.

        Returns:
            The IEEE 754 binary128 floating point.

        """
        ...

    def __str__(self) -> str:
        ...

    def __pos__(self) -> Self:
        ...

    def __neg__(self) -> Self:
        ...

    def __invert__(self) -> Self:
        ...

    def __add__(self, other: Self) -> Self:
        ...

    def __sub__(self, other: Self) -> Self:
        ...

    def __mul__(self, other: Self) -> Self:
        ...

    def __floordiv__(self, other: Self) -> Self:
        ...

    def __mod__(self, other: Self) -> Self:
        ...

    def __lshift__(self, other: Self) -> Self:
        ...

    def __rshift__(self, other: Self) -> Self:
        ...

    def __and__(self, other: Self) -> Self:
        ...

    def __or__(self, other: Self) -> Self:
        ...

    def __xor__(self, other: Self) -> Self:
        ...

    def __lt__(self, other: Self) -> bool:
        ...

    def __le__(self, other: Self) -> bool:
        ...

    def __gt__(self, other: Self) -> bool:
        ...

    def __ge__(self, other: Self) -> bool:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __iadd__(self, other: Self) -> Self:
        ...

    def __isub__(self, other: Self) -> Self:
        ...

    def __imul__(self, other: Self) -> Self:
        ...

    def __ifloordiv__(self, other: Self) -> Self:
        ...

    def __imod__(self, other: Self) -> Self:
        ...

    def __ilshift__(self, other: Self) -> Self:
        ...

    def __irshift__(self, other: Self) -> Self:
        ...

    def __iand__(self, other: Self) -> Self:
        ...

    def __ior__(self, other: Self) -> Self:
        ...

    def __ixor__(self, other: Self) -> Self:
        ...


class UInt64:
    """A 64-bit unsigned integer.

    The object is immutable.

    The following operators are supported:

    - unary operators: ``+``, ``~``.
    - binary operators: ``+``, ``-``, ``*``, ``//``, ``%``,
      ``<<``, ``>>``, ``&``, ``|``, ``^``, ``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``,
      ``+=``, ``-=``, ``*=``, ``//=``, ``%=``, ``<<=``, ``>>=``, ``&=``, ``|=``, ``^=``.

    The following operators are unsupported:

    - unary operators: ``-``.
    - binary operators: ``**``, ``/``, ``**=``, ``/=``.

    """

    @classmethod
    def size(cls) -> int:
        """Returns the native data size in bits.

        Returns:
            The native data size in bits, i.e. 64.

        """
        ...

    @classmethod
    def from_bytes(cls, src: bytes) -> Self:
        """Creates a new instance from the specified byte sequence.

        Args:
            src: The byte sequence representing the native data with big endian.
                 The length must be 8.

        Returns:
            A new instance created from the specified byte sequence.

        Raises:
            ValueError: If the length of bytes is not 8.

        """
        ...

    def to_bytes(self) -> bytes:
        """Returns the native data as a byte sequence.

        Returns:
            A byte sequence representing the native data with big endian.
            The length is 8.

        """
        ...

    @classmethod
    def from_int(cls, src: int) -> Self:
        """Creates a new instance from the specified integer.

        Args:
            src: The integer from which a new instance is created.

        Returns:
            A new instance created from the specified integer.

        """
        ...

    def to_int(self) -> int:
        """Returns the native data as an integer.

        Returns:
            An integer that represents the native data.

        """
        ...

    @classmethod
    def from_f16(cls, src: Float16, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True) -> Self:
        """Creates a new instance from the specified IEEE 754 binary16 floating point.

        The result is the same as that of :func:`f16_to_ui64()`.

        Args:
            src: The IEEE 754 binary16 floating point from which a new instance is created.
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            A new instance created from the specified IEEE 754 binary16 floating point.

        """
        ...

    def to_f16(self) -> Float16:
        """Converts the 64-bit unsigned integer to an IEEE 754 binary16 floating point.

        The result is the same as that of :func:`ui64_to_f16()`.

        Returns:
            The IEEE 754 binary16 floating point.

        """
        ...

    @classmethod
    def from_f32(cls, src: Float32, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True) -> Self:
        """Creates a new instance from the specified IEEE 754 binary32 floating point.

        The result is the same as that of :func:`f32_to_ui64()`.

        Args:
            src: The IEEE 754 binary32 floating point from which a new instance is created.
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            A new instance created from the specified IEEE 754 binary32 floating point.

        """
        ...

    def to_f32(self) -> Float32:
        """Converts the 64-bit unsigned integer to an IEEE 754 binary32 floating point.

        The result is the same as that of :func:`ui64_to_f32()`.

        Returns:
            The IEEE 754 binary32 floating point.

        """
        ...

    @classmethod
    def from_f64(cls, src: Float64, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True) -> Self:
        """Creates a new instance from the specified IEEE 754 binary64 floating point.

        The result is the same as that of :func:`f64_to_ui64()`.

        Args:
            src: The IEEE 754 binary64 floating point from which a new instance is created.
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            A new instance created from the specified IEEE 754 binary64 floating point.

        """
        ...

    def to_f64(self) -> Float64:
        """Converts the 64-bit unsigned integer to an IEEE 754 binary64 floating point.

        The result is the same as that of :func:`ui64_to_f64()`.

        Returns:
            The IEEE 754 binary64 floating point.

        """
        ...

    @classmethod
    def from_f128(cls, src: Float128, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True) -> Self:
        """Creates a new instance from the specified IEEE 754 binary128 floating point.

        The result is the same as that of :func:`f128_to_ui64()`.

        Args:
            src: The IEEE 754 binary128 floating point from which a new instance is created.
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            A new instance created from the specified IEEE 754 binary128 floating point.

        """
        ...

    def to_f128(self) -> Float128:
        """Converts the 64-bit unsigned integer to an IEEE 754 binary128 floating point.

        The result is the same as that of :func:`ui64_to_f128()`.

        Returns:
            The IEEE 754 binary128 floating point.

        """
        ...

    def __str__(self) -> str:
        ...

    def __pos__(self) -> Self:
        ...

    def __neg__(self) -> Self:
        ...

    def __invert__(self) -> Self:
        ...

    def __add__(self, other: Self) -> Self:
        ...

    def __sub__(self, other: Self) -> Self:
        ...

    def __mul__(self, other: Self) -> Self:
        ...

    def __floordiv__(self, other: Self) -> Self:
        ...

    def __mod__(self, other: Self) -> Self:
        ...

    def __lshift__(self, other: Self) -> Self:
        ...

    def __rshift__(self, other: Self) -> Self:
        ...

    def __and__(self, other: Self) -> Self:
        ...

    def __or__(self, other: Self) -> Self:
        ...

    def __xor__(self, other: Self) -> Self:
        ...

    def __lt__(self, other: Self) -> bool:
        ...

    def __le__(self, other: Self) -> bool:
        ...

    def __gt__(self, other: Self) -> bool:
        ...

    def __ge__(self, other: Self) -> bool:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __iadd__(self, other: Self) -> Self:
        ...

    def __isub__(self, other: Self) -> Self:
        ...

    def __imul__(self, other: Self) -> Self:
        ...

    def __ifloordiv__(self, other: Self) -> Self:
        ...

    def __imod__(self, other: Self) -> Self:
        ...

    def __ilshift__(self, other: Self) -> Self:
        ...

    def __irshift__(self, other: Self) -> Self:
        ...

    def __iand__(self, other: Self) -> Self:
        ...

    def __ior__(self, other: Self) -> Self:
        ...

    def __ixor__(self, other: Self) -> Self:
        ...


class Int32:
    """A 32-bit signed integer.

    The object is immutable.

    The following operators are supported:

    - unary operators: ``+``, ``-``, ``~``.
    - binary operators: ``+``, ``-``, ``*``, ``//``, ``%``,
      ``<<``, ``>>``, ``&``, ``|``, ``^``, ``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``,
      ``+=``, ``-=``, ``*=``, ``//=``, ``%=``, ``<<=``, ``>>=``, ``&=``, ``|=``, ``^=``.

    The following operators are unsupported:

    - binary operators: ``**``, ``/``, ``**=``, ``/=``.

    """

    @classmethod
    def size(cls) -> int:
        """Returns the native data size in bits.

        Returns:
            The native data size in bits, i.e. 32.

        """
        ...

    @classmethod
    def from_bytes(cls, src: bytes) -> Self:
        """Creates a new instance from the specified byte sequence.

        Args:
            src: The byte sequence representing the native data with big endian.
                 The length must be 4.

        Returns:
            A new instance created from the specified byte sequence.

        Raises:
            ValueError: If the length of bytes is not 4.

        """
        ...

    def to_bytes(self) -> bytes:
        """Returns the native data as a byte sequence.

        Returns:
            A byte sequence representing the native data with big endian.
            The length is 4.

        """
        ...

    @classmethod
    def from_int(cls, src: int) -> Self:
        """Creates a new instance from the specified integer.

        Args:
            src: The integer from which a new instance is created.

        Returns:
            A new instance created from the specified integer.

        """
        ...

    def to_int(self) -> int:
        """Returns the native data as an integer.

        Returns:
            An integer that represents the native data.

        """
        ...

    @classmethod
    def from_f16(cls, src: Float16, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True) -> Self:
        """Creates a new instance from the specified IEEE 754 binary16 floating point.

        The result is the same as that of :func:`f16_to_i32()`.

        Args:
            src: The IEEE 754 binary16 floating point from which a new instance is created.
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            A new instance created from the specified IEEE 754 binary16 floating point.

        """
        ...

    def to_f16(self) -> Float16:
        """Converts the 32-bit signed integer to an IEEE 754 binary16 floating point.

        The result is the same as that of :func:`i32_to_f16()`.

        Returns:
            The IEEE 754 binary16 floating point.

        """
        ...

    @classmethod
    def from_f32(cls, src: Float32, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True) -> Self:
        """Creates a new instance from the specified IEEE 754 binary32 floating point.

        The result is the same as that of :func:`f32_to_i32()`.

        Args:
            src: The IEEE 754 binary32 floating point from which a new instance is created.
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            A new instance created from the specified IEEE 754 binary32 floating point.

        """
        ...

    def to_f32(self) -> Float32:
        """Converts the 32-bit signed integer to an IEEE 754 binary32 floating point.

        The result is the same as that of :func:`i32_to_f32()`.

        Returns:
            The IEEE 754 binary32 floating point.

        """
        ...

    @classmethod
    def from_f64(cls, src: Float64, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True) -> Self:
        """Creates a new instance from the specified IEEE 754 binary64 floating point.

        The result is the same as that of :func:`f64_to_i32()`.

        Args:
            src: The IEEE 754 binary64 floating point from which a new instance is created.
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            A new instance created from the specified IEEE 754 binary64 floating point.

        """
        ...

    def to_f64(self) -> Float64:
        """Converts the 32-bit signed integer to an IEEE 754 binary64 floating point.

        The result is the same as that of :func:`i32_to_f64()`.

        Returns:
            The IEEE 754 binary64 floating point.

        """
        ...

    @classmethod
    def from_f128(cls, src: Float128, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True) -> Self:
        """Creates a new instance from the specified IEEE 754 binary128 floating point.

        The result is the same as that of :func:`f128_to_i32()`.

        Args:
            src: The IEEE 754 binary128 floating point from which a new instance is created.
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            A new instance created from the specified IEEE 754 binary128 floating point.

        """
        ...

    def to_f128(self) -> Float128:
        """Converts the 32-bit signed integer to an IEEE 754 binary128 floating point.

        The result is the same as that of :func:`i32_to_f128()`.

        Returns:
            The IEEE 754 binary128 floating point.

        """
        ...

    def __str__(self) -> str:
        ...

    def __pos__(self) -> Self:
        ...

    def __neg__(self) -> Self:
        ...

    def __invert__(self) -> Self:
        ...

    def __add__(self, other: Self) -> Self:
        ...

    def __sub__(self, other: Self) -> Self:
        ...

    def __mul__(self, other: Self) -> Self:
        ...

    def __floordiv__(self, other: Self) -> Self:
        ...

    def __mod__(self, other: Self) -> Self:
        ...

    def __lshift__(self, other: Self) -> Self:
        ...

    def __rshift__(self, other: Self) -> Self:
        ...

    def __and__(self, other: Self) -> Self:
        ...

    def __or__(self, other: Self) -> Self:
        ...

    def __xor__(self, other: Self) -> Self:
        ...

    def __lt__(self, other: Self) -> bool:
        ...

    def __le__(self, other: Self) -> bool:
        ...

    def __gt__(self, other: Self) -> bool:
        ...

    def __ge__(self, other: Self) -> bool:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __iadd__(self, other: Self) -> Self:
        ...

    def __isub__(self, other: Self) -> Self:
        ...

    def __imul__(self, other: Self) -> Self:
        ...

    def __ifloordiv__(self, other: Self) -> Self:
        ...

    def __imod__(self, other: Self) -> Self:
        ...

    def __ilshift__(self, other: Self) -> Self:
        ...

    def __irshift__(self, other: Self) -> Self:
        ...

    def __iand__(self, other: Self) -> Self:
        ...

    def __ior__(self, other: Self) -> Self:
        ...

    def __ixor__(self, other: Self) -> Self:
        ...


class Int64:
    """A 64-bit signed integer.

    The object is immutable.

    The following operators are supported:

    - unary operators: ``+``, ``-``, ``~``.
    - binary operators: ``+``, ``-``, ``*``, ``//``, ``%``,
      ``<<``, ``>>``, ``&``, ``|``, ``^``, ``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``,
      ``+=``, ``-=``, ``*=``, ``//=``, ``%=``, ``<<=``, ``>>=``, ``&=``, ``|=``, ``^=``.

    The following operators are unsupported:

    - binary operators: ``**``, ``/``, ``**=``, ``/=``.

    """

    @classmethod
    def size(cls) -> int:
        """Returns the native data size in bits.

        Returns:
            The native data size in bits, i.e. 64.

        """
        ...

    @classmethod
    def from_bytes(cls, src: bytes) -> Self:
        """Creates a new instance from the specified byte sequence.

        Args:
            src: The byte sequence representing the native data with big endian.
                 The length must be 8.

        Returns:
            A new instance created from the specified byte sequence.

        Raises:
            ValueError: If the length of bytes is not 8.

        """
        ...

    def to_bytes(self) -> bytes:
        """Returns the native data as a byte sequence.

        Returns:
            A byte sequence representing the native data with big endian.
            The length is 8.

        """
        ...

    @classmethod
    def from_int(cls, src: int) -> Self:
        """Creates a new instance from the specified integer.

        Args:
            src: The integer from which a new instance is created.

        Returns:
            A new instance created from the specified integer.

        """
        ...

    def to_int(self) -> int:
        """Returns the native data as an integer.

        Returns:
            An integer that represents the native data.

        """
        ...

    @classmethod
    def from_f16(cls, src: Float16, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True) -> Self:
        """Creates a new instance from the specified IEEE 754 binary16 floating point.

        The result is the same as that of :func:`f16_to_i64()`.

        Args:
            src: The IEEE 754 binary16 floating point from which a new instance is created.
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            A new instance created from the specified IEEE 754 binary16 floating point.

        """
        ...

    def to_f16(self) -> Float16:
        """Converts the 64-bit signed integer to an IEEE 754 binary16 floating point.

        The result is the same as that of :func:`i64_to_f16()`.

        Returns:
            The IEEE 754 binary16 floating point.

        """
        ...

    @classmethod
    def from_f32(cls, src: Float32, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True) -> Self:
        """Creates a new instance from the specified IEEE 754 binary32 floating point.

        The result is the same as that of :func:`f32_to_i64()`.

        Args:
            src: The IEEE 754 binary32 floating point from which a new instance is created.
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            A new instance created from the specified IEEE 754 binary32 floating point.

        """
        ...

    def to_f32(self) -> Float32:
        """Converts the 64-bit signed integer to an IEEE 754 binary32 floating point.

        The result is the same as that of :func:`i64_to_f32()`.

        Returns:
            The IEEE 754 binary32 floating point.

        """
        ...

    @classmethod
    def from_f64(cls, src: Float64, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True) -> Self:
        """Creates a new instance from the specified IEEE 754 binary64 floating point.

        The result is the same as that of :func:`f64_to_i64()`.

        Args:
            src: The IEEE 754 binary64 floating point from which a new instance is created.
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            A new instance created from the specified IEEE 754 binary64 floating point.

        """
        ...

    def to_f64(self) -> Float64:
        """Converts the 64-bit signed integer to an IEEE 754 binary64 floating point.

        The result is the same as that of :func:`i64_to_f64()`.

        Returns:
            The IEEE 754 binary64 floating point.

        """
        ...

    @classmethod
    def from_f128(cls, src: Float128, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True) -> Self:
        """Creates a new instance from the specified IEEE 754 binary128 floating point.

        The result is the same as that of :func:`f128_to_i64()`.

        Args:
            src: The IEEE 754 binary128 floating point from which a new instance is created.
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            A new instance created from the specified IEEE 754 binary128 floating point.

        """
        ...

    def to_f128(self) -> Float128:
        """Converts the 64-bit signed integer to an IEEE 754 binary128 floating point.

        The result is the same as that of :func:`i64_to_f128()`.

        Returns:
            The IEEE 754 binary128 floating point.

        """
        ...

    def __str__(self) -> str:
        ...

    def __pos__(self) -> Self:
        ...

    def __neg__(self) -> Self:
        ...

    def __invert__(self) -> Self:
        ...

    def __add__(self, other: Self) -> Self:
        ...

    def __sub__(self, other: Self) -> Self:
        ...

    def __mul__(self, other: Self) -> Self:
        ...

    def __floordiv__(self, other: Self) -> Self:
        ...

    def __mod__(self, other: Self) -> Self:
        ...

    def __lshift__(self, other: Self) -> Self:
        ...

    def __rshift__(self, other: Self) -> Self:
        ...

    def __and__(self, other: Self) -> Self:
        ...

    def __or__(self, other: Self) -> Self:
        ...

    def __xor__(self, other: Self) -> Self:
        ...

    def __lt__(self, other: Self) -> bool:
        ...

    def __le__(self, other: Self) -> bool:
        ...

    def __gt__(self, other: Self) -> bool:
        ...

    def __ge__(self, other: Self) -> bool:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __iadd__(self, other: Self) -> Self:
        ...

    def __isub__(self, other: Self) -> Self:
        ...

    def __imul__(self, other: Self) -> Self:
        ...

    def __ifloordiv__(self, other: Self) -> Self:
        ...

    def __imod__(self, other: Self) -> Self:
        ...

    def __ilshift__(self, other: Self) -> Self:
        ...

    def __irshift__(self, other: Self) -> Self:
        ...

    def __iand__(self, other: Self) -> Self:
        ...

    def __ior__(self, other: Self) -> Self:
        ...

    def __ixor__(self, other: Self) -> Self:
        ...


class BFloat16:
    """A 16-bit brain floating point.

    The object is immutable.

    No operators are supported.

    """

    @classmethod
    def size(cls) -> int:
        """Returns the native data size in bits.

        Returns:
            The native data size in bits, i.e. 16.

        """
        ...

    @classmethod
    def from_bytes(cls, src: bytes) -> Self:
        """Creates a new instance from the specified byte sequence.

        Args:
            src: The byte sequence representing the native data with big endian.
                 The length must be 2.

        Returns:
            A new instance created from the specified byte sequence.

        Raises:
            ValueError: If the length of bytes is not 2.

        """
        ...

    def to_bytes(self) -> bytes:
        """Returns the native data as a byte sequence.

        Returns:
            A byte sequence representing the native data with big endian.
            The length is 2.

        """
        ...

    @classmethod
    def from_float(cls, src: float) -> Self:
        """Creates a new instance from the specified floating point.

        Args:
            src: The floating point from which a new instance is created.

        Returns:
            A new instance created from the specified floating point.

        """
        ...

    def to_float(self) -> float:
        """Returns the native data as a floating point.

        Returns:
            A floating point that represents the native data.

        """
        ...

    @classmethod
    def from_f32(cls, src: Float32) -> BFloat16:
        """Creates a new instance from the specified IEEE 754 binary32 floating point.

        The result is the same as that of :func:`f32_to_bf16()`.

        Args:
            src: The IEEE 754 binary32 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary32 floating point.

        """
        ...

    def to_f32(self) -> Float32:
        """Converts the 16-bit brain floating point to an IEEE 754 binary32 floating point.

        The result is the same as that of :func:`bf16_to_f32()`.

        Returns:
            The IEEE 754 binary32 floating point.

        """
        ...

    def is_signaling_nan(self) -> bool:
        """Tests if the 16-bit brain floating point is a signaling NaN.

        The result is the same as that of :func:`bf16_is_signaling_nan()`.

        Returns:
            ``True`` if the floating point is a signaling NaN, ``False`` otherwise.

        """
        ...

    def is_nan(self) -> bool:
        """Tests if the 16-bit brain floating point is a NaN.

        The result is the same as that of :func:`bf16_is_nan()`.

        Returns:
            ``True`` if the floating point is a NaN, ``False`` otherwise.

        """
        ...

    def is_inf(self) -> bool:
        """Tests if the 16-bit brain floating point is an infinity.

        The result is the same as that of :func:`bf16_is_inf()`.

        Returns:
            ``True`` if the floating point is an infinity, ``False`` otherwise.

        """
        ...

    def __str__(self) -> str:
        ...


class Float16:
    """An IEEE 754 binary16 floating point.

    The object is immutable.

    The following operators are supported:

    - unary operators: ``+``, ``-``.
    - binary operators: ``+``, ``-``, ``*``, ``/``, ``//``, ``%``,
      ``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``,
      ``+=``, ``-=``, ``*=``, ``/=``, ``//=``, ``%=``.

    The following operators are unsupported:

    - unary operator: ``~``.
    - binary operators: ``<<``, ``>>``, ``&``, ``|``, ``^``, ``**``,
      ``<<=``, ``>>=``, ``&=``, ``|=``, ``^=``, ``**=``.

    """

    @classmethod
    def size(cls) -> int:
        """Returns the native data size in bits.

        Returns:
            The native data size in bits, i.e. 16.

        """
        ...

    @classmethod
    def from_bytes(cls, src: bytes) -> Self:
        """Creates a new instance from the specified byte sequence.

        Args:
            src: The byte sequence representing the native data with big endian.
                 The length must be 2.

        Returns:
            A new instance created from the specified byte sequence.

        Raises:
            ValueError: If the length of bytes is not 2.

        """
        ...

    def to_bytes(self) -> bytes:
        """Returns the native data as a byte sequence.

        Returns:
            A byte sequence representing the native data with big endian.
            The length is 2.

        """
        ...

    @classmethod
    def from_float(cls, src: float) -> Self:
        """Creates a new instance from the specified floating point.

        Args:
            src: The floating point from which a new instance is created.

        Returns:
            A new instance created from the specified floating point.

        """
        ...

    def to_float(self) -> float:
        """Returns the native data as a floating point.

        Returns:
            A floating point that represents the native data.

        """
        ...

    @classmethod
    def from_ui32(cls, src: UInt32) -> Self:
        """Creates a new instance from the specified 32-bit unsigned integer.

        The result is the same as that of :func:`ui32_to_f16()`.

        Args:
            src: The 32-bit unsigned integer from which a new instance is created.

        Returns:
            A new instance created from the specified 32-bit unsigned integer.

        """
        ...

    def to_ui32(
        self, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
    ) -> UInt32:
        """Converts the IEEE 754 binary16 floating point to a 32-bit unsigned integer.

        The result is the same as that of :func:`f16_to_ui32()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 32-bit unsigned integer.

        """
        ...

    @classmethod
    def from_ui64(cls, src: UInt64) -> Self:
        """Creates a new instance from the specified 64-bit unsigned integer.

        The result is the same as that of :func:`ui64_to_f16()`.

        Args:
            src: The 64-bit unsigned integer from which a new instance is created.

        Returns:
            A new instance created from the specified 64-bit unsigned integer.

        """
        ...

    def to_ui64(
        self, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
    ) -> UInt64:
        """Converts the IEEE 754 binary16 floating point to a 64-bit unsigned integer.

        The result is the same as that of :func:`f16_to_ui64()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 64-bit unsigned integer.

        """
        ...

    @classmethod
    def from_i32(cls, src: Int32) -> Self:
        """Creates a new instance from the specified 32-bit signed integer.

        The result is the same as that of :func:`i32_to_f16()`.

        Args:
            src: The 32-bit signed integer from which a new instance is created.

        Returns:
            A new instance created from the specified 32-bit signed integer.

        """
        ...

    def to_i32(
        self, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
    ) -> Int32:
        """Converts the IEEE 754 binary16 floating point to a 32-bit signed integer.

        The result is the same as that of :func:`f16_to_i32()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 32-bit signed integer.

        """
        ...

    @classmethod
    def from_i64(cls, src: Int64) -> Self:
        """Creates a new instance from the specified 64-bit signed integer.

        The result is the same as that of :func:`i64_to_f16()`.

        Args:
            src: The 64-bit signed integer from which a new instance is created.

        Returns:
            A new instance created from the specified 64-bit signed integer.

        """
        ...

    def to_i64(
        self, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
    ) -> Int64:
        """Converts the IEEE 754 binary16 floating point to a 64-bit signed integer.

        The result is the same as that of :func:`f16_to_i64()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 64-bit signed integer.

        """
        ...

    @classmethod
    def from_f16(cls, src: Float16) -> Float16:
        """Creates a new instance from the specified IEEE 754 binary16 floating point.

        The result is a copy of the specified instance.

        Args:
            src: The IEEE 754 binary16 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary16 floating point.

        """
        ...

    def to_f16(self) -> Float16:
        """Converts the IEEE 754 binary16 floating point to a binary16 floating point.

        The result is a copy.

        Returns:
            The IEEE 754 binary16 floating point.

        """
        ...

    @classmethod
    def from_f32(cls, src: Float32) -> Self:
        """Creates a new instance from the specified IEEE 754 binary32 floating point.

        The result is the same as that of :func:`f32_to_f16()`.

        Args:
            src: The IEEE 754 binary32 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary32 floating point.

        """
        ...

    def to_f32(self) -> Float32:
        """Converts the IEEE 754 binary16 floating point to a binary32 floating point.

        The result is the same as that of :func:`f16_to_f32()`.

        Returns:
            The IEEE 754 binary32 floating point.

        """
        ...

    @classmethod
    def from_f64(cls, src: Float64) -> Self:
        """Creates a new instance from the specified IEEE 754 binary64 floating point.

        The result is the same as that of :func:`f64_to_f16()`.

        Args:
            src: The IEEE 754 binary64 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary64 floating point.

        """
        ...

    def to_f64(self) -> Float64:
        """Converts the IEEE 754 binary16 floating point to a binary64 floating point.

        The result is the same as that of :func:`f16_to_f64()`.

        Returns:
            The IEEE 754 binary64 floating point.

        """
        ...

    @classmethod
    def from_f128(cls, src: Float128) -> Self:
        """Creates a new instance from the specified IEEE 754 binary128 floating point.

        The result is the same as that of :func:`f128_to_f16()`.

        Args:
            src: The IEEE 754 binary128 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary128 floating point.

        """
        ...

    def to_f128(self) -> Float128:
        """Converts the IEEE 754 binary16 floating point to a binary128 floating point.

        The result is the same as that of :func:`f16_to_f128()`.

        Returns:
            The IEEE 754 binary128 floating point.

        """
        ...

    def round_to_int(
        self, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
    ) -> Self:
        """Rounds the number.

        The result is the same as that of :func:`f16_round_to_int()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact rounding is unable.

        Returns:
            The resulted integer.

        """
        ...

    def neg(self) -> Self:
        """Negates the IEEE 754 binary16 floating point.

        The result is the same as that of :func:`f16_neg()`.

        Returns:
            The resulted number (``-x``).

        """
        ...

    @classmethod
    def add(cls, x: Self, y: Self) -> Self:
        """Adds the IEEE 754 binary16 floating points.

        The result is the same as that of :func:`f16_add()`.

        Args:
            x: The floating point to be added.
            y: The floating point to add.

        Returns:
            The resulted number (``x + y``).

        """
        ...

    @classmethod
    def sub(cls, x: Self, y: Self) -> Self:
        """Subtracts the IEEE 754 binary16 floating points.

        The result is the same as that of :func:`f16_sub()`.

        Args:
            x: The floating point to be subtracted.
            y: The floating point to subtract.

        Returns:
            The resulted number (``x - y``).

        """
        ...

    @classmethod
    def mul(cls, x: Self, y: Self) -> Self:
        """Multiplies the IEEE 754 binary16 floating points.

        The result is the same as that of :func:`f16_mul()`.

        Args:
            x: The floating point to be multiplied.
            y: The floating point to multiply.

        Returns:
            The resulted number (``x * y``).

        """
        ...

    @classmethod
    def mul_add(cls, x: Self, y: Self, z: Self) -> Self:
        """Multiplies and Adds the IEEE 754 binary16 floating points.

        The result is the same as that of :func:`f16_mul_add()`.

        Args:
            x: The floating point to be multiplied.
            y: The floating point to multiply.
            z: The floating point to add.

        Returns:
            The resulted number (``x * y + z``).

        """
        ...

    @classmethod
    def div(cls, x: Self, y: Self) -> Self:
        """Divides the IEEE 754 binary16 floating points.

        The result is the same as that of :func:`f16_div()`.

        Args:
            x: The floating point to be divided.
            y: The floating point to divide.

        Returns:
            The resulted number (``x / y``).

        """
        ...

    @classmethod
    def rem(cls, x: Self, y: Self) -> Self:
        """Calculates a remainder by dividing the IEEE 754 binary16 floating points.

        The result is the same as that of :func:`f16_rem()`.

        Args:
            x: The floating point to be divided.
            y: The floating point to divide.

        Returns:
            The resulted number (``x % y``).

        """
        ...

    def sqrt(self) -> Self:
        """Calculates a square root of the IEEE 754 binary16 floating point.

        The result is the same as that of :func:`f16_sqrt()`.

        Returns:
            The resulted number (``sqrt(x)``).

        """
        ...

    @classmethod
    def eq(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is equal to the second one expressed as IEEE 754 binary16 floating points.

        The result is the same as that of :func:`f16_eq()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

        """
        ...

    @classmethod
    def le(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary16 floating points.

        The result is the same as that of :func:`f16_le()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

        """
        ...

    @classmethod
    def lt(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is less than the second one expressed as IEEE 754 binary16 floating points.

        The result is the same as that of :func:`f16_lt()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

        """
        ...

    @classmethod
    def eq_signaling(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is equal to the second one expressed as IEEE 754 binary16 floating points.

        The invalid exception flag is set for any NaN input, not just for signaling NaNs.

        The result is the same as that of :func:`f16_eq_signaling()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

        """
        ...

    @classmethod
    def le_quiet(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary16 floating points.

        The invalid exception flag is not set for quiet NaNs.

        The result is the same as that of :func:`f16_le_quiet()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

        """
        ...

    @classmethod
    def lt_quiet(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is less than the second one expressed as IEEE 754 binary16 floating points.

        The invalid exception flag is not set for quiet NaNs.

        The result is the same as that of :func:`f16_lt_quiet()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

        """
        ...

    def is_signaling_nan(self) -> bool:
        """Tests if the IEEE 754 binary16 floating point is a signaling NaN.

        The result is the same as that of :func:`f16_is_signaling_nan()`.

        Returns:
            ``True`` if the floating point is a signaling NaN, ``False`` otherwise.

        """
        ...

    def is_nan(self) -> bool:
        """Tests if the IEEE 754 binary16 floating point is a NaN.

        The result is the same as that of :func:`f16_is_nan()`.

        Returns:
            ``True`` if the floating point is a NaN, ``False`` otherwise.

        """
        ...

    def is_inf(self) -> bool:
        """Tests if the IEEE 754 binary16 floating point is an infinity.

        The result is the same as that of :func:`f16_is_inf()`.

        Returns:
            ``True`` if the floating point is an infinity, ``False`` otherwise.

        """
        ...

    def __str__(self) -> str:
        ...

    def __pos__(self) -> Self:
        ...

    def __neg__(self) -> Self:
        ...

    def __add__(self, other: Self) -> Self:
        ...

    def __sub__(self, other: Self) -> Self:
        ...

    def __mul__(self, other: Self) -> Self:
        ...

    def __truediv__(self, other: Self) -> Self:
        ...

    def __floordiv__(self, other: Self) -> Self:
        ...

    def __mod__(self, other: Self) -> Self:
        ...

    def __lt__(self, other: Self) -> bool:
        ...

    def __le__(self, other: Self) -> bool:
        ...

    def __gt__(self, other: Self) -> bool:
        ...

    def __ge__(self, other: Self) -> bool:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __iadd__(self, other: Self) -> Self:
        ...

    def __isub__(self, other: Self) -> Self:
        ...

    def __imul__(self, other: Self) -> Self:
        ...

    def __itruediv__(self, other: Self) -> Self:
        ...

    def __ifloordiv__(self, other: Self) -> Self:
        ...

    def __imod__(self, other: Self) -> Self:
        ...


class Float32:
    """An IEEE 754 binary32 floating point.

    The object is immutable.

    The following operators are supported:

    - unary operators: ``+``, ``-``.
    - binary operators: ``+``, ``-``, ``*``, ``/``, ``//``, ``%``,
      ``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``,
      ``+=``, ``-=``, ``*=``, ``/=``, ``//=``, ``%=``.

    The following operators are unsupported:

    - unary operator: ``~``.
    - binary operators: ``<<``, ``>>``, ``&``, ``|``, ``^``, ``**``,
      ``<<=``, ``>>=``, ``&=``, ``|=``, ``^=``, ``**=``.

    """

    @classmethod
    def size(cls) -> int:
        """Returns the native data size in bits.

        Returns:
            The native data size in bits, i.e. 32.

        """
        ...

    @classmethod
    def from_bytes(cls, src: bytes) -> Self:
        """Creates a new instance from the specified byte sequence.

        Args:
            src: The byte sequence representing the native data with big endian.
                 The length must be 4.

        Returns:
            A new instance created from the specified byte sequence.

        Raises:
            ValueError: If the length of bytes is not 4.

        """
        ...

    def to_bytes(self) -> bytes:
        """Returns the native data as a byte sequence.

        Returns:
            A byte sequence representing the native data with big endian.
            The length is 4.

        """
        ...

    @classmethod
    def from_float(cls, src: float) -> Self:
        """Creates a new instance from the specified floating point.

        Args:
            src: The floating point from which a new instance is created.

        Returns:
            A new instance created from the specified floating point.

        """
        ...

    def to_float(self) -> float:
        """Returns the native data as a floating point.

        Returns:
            A floating point that represents the native data.

        """
        ...

    @classmethod
    def from_bf16(cls, src: BFloat16) -> Self:
        """Creates a new instance from the 16-bit brain floating point.

        The result is the same as that of :func:`bf16_to_f32()`.

        Args:
            src: The 16-bit brain floating point from which a new instance is created.

        Returns:
            A new instance created from the specified 16-bit brain floating point.

        """
        ...

    def to_bf16(self) -> BFloat16:
        """Converts the IEEE 754 binary32 floating point to a 16-bit brain floating point.

        The result is the same as that of :func:`f32_to_bf16()`.

        Returns:
            The 16-bit brain floating point.

        """
        ...

    @classmethod
    def from_ui32(cls, src: UInt32) -> Self:
        """Creates a new instance from the specified 32-bit unsigned integer.

        The result is the same as that of :func:`ui32_to_f32()`.

        Args:
            src: The 32-bit unsigned integer from which a new instance is created.

        Returns:
            A new instance created from the specified 32-bit unsigned integer.

        """
        ...

    def to_ui32(
        self, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
    ) -> UInt32:
        """Converts the IEEE 754 binary32 floating point to a 32-bit unsigned integer.

        The result is the same as that of :func:`f32_to_ui32()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 32-bit unsigned integer.

        """
        ...

    @classmethod
    def from_ui64(cls, src: UInt64) -> Self:
        """Creates a new instance from the specified 64-bit unsigned integer.

        The result is the same as that of :func:`ui64_to_f32()`.

        Args:
            src: The 64-bit unsigned integer from which a new instance is created.

        Returns:
            A new instance created from the specified 64-bit unsigned integer.

        """
        ...

    def to_ui64(
        self, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
    ) -> UInt64:
        """Converts the IEEE 754 binary32 floating point to a 64-bit unsigned integer.

        The result is the same as that of :func:`f32_to_ui64()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 64-bit unsigned integer.

        """
        ...

    @classmethod
    def from_i32(cls, src: Int32) -> Self:
        """Creates a new instance from the specified 32-bit signed integer.

        The result is the same as that of :func:`i32_to_f32()`.

        Args:
            src: The 32-bit signed integer from which a new instance is created.

        Returns:
            A new instance created from the specified 32-bit signed integer.

        """
        ...

    def to_i32(
        self, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
    ) -> Int32:
        """Converts the IEEE 754 binary32 floating point to a 32-bit signed integer.

        The result is the same as that of :func:`f32_to_i32()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 32-bit signed integer.

        """
        ...

    @classmethod
    def from_i64(cls, src: Int64) -> Self:
        """Creates a new instance from the specified 64-bit signed integer.

        The result is the same as that of :func:`i64_to_f32()`.

        Args:
            src: The 64-bit signed integer from which a new instance is created.

        Returns:
            A new instance created from the specified 64-bit signed integer.

        """
        ...

    def to_i64(
        self, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
    ) -> Int64:
        """Converts the IEEE 754 binary32 floating point to a 64-bit signed integer.

        The result is the same as that of :func:`f32_to_i64()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 64-bit signed integer.

        """
        ...

    @classmethod
    def from_f16(cls, src: Float16) -> Self:
        """Creates a new instance from the specified IEEE 754 binary16 floating point.

        The result is the same as that of :func:`f16_to_f32()`.

        Args:
            src: The IEEE 754 binary16 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary16 floating point.

        """
        ...

    def to_f16(self) -> Float16:
        """Converts the IEEE 754 binary32 floating point to a binary16 floating point.

        The result is the same as that of :func:`f32_to_f16()`.

        Returns:
            The IEEE 754 binary16 floating point.

        """
        ...

    @classmethod
    def from_f32(cls, src: Float32) -> Float32:
        """Creates a new instance from the specified IEEE 754 binary32 floating point.

        The result is a copy of the specified instance.

        Args:
            src: The IEEE 754 binary32 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary32 floating point.

        """
        ...

    def to_f32(self) -> Float32:
        """Converts the IEEE 754 binary32 floating point to a binary32 floating point.

        The result is a copy.

        Returns:
            The IEEE 754 binary32 floating point.

        """
        ...

    @classmethod
    def from_f64(cls, src: Float64) -> Self:
        """Creates a new instance from the specified IEEE 754 binary64 floating point.

        The result is the same as that of :func:`f64_to_f32()`.

        Args:
            src: The IEEE 754 binary64 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary64 floating point.

        """
        ...

    def to_f64(self) -> Float64:
        """Converts the IEEE 754 binary32 floating point to a binary64 floating point.

        The result is the same as that of :func:`f32_to_f64()`.

        Returns:
            The IEEE 754 binary64 floating point.

        """
        ...

    @classmethod
    def from_f128(cls, src: Float128) -> Self:
        """Creates a new instance from the specified IEEE 754 binary128 floating point.

        The result is the same as that of :func:`f128_to_f32()`.

        Args:
            src: The IEEE 754 binary128 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary128 floating point.

        """
        ...

    def to_f128(self) -> Float128:
        """Converts the IEEE 754 binary32 floating point to a binary128 floating point.

        The result is the same as that of :func:`f32_to_f128()`.

        Returns:
            The IEEE 754 binary128 floating point.

        """
        ...

    def round_to_int(
        self, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
    ) -> Self:
        """Rounds the number.

        The result is the same as that of :func:`f32_round_to_int()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact rounding is unable.

        Returns:
            The resulted integer.

        """
        ...

    def neg(self) -> Self:
        """Negates the IEEE 754 binary32 floating point.

        The result is the same as that of :func:`f32_neg()`.

        Returns:
            The resulted number (``-x``).

        """
        ...

    @classmethod
    def add(cls, x: Self, y: Self) -> Self:
        """Adds the IEEE 754 binary32 floating points.

        The result is the same as that of :func:`f32_add()`.

        Args:
            x: The floating point to be added.
            y: The floating point to add.

        Returns:
            The resulted number (``x + y``).

        """
        ...

    @classmethod
    def sub(cls, x: Self, y: Self) -> Self:
        """Subtracts the IEEE 754 binary32 floating points.

        The result is the same as that of :func:`f32_sub()`.

        Args:
            x: The floating point to be subtracted.
            y: The floating point to subtract.

        Returns:
            The resulted number (``x - y``).

        """
        ...

    @classmethod
    def mul(cls, x: Self, y: Self) -> Self:
        """Multiplies the IEEE 754 binary32 floating points.

        The result is the same as that of :func:`f32_mul()`.

        Args:
            x: The floating point to be multiplied.
            y: The floating point to multiply.

        Returns:
            The resulted number (``x * y``).

        """
        ...

    @classmethod
    def mul_add(cls, x: Self, y: Self, z: Self) -> Self:
        """Multiplies and Adds the IEEE 754 binary32 floating points.

        The result is the same as that of :func:`f32_mul_add()`.

        Args:
            x: The floating point to be multiplied.
            y: The floating point to multiply.
            z: The floating point to add.

        Returns:
            The resulted number (``x * y + z``).

        """
        ...

    @classmethod
    def div(cls, x: Self, y: Self) -> Self:
        """Divides the IEEE 754 binary32 floating points.

        The result is the same as that of :func:`f32_div()`.

        Args:
            x: The floating point to be divided.
            y: The floating point to divide.

        Returns:
            The resulted number (``x / y``).

        """
        ...

    @classmethod
    def rem(cls, x: Self, y: Self) -> Self:
        """Calculates a remainder by dividing the IEEE 754 binary32 floating points.

        The result is the same as that of :func:`f32_rem()`.

        Args:
            x: The floating point to be divided.
            y: The floating point to divide.

        Returns:
            The resulted number (``x % y``).

        """
        ...

    def sqrt(self) -> Self:
        """Calculates a square root of the IEEE 754 binary32 floating point.

        The result is the same as that of :func:`f32_sqrt()`.

        Returns:
            The resulted number (``sqrt(x)``).

        """
        ...

    @classmethod
    def eq(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is equal to the second one expressed as IEEE 754 binary32 floating points.

        The result is the same as that of :func:`f32_eq()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

        """
        ...

    @classmethod
    def le(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary32 floating points.

        The result is the same as that of :func:`f32_le()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

        """
        ...

    @classmethod
    def lt(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is less than the second one expressed as IEEE 754 binary32 floating points.

        The result is the same as that of :func:`f32_lt()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

        """
        ...

    @classmethod
    def eq_signaling(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is equal to the second one expressed as IEEE 754 binary32 floating points.

        The invalid exception flag is set for any NaN input, not just for signaling NaNs.

        The result is the same as that of :func:`f32_eq_signaling()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

        """
        ...

    @classmethod
    def le_quiet(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary32 floating points.

        The invalid exception flag is not set for quiet NaNs.

        The result is the same as that of :func:`f32_le_quiet()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

        """
        ...

    @classmethod
    def lt_quiet(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is less than the second one expressed as IEEE 754 binary32 floating points.

        The invalid exception flag is not set for quiet NaNs.

        The result is the same as that of :func:`f32_lt_quiet()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

        """
        ...

    def is_signaling_nan(self) -> bool:
        """Tests if the IEEE 754 binary32 floating point is a signaling NaN.

        The result is the same as that of :func:`f32_is_signaling_nan()`.

        Returns:
            ``True`` if the floating point is a signaling NaN, ``False`` otherwise.

        """
        ...

    def is_nan(self) -> bool:
        """Tests if the IEEE 754 binary32 floating point is a NaN.

        The result is the same as that of :func:`f32_is_nan()`.

        Returns:
            ``True`` if the floating point is a NaN, ``False`` otherwise.

        """
        ...

    def is_inf(self) -> bool:
        """Tests if the IEEE 754 binary32 floating point is an infinity.

        The result is the same as that of :func:`f32_is_inf()`.

        Returns:
            ``True`` if the floating point is an infinity, ``False`` otherwise.

        """
        ...

    def __str__(self) -> str:
        ...

    def __pos__(self) -> Self:
        ...

    def __neg__(self) -> Self:
        ...

    def __add__(self, other: Self) -> Self:
        ...

    def __sub__(self, other: Self) -> Self:
        ...

    def __mul__(self, other: Self) -> Self:
        ...

    def __truediv__(self, other: Self) -> Self:
        ...

    def __floordiv__(self, other: Self) -> Self:
        ...

    def __mod__(self, other: Self) -> Self:
        ...

    def __lt__(self, other: Self) -> bool:
        ...

    def __le__(self, other: Self) -> bool:
        ...

    def __gt__(self, other: Self) -> bool:
        ...

    def __ge__(self, other: Self) -> bool:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __iadd__(self, other: Self) -> Self:
        ...

    def __isub__(self, other: Self) -> Self:
        ...

    def __imul__(self, other: Self) -> Self:
        ...

    def __itruediv__(self, other: Self) -> Self:
        ...

    def __ifloordiv__(self, other: Self) -> Self:
        ...

    def __imod__(self, other: Self) -> Self:
        ...


class Float64:
    """An IEEE 754 binary64 floating point.

    The object is immutable.

    The following operators are supported:

    - unary operators: ``+``, ``-``.
    - binary operators: ``+``, ``-``, ``*``, ``/``, ``//``, ``%``,
      ``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``,
      ``+=``, ``-=``, ``*=``, ``/=``, ``//=``, ``%=``.

    The following operators are unsupported:

    - unary operator: ``~``.
    - binary operators: ``<<``, ``>>``, ``&``, ``|``, ``^``, ``**``,
      ``<<=``, ``>>=``, ``&=``, ``|=``, ``^=``, ``**=``.

    """

    @classmethod
    def size(cls) -> int:
        """Returns the native data size in bits.

        Returns:
            The native data size in bits, i.e. 64.

        """
        ...

    @classmethod
    def from_bytes(cls, src: bytes) -> Self:
        """Creates a new instance from the specified byte sequence.

        Args:
            src: The byte sequence representing the native data with big endian.
                 The length must be 8.

        Returns:
            A new instance created from the specified byte sequence.

        Raises:
            ValueError: If the length of bytes is not 8.

        """
        ...

    def to_bytes(self) -> bytes:
        """Returns the native data as a byte sequence.

        Returns:
            A byte sequence representing the native data with big endian.
            The length is 8.

        """
        ...

    @classmethod
    def from_float(cls, src: float) -> Self:
        """Creates a new instance from the specified floating point.

        Args:
            src: The floating point from which a new instance is created.

        Returns:
            A new instance created from the specified floating point.

        """
        ...

    def to_float(self) -> float:
        """Returns the native data as a floating point.

        Returns:
            A floating point that represents the native data.

        """
        ...

    @classmethod
    def from_ui32(cls, src: UInt32) -> Self:
        """Creates a new instance from the specified 32-bit unsigned integer.

        The result is the same as that of :func:`ui32_to_f64()`.

        Args:
            src: The 32-bit unsigned integer from which a new instance is created.

        Returns:
            A new instance created from the specified 32-bit unsigned integer.

        """
        ...

    def to_ui32(
        self, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
    ) -> UInt32:
        """Converts the IEEE 754 binary64 floating point to a 32-bit unsigned integer.

        The result is the same as that of :func:`f64_to_ui32()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 32-bit unsigned integer.

        """
        ...

    @classmethod
    def from_ui64(cls, src: UInt64) -> Self:
        """Creates a new instance from the specified 64-bit unsigned integer.

        The result is the same as that of :func:`ui64_to_f64()`.

        Args:
            src: The 64-bit unsigned integer from which a new instance is created.

        Returns:
            A new instance created from the specified 64-bit unsigned integer.

        """
        ...

    def to_ui64(
        self, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
    ) -> UInt64:
        """Converts the IEEE 754 binary64 floating point to a 64-bit unsigned integer.

        The result is the same as that of :func:`f64_to_ui64()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 64-bit unsigned integer.

        """
        ...

    @classmethod
    def from_i32(cls, src: Int32) -> Self:
        """Creates a new instance from the specified 32-bit signed integer.

        The result is the same as that of :func:`i32_to_f64()`.

        Args:
            src: The 32-bit signed integer from which a new instance is created.

        Returns:
            A new instance created from the specified 32-bit signed integer.

        """
        ...

    def to_i32(
        self, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
    ) -> Int32:
        """Converts the IEEE 754 binary64 floating point to a 32-bit signed integer.

        The result is the same as that of :func:`f64_to_i32()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 32-bit signed integer.

        """
        ...

    @classmethod
    def from_i64(cls, src: Int64) -> Self:
        """Creates a new instance from the specified 64-bit signed integer.

        The result is the same as that of :func:`i64_to_f64()`.

        Args:
            src: The 64-bit signed integer from which a new instance is created.

        Returns:
            A new instance created from the specified 64-bit signed integer.

        """
        ...

    def to_i64(
        self, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
    ) -> Int64:
        """Converts the IEEE 754 binary64 floating point to a 64-bit signed integer.

        The result is the same as that of :func:`f64_to_i64()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 64-bit signed integer.

        """
        ...

    @classmethod
    def from_f16(cls, src: Float16) -> Self:
        """Creates a new instance from the specified IEEE 754 binary16 floating point.

        The result is the same as that of :func:`f16_to_f64()`.

        Args:
            src: The IEEE 754 binary16 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary16 floating point.

        """
        ...

    def to_f16(self) -> Float16:
        """Converts the IEEE 754 binary64 floating point to a binary16 floating point.

        The result is the same as that of :func:`f64_to_f16()`.

        Returns:
            The IEEE 754 binary16 floating point.

        """
        ...

    @classmethod
    def from_f32(cls, src: Float32) -> Self:
        """Creates a new instance from the specified IEEE 754 binary32 floating point.

        The result is the same as that of :func:`f32_to_f64()`.

        Args:
            src: The IEEE 754 binary32 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary32 floating point.

        """
        ...

    def to_f32(self) -> Float32:
        """Converts the IEEE 754 binary64 floating point to a binary32 floating point.

        The result is the same as that of :func:`f64_to_f32()`.

        Returns:
            The IEEE 754 binary32 floating point.

        """
        ...

    @classmethod
    def from_f64(cls, src: Float64) -> Float64:
        """Creates a new instance from the specified IEEE 754 binary64 floating point.

        The result is a copy of the specified instance.

        Args:
            src: The IEEE 754 binary64 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary64 floating point.

        """
        ...

    def to_f64(self) -> Float64:
        """Converts the IEEE 754 binary64 floating point to a binary64 floating point.

        The result is a copy.

        Returns:
            The IEEE 754 binary64 floating point.

        """
        ...

    @classmethod
    def from_f128(cls, src: Float128) -> Self:
        """Creates a new instance from the specified IEEE 754 binary128 floating point.

        The result is the same as that of :func:`f128_to_f64()`.

        Args:
            src: The IEEE 754 binary128 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary128 floating point.

        """
        ...

    def to_f128(self) -> Float128:
        """Converts the IEEE 754 binary64 floating point to a binary128 floating point.

        The result is the same as that of :func:`f64_to_f128()`.

        Returns:
            The IEEE 754 binary128 floating point.

        """
        ...

    def round_to_int(
        self, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
    ) -> Self:
        """Rounds the number.

        The result is the same as that of :func:`f64_round_to_int()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact rounding is unable.

        Returns:
            The resulted integer.

        """
        ...

    def neg(self) -> Self:
        """Negates the IEEE 754 binary64 floating point.

        The result is the same as that of :func:`f64_neg()`.

        Returns:
            The resulted number (``-x``).

        """
        ...

    @classmethod
    def add(cls, x: Self, y: Self) -> Self:
        """Adds the IEEE 754 binary64 floating points.

        The result is the same as that of :func:`f64_add()`.

        Args:
            x: The floating point to be added.
            y: The floating point to add.

        Returns:
            The resulted number (``x + y``).

        """
        ...

    @classmethod
    def sub(cls, x: Self, y: Self) -> Self:
        """Subtracts the IEEE 754 binary64 floating points.

        The result is the same as that of :func:`f64_sub()`.

        Args:
            x: The floating point to be subtracted.
            y: The floating point to subtract.

        Returns:
            The resulted number (``x - y``).

        """
        ...

    @classmethod
    def mul(cls, x: Self, y: Self) -> Self:
        """Multiplies the IEEE 754 binary64 floating points.

        The result is the same as that of :func:`f64_mul()`.

        Args:
            x: The floating point to be multiplied.
            y: The floating point to multiply.

        Returns:
            The resulted number (``x * y``).

        """
        ...

    @classmethod
    def mul_add(cls, x: Self, y: Self, z: Self) -> Self:
        """Multiplies and Adds the IEEE 754 binary64 floating points.

        The result is the same as that of :func:`f64_mul_add()`.

        Args:
            x: The floating point to be multiplied.
            y: The floating point to multiply.
            z: The floating point to add.

        Returns:
            The resulted number (``x * y + z``).

        """
        ...

    @classmethod
    def div(cls, x: Self, y: Self) -> Self:
        """Divides the IEEE 754 binary64 floating points.

        The result is the same as that of :func:`f64_div()`.

        Args:
            x: The floating point to be divided.
            y: The floating point to divide.

        Returns:
            The resulted number (``x / y``).

        """
        ...

    @classmethod
    def rem(cls, x: Self, y: Self) -> Self:
        """Calculates a remainder by dividing the IEEE 754 binary64 floating points.

        The result is the same as that of :func:`f64_rem()`.

        Args:
            x: The floating point to be divided.
            y: The floating point to divide.

        Returns:
            The resulted number (``x % y``).

        """
        ...

    def sqrt(self) -> Self:
        """Calculates a square root of the IEEE 754 binary64 floating point.

        The result is the same as that of :func:`f64_sqrt()`.

        Returns:
            The resulted number (``sqrt(x)``).

        """
        ...

    @classmethod
    def eq(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is equal to the second one expressed as IEEE 754 binary64 floating points.

        The result is the same as that of :func:`f64_eq()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

        """
        ...

    @classmethod
    def le(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary64 floating points.

        The result is the same as that of :func:`f64_le()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

        """
        ...

    @classmethod
    def lt(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is less than the second one expressed as IEEE 754 binary64 floating points.

        The result is the same as that of :func:`f64_lt()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

        """
        ...

    @classmethod
    def eq_signaling(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is equal to the second one expressed as IEEE 754 binary64 floating points.

        The invalid exception flag is set for any NaN input, not just for signaling NaNs.

        The result is the same as that of :func:`f64_eq_signaling()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

        """
        ...

    @classmethod
    def le_quiet(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary64 floating points.

        The invalid exception flag is not set for quiet NaNs.

        The result is the same as that of :func:`f64_le_quiet()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

        """
        ...

    @classmethod
    def lt_quiet(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is less than the second one expressed as IEEE 754 binary64 floating points.

        The invalid exception flag is not set for quiet NaNs.

        The result is the same as that of :func:`f64_lt_quiet()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

        """
        ...

    def is_signaling_nan(self) -> bool:
        """Tests if the IEEE 754 binary64 floating point is a signaling NaN.

        The result is the same as that of :func:`f64_is_signaling_nan()`.

        Returns:
            ``True`` if the floating point is a signaling NaN, ``False`` otherwise.

        """
        ...

    def is_nan(self) -> bool:
        """Tests if the IEEE 754 binary64 floating point is a NaN.

        The result is the same as that of :func:`f64_is_nan()`.

        Returns:
            ``True`` if the floating point is a NaN, ``False`` otherwise.

        """
        ...

    def is_inf(self) -> bool:
        """Tests if the IEEE 754 binary64 floating point is an infinity.

        The result is the same as that of :func:`f64_is_inf()`.

        Returns:
            ``True`` if the floating point is an infinity, ``False`` otherwise.

        """
        ...

    def __str__(self) -> str:
        ...

    def __pos__(self) -> Self:
        ...

    def __neg__(self) -> Self:
        ...

    def __add__(self, other: Self) -> Self:
        ...

    def __sub__(self, other: Self) -> Self:
        ...

    def __mul__(self, other: Self) -> Self:
        ...

    def __truediv__(self, other: Self) -> Self:
        ...

    def __floordiv__(self, other: Self) -> Self:
        ...

    def __mod__(self, other: Self) -> Self:
        ...

    def __lt__(self, other: Self) -> bool:
        ...

    def __le__(self, other: Self) -> bool:
        ...

    def __gt__(self, other: Self) -> bool:
        ...

    def __ge__(self, other: Self) -> bool:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __iadd__(self, other: Self) -> Self:
        ...

    def __isub__(self, other: Self) -> Self:
        ...

    def __imul__(self, other: Self) -> Self:
        ...

    def __itruediv__(self, other: Self) -> Self:
        ...

    def __ifloordiv__(self, other: Self) -> Self:
        ...

    def __imod__(self, other: Self) -> Self:
        ...


class Float128:
    """An IEEE 754 binary128 floating point.

    The object is immutable.

    The following operators are supported:

    - unary operators: ``+``, ``-``.
    - binary operators: ``+``, ``-``, ``*``, ``/``, ``//``, ``%``,
      ``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``,
      ``+=``, ``-=``, ``*=``, ``/=``, ``//=``, ``%=``.

    The following operators are unsupported:

    - unary operator: ``~``.
    - binary operators: ``<<``, ``>>``, ``&``, ``|``, ``^``, ``**``,
      ``<<=``, ``>>=``, ``&=``, ``|=``, ``^=``, ``**=``.

    Note:
        Currently, cannot represent the exact number as a string
        if the number is unable to be expressed as an IEEE 754 binary64 floating point.

    """

    @classmethod
    def size(cls) -> int:
        """Returns the native data size in bits.

        Returns:
            The native data size in bits, i.e. 128.

        """
        ...

    @classmethod
    def from_bytes(cls, src: bytes) -> Self:
        """Creates a new instance from the specified byte sequence.

        Args:
            src: The byte sequence representing the native data with big endian.
                 The length must be 16.

        Returns:
            A new instance created from the specified byte sequence.

        Raises:
            ValueError: If the length of bytes is not 16.

        """
        ...

    def to_bytes(self) -> bytes:
        """Returns the native data as a byte sequence.

        Returns:
            A byte sequence representing the native data with big endian.
            The length is 16.

        """
        ...

    @classmethod
    def from_float(cls, src: float) -> Self:
        """Creates a new instance from the specified floating point.

        Args:
            src: The floating point from which a new instance is created.

        Returns:
            A new instance created from the specified floating point.

        Note:
            Cannot create an instance with a number that an IEEE 754 binary64
            floating point is unable to express.

        """
        ...

    def to_float(self) -> float:
        """Returns the native data as a floating point.

        Returns:
            A floating point that represents the native data.

        Note:
            Cannot return the exact number if it is unable to be expressed
            as an IEEE 754 binary64 floating point.

        """
        ...

    @classmethod
    def from_ui32(cls, src: UInt32) -> Self:
        """Creates a new instance from the specified 32-bit unsigned integer.

        The result is the same as that of :func:`ui32_to_f128()`.

        Args:
            src: The 32-bit unsigned integer from which a new instance is created.

        Returns:
            A new instance created from the specified 32-bit unsigned integer.

        """
        ...

    def to_ui32(
        self, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
    ) -> UInt32:
        """Converts the IEEE 754 binary128 floating point to a 32-bit unsigned integer.

        The result is the same as that of :func:`f128_to_ui32()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 32-bit unsigned integer.

        """
        ...

    @classmethod
    def from_ui64(cls, src: UInt64) -> Self:
        """Creates a new instance from the specified 64-bit unsigned integer.

        The result is the same as that of :func:`ui64_to_f128()`.

        Args:
            src: The 64-bit unsigned integer from which a new instance is created.

        Returns:
            A new instance created from the specified 64-bit unsigned integer.

        """
        ...

    def to_ui64(
        self, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
    ) -> UInt64:
        """Converts the IEEE 754 binary128 floating point to a 64-bit unsigned integer.

        The result is the same as that of :func:`f128_to_ui64()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 64-bit unsigned integer.

        """
        ...

    @classmethod
    def from_i32(cls, src: Int32) -> Self:
        """Creates a new instance from the specified 32-bit signed integer.

        The result is the same as that of :func:`i32_to_f128()`.

        Args:
            src: The 32-bit signed integer from which a new instance is created.

        Returns:
            A new instance created from the specified 32-bit signed integer.

        """
        ...

    def to_i32(
        self, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
    ) -> Int32:
        """Converts the IEEE 754 binary128 floating point to a 32-bit signed integer.

        The result is the same as that of :func:`f128_to_i32()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 32-bit signed integer.

        """
        ...

    @classmethod
    def from_i64(cls, src: Int64) -> Self:
        """Creates a new instance from the specified 64-bit signed integer.

        The result is the same as that of :func:`i64_to_f128()`.

        Args:
            src: The 64-bit signed integer from which a new instance is created.

        Returns:
            A new instance created from the specified 64-bit signed integer.

        """
        ...

    def to_i64(
        self, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
    ) -> Int64:
        """Converts the IEEE 754 binary128 floating point to a 64-bit signed integer.

        The result is the same as that of :func:`f128_to_i64()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 64-bit signed integer.

        """
        ...

    @classmethod
    def from_f16(cls, src: Float16) -> Self:
        """Creates a new instance from the specified IEEE 754 binary16 floating point.

        The result is the same as that of :func:`f16_to_f128()`.

        Args:
            src: The IEEE 754 binary16 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary16 floating point.

        """
        ...

    def to_f16(self) -> Float16:
        """Converts the IEEE 754 binary128 floating point to a binary16 floating point.

        The result is the same as that of :func:`f128_to_f16()`.

        Returns:
            The IEEE 754 binary16 floating point.

        """
        ...

    @classmethod
    def from_f32(cls, src: Float32) -> Self:
        """Creates a new instance from the specified IEEE 754 binary32 floating point.

        The result is the same as that of :func:`f32_to_f128()`.

        Args:
            src: The IEEE 754 binary32 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary32 floating point.

        """
        ...

    def to_f32(self) -> Float32:
        """Converts the IEEE 754 binary128 floating point to a binary32 floating point.

        The result is the same as that of :func:`f128_to_f32()`.

        Returns:
            The IEEE 754 binary32 floating point.

        """
        ...

    @classmethod
    def from_f64(cls, src: Float64) -> Self:
        """Creates a new instance from the specified IEEE 754 binary64 floating point.

        The result is the same as that of :func:`f64_to_f128()`.

        Args:
            src: The IEEE 754 binary64 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary64 floating point.

        """
        ...

    def to_f64(self) -> Float64:
        """Converts the IEEE 754 binary128 floating point to a binary64 floating point.

        The result is the same as that of :func:`f128_to_f64()`.

        Returns:
            The IEEE 754 binary64 floating point.

        """
        ...

    @classmethod
    def from_f128(cls, src: Float128) -> Float128:
        """Creates a new instance from the specified IEEE 754 binary128 floating point.

        The result is a copy of the specified instance.

        Args:
            src: The IEEE 754 binary128 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary128 floating point.

        """
        ...

    def to_f128(self) -> Float128:
        """Converts the IEEE 754 binary128 floating point to a binary128 floating point.

        The result is a copy.

        Returns:
            The IEEE 754 binary128 floating point.

        """
        ...

    def round_to_int(
        self, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
    ) -> Self:
        """Rounds the number.

        The result is the same as that of :func:`f128_round_to_int()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact rounding is unable.

        Returns:
            The resulted integer.

        """
        ...

    def neg(self) -> Self:
        """Negates the IEEE 754 binary128 floating point.

        The result is the same as that of :func:`f128_neg()`.

        Returns:
            The resulted number (``-x``).

        """
        ...

    @classmethod
    def add(cls, x: Self, y: Self) -> Self:
        """Adds the IEEE 754 binary128 floating points.

        The result is the same as that of :func:`f128_add()`.

        Args:
            x: The floating point to be added.
            y: The floating point to add.

        Returns:
            The resulted number (``x + y``).

        """
        ...

    @classmethod
    def sub(cls, x: Self, y: Self) -> Self:
        """Subtracts the IEEE 754 binary128 floating points.

        The result is the same as that of :func:`f128_sub()`.

        Args:
            x: The floating point to be subtracted.
            y: The floating point to subtract.

        Returns:
            The resulted number (``x - y``).

        """
        ...

    @classmethod
    def mul(cls, x: Self, y: Self) -> Self:
        """Multiplies the IEEE 754 binary128 floating points.

        The result is the same as that of :func:`f128_mul()`.

        Args:
            x: The floating point to be multiplied.
            y: The floating point to multiply.

        Returns:
            The resulted number (``x * y``).

        """
        ...

    @classmethod
    def mul_add(cls, x: Self, y: Self, z: Self) -> Self:
        """Multiplies and Adds the IEEE 754 binary128 floating points.

        The result is the same as that of :func:`f128_mul_add()`.

        Args:
            x: The floating point to be multiplied.
            y: The floating point to multiply.
            z: The floating point to add.

        Returns:
            The resulted number (``x * y + z``).

        """
        ...

    @classmethod
    def div(cls, x: Self, y: Self) -> Self:
        """Divides the IEEE 754 binary128 floating points.

        The result is the same as that of :func:`f128_div()`.

        Args:
            x: The floating point to be divided.
            y: The floating point to divide.

        Returns:
            The resulted number (``x / y``).

        """
        ...

    @classmethod
    def rem(cls, x: Self, y: Self) -> Self:
        """Calculates a remainder by dividing the IEEE 754 binary128 floating points.

        The result is the same as that of :func:`f128_rem()`.

        Args:
            x: The floating point to be divided.
            y: The floating point to divide.

        Returns:
            The resulted number (``x % y``).

        """
        ...

    def sqrt(self) -> Self:
        """Calculates a square root of the IEEE 754 binary128 floating point.

        The result is the same as that of :func:`f128_sqrt()`.

        Returns:
            The resulted number (``sqrt(x)``).

        """
        ...

    @classmethod
    def eq(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is equal to the second one expressed as IEEE 754 binary128 floating points.

        The result is the same as that of :func:`f128_eq()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

        """
        ...

    @classmethod
    def le(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary128 floating points.

        The result is the same as that of :func:`f128_le()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

        """
        ...

    @classmethod
    def lt(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is less than the second one expressed as IEEE 754 binary128 floating points.

        The result is the same as that of :func:`f128_lt()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

        """
        ...

    @classmethod
    def eq_signaling(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is equal to the second one expressed as IEEE 754 binary128 floating points.

        The invalid exception flag is set for any NaN input, not just for signaling NaNs.

        The result is the same as that of :func:`f128_eq_signaling()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

        """
        ...

    @classmethod
    def le_quiet(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary128 floating points.

        The invalid exception flag is not set for quiet NaNs.

        The result is the same as that of :func:`f128_le_quiet()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

        """
        ...

    @classmethod
    def lt_quiet(cls, x: Self, y: Self) -> bool:
        """Tests if the first one is less than the second one expressed as IEEE 754 binary128 floating points.

        The invalid exception flag is not set for quiet NaNs.

        The result is the same as that of :func:`f128_lt_quiet()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

        """
        ...

    def is_signaling_nan(self) -> bool:
        """Tests if the IEEE 754 binary128 floating point is a signaling NaN.

        The result is the same as that of :func:`f128_is_signaling_nan()`.

        Returns:
            ``True`` if the floating point is a signaling NaN, ``False`` otherwise.

        """
        ...

    def is_nan(self) -> bool:
        """Tests if the IEEE 754 binary128 floating point is a NaN.

        The result is the same as that of :func:`f128_is_nan()`.

        Returns:
            ``True`` if the floating point is a NaN, ``False`` otherwise.

        """
        ...

    def is_inf(self) -> bool:
        """Tests if the IEEE 754 binary128 floating point is an infinity.

        The result is the same as that of :func:`f128_is_inf()`.

        Returns:
            ``True`` if the floating point is an infinity, ``False`` otherwise.

        """
        ...

    def __str__(self) -> str:
        ...

    def __pos__(self) -> Self:
        ...

    def __neg__(self) -> Self:
        ...

    def __add__(self, other: Self) -> Self:
        ...

    def __sub__(self, other: Self) -> Self:
        ...

    def __mul__(self, other: Self) -> Self:
        ...

    def __truediv__(self, other: Self) -> Self:
        ...

    def __floordiv__(self, other: Self) -> Self:
        ...

    def __mod__(self, other: Self) -> Self:
        ...

    def __lt__(self, other: Self) -> bool:
        ...

    def __le__(self, other: Self) -> bool:
        ...

    def __gt__(self, other: Self) -> bool:
        ...

    def __ge__(self, other: Self) -> bool:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __iadd__(self, other: Self) -> Self:
        ...

    def __isub__(self, other: Self) -> Self:
        ...

    def __imul__(self, other: Self) -> Self:
        ...

    def __itruediv__(self, other: Self) -> Self:
        ...

    def __ifloordiv__(self, other: Self) -> Self:
        ...

    def __imod__(self, other: Self) -> Self:
        ...


def set_tininess_mode(mode: TininessMode) -> None:
    """Sets the tininess detection mode.

    Args:
        mode: The tininess detection mode to be set.

    """
    ...


def get_tininess_mode() -> TininessMode:
    """Returns the current tininess detection mode.

    Returns:
        The current tininess detection mode.

    """
    ...


def set_rounding_mode(mode: RoundingMode) -> None:
    """Sets the rounding mode.

    Args:
        mode: The rounding mode to be set.

    """
    ...


def get_rounding_mode() -> RoundingMode:
    """Returns the current rounding mode.

    Returns:
        The current rounding mode.

    """
    ...


def set_exception_flags(flags: int) -> None:
    """Sets the floating-point exception flags.

    Args:
        flags: The floating-point exception flags to be set.

    """
    ...


def get_exception_flags() -> int:
    """Returns the current floating-point exception flags.

    Returns:
        The current floating-point exception flags.

    """
    ...


def test_exception_flags(flags: int) -> bool:
    """Tests the floating-point exception flags.

    Args:
        flags: The floating-point exception flags to be tested.

    Returns:
        ``True`` if any of the specified exception flags is nonzero, ``False`` otherwise.

    """
    ...


def ui32_to_f16(x: UInt32) -> Float16:
    """Converts the 32-bit unsigned integer to an IEEE 754 binary16 floating point.

    Args:
        x: The 32-bit unsigned integer to be converted.

    Returns:
        The IEEE 754 binary16 floating point.

    """
    ...


def ui32_to_f32(x: UInt32) -> Float32:
    """Converts the 32-bit unsigned integer to an IEEE 754 binary32 floating point.

    Args:
        x: The 32-bit unsigned integer to be converted.

    Returns:
        The IEEE 754 binary32 floating point.

    """
    ...


def ui32_to_f64(x: UInt32) -> Float64:
    """Converts the 32-bit unsigned integer to an IEEE 754 binary64 floating point.

    Args:
        x: The 32-bit unsigned integer to be converted.

    Returns:
        The IEEE 754 binary64 floating point.

    """
    ...


def ui32_to_f128(x: UInt32) -> Float128:
    """Converts the 32-bit unsigned integer to an IEEE 754 binary128 floating point.

    Args:
        x: The 32-bit unsigned integer to be converted.

    Returns:
        The IEEE 754 binary128 floating point.

    """
    ...


def ui64_to_f16(x: UInt64) -> Float16:
    """Converts the 64-bit unsigned integer to an IEEE 754 binary16 floating point.

    Args:
        x: The 64-bit unsigned integer to be converted.

    Returns:
        The IEEE 754 binary16 floating point.

    """
    ...


def ui64_to_f32(x: UInt64) -> Float32:
    """Converts the 64-bit unsigned integer to an IEEE 754 binary32 floating point.

    Args:
        x: The 64-bit unsigned integer to be converted.

    Returns:
        The IEEE 754 binary32 floating point.

    """
    ...


def ui64_to_f64(x: UInt64) -> Float64:
    """Converts the 64-bit unsigned integer to an IEEE 754 binary64 floating point.

    Args:
        x: The 64-bit unsigned integer to be converted.

    Returns:
        The IEEE 754 binary64 floating point.

    """
    ...


def ui64_to_f128(x: UInt64) -> Float128:
    """Converts the 64-bit unsigned integer to an IEEE 754 binary128 floating point.

    Args:
        x: The 64-bit unsigned integer to be converted.

    Returns:
        The IEEE 754 binary128 floating point.

    """
    ...


def i32_to_f16(x: Int32) -> Float16:
    """Converts the 32-bit signed integer to an IEEE 754 binary16 floating point.

    Args:
        x: The 32-bit signed integer to be converted.

    Returns:
        The IEEE 754 binary16 floating point.

    """
    ...


def i32_to_f32(x: Int32) -> Float32:
    """Converts the 32-bit signed integer to an IEEE 754 binary32 floating point.

    Args:
        x: The 32-bit signed integer to be converted.

    Returns:
        The IEEE 754 binary32 floating point.

    """
    ...


def i32_to_f64(x: Int32) -> Float64:
    """Converts the 32-bit signed integer to an IEEE 754 binary64 floating point.

    Args:
        x: The 32-bit signed integer to be converted.

    Returns:
        The IEEE 754 binary64 floating point.

    """
    ...


def i32_to_f128(x: Int32) -> Float128:
    """Converts the 32-bit signed integer to an IEEE 754 binary128 floating point.

    Args:
        x: The 32-bit signed integer to be converted.

    Returns:
        The IEEE 754 binary128 floating point.

    """
    ...


def i64_to_f16(x: Int64) -> Float16:
    """Converts the 64-bit signed integer to an IEEE 754 binary16 floating point.

    Args:
        x: The 64-bit signed integer to be converted.

    Returns:
        The IEEE 754 binary16 floating point.

    """
    ...


def i64_to_f32(x: Int64) -> Float32:
    """Converts the 64-bit signed integer to an IEEE 754 binary32 floating point.

    Args:
        x: The 64-bit signed integer to be converted.

    Returns:
        The IEEE 754 binary32 floating point.

    """
    ...


def i64_to_f64(x: Int64) -> Float64:
    """Converts the 64-bit signed integer to an IEEE 754 binary64 floating point.

    Args:
        x: The 64-bit signed integer to be converted.

    Returns:
        The IEEE 754 binary64 floating point.

    """
    ...


def i64_to_f128(x: Int64) -> Float128:
    """Converts the 64-bit signed integer to an IEEE 754 binary128 floating point.

    Args:
        x: The 64-bit signed integer to be converted.

    Returns:
        The IEEE 754 binary128 floating point.

    """
    ...


def f16_to_ui32(
    x: Float16, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
) -> UInt32:
    """Converts the IEEE 754 binary16 floating point to a 32-bit unsigned integer.

    Args:
        x: The IEEE 754 binary16 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 32-bit unsigned integer.

    """
    ...


def f16_to_ui64(
    x: Float16, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
) -> UInt64:
    """Converts the IEEE 754 binary16 floating point to a 64-bit unsigned integer.

    Args:
        x: The IEEE 754 binary16 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 64-bit unsigned integer.

    """
    ...


def f16_to_i32(
    x: Float16, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
) -> Int32:
    """Converts the IEEE 754 binary16 floating point to a 32-bit signed integer.

    Args:
        x: The IEEE 754 binary16 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 32-bit signed integer.

    """
    ...


def f16_to_i64(
    x: Float16, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
) -> Int64:
    """Converts the IEEE 754 binary16 floating point to a 64-bit signed integer.

    Args:
        x: The IEEE 754 binary16 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 64-bit signed integer.

    """
    ...


def f16_to_f32(x: Float16) -> Float32:
    """Converts the IEEE 754 binary16 floating point to a binary32 floating point.

    Args:
        x: The IEEE 754 binary16 floating point to be converted.

    Returns:
        The IEEE 754 binary32 floating point.

    """
    ...


def f16_to_f64(x: Float16) -> Float64:
    """Converts the IEEE 754 binary16 floating point to a binary64 floating point.

    Args:
        x: The IEEE 754 binary16 floating point to be converted.

    Returns:
        The IEEE 754 binary64 floating point.

    """
    ...


def f16_to_f128(x: Float16) -> Float128:
    """Converts the IEEE 754 binary16 floating point to a binary128 floating point.

    Args:
        x: The IEEE 754 binary16 floating point to be converted.

    Returns:
        The IEEE 754 binary128 floating point.

    """
    ...


def f16_round_to_int(
    x: Float16, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
) -> Float16:
    """Rounds the number expressed as an IEEE 754 binary16 floating point.

    Args:
        x: The IEEE 754 binary16 floating point to be rounded.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact rounding is unable.

    Returns:
        The resulted integer expressed as an IEEE 754 binary16 floating point.

    """
    ...


def f16_neg(x: Float16) -> Float16:
    """Negates the IEEE 754 binary16 floating point.

    Args:
        x: The floating point to be negated.

    Returns:
        The resulted number expressed as an IEEE 754 binary16 floating point (``-x``).

    """
    ...


def f16_add(x: Float16, y: Float16) -> Float16:
    """Adds the IEEE 754 binary16 floating points.

    Args:
        x: The floating point to be added.
        y: The floating point to add.

    Returns:
        The resulted number expressed as an IEEE 754 binary16 floating point (``x + y``).

    """
    ...


def f16_sub(x: Float16, y: Float16) -> Float16:
    """Subtracts the IEEE 754 binary16 floating points.

    Args:
        x: The floating point to be subtracted.
        y: The floating point to subtract.

    Returns:
        The resulted number expressed as an IEEE 754 binary16 floating point (``x - y``).

    """
    ...


def f16_mul(x: Float16, y: Float16) -> Float16:
    """Multiplies the IEEE 754 binary16 floating points.

    Args:
        x: The floating point to be multiplied.
        y: The floating point to multiply.

    Returns:
        The resulted number expressed as an IEEE 754 binary16 floating point (``x * y``).

    """
    ...


def f16_mul_add(x: Float16, y: Float16, z: Float16) -> Float16:
    """Multiplies and Adds the IEEE 754 binary16 floating points.

    Args:
        x: The floating point to be multiplied.
        y: The floating point to multiply.
        z: The floating point to add.

    Returns:
        The resulted number expressed as an IEEE 754 binary16 floating point (``x * y + z``).

    """
    ...


def f16_div(x: Float16, y: Float16) -> Float16:
    """Divides the IEEE 754 binary16 floating points.

    Args:
        x: The floating point to be divided.
        y: The floating point to divide.

    Returns:
        The resulted number expressed as an IEEE 754 binary16 floating point (``x / y``).

    """
    ...


def f16_rem(x: Float16, y: Float16) -> Float16:
    """Calculates a remainder by dividing the IEEE 754 binary16 floating points.

    Args:
        x: The floating point to be divided.
        y: The floating point to divide.

    Returns:
        The resulted number expressed as an IEEE 754 binary16 floating point (``x % y``).

    """
    ...


def f16_sqrt(x: Float16) -> Float16:
    """Calculates a square root of the IEEE 754 binary16 floating point.

    Args:
        x: The floating point whose square root is to be calculated.

    Returns:
        The resulted number expressed as an IEEE 754 binary16 floating point (``sqrt(x)``).

    """
    ...


def f16_eq(x: Float16, y: Float16) -> bool:
    """Tests if the first one is equal to the second one expressed as IEEE 754 binary16 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

    """
    ...


def f16_le(x: Float16, y: Float16) -> bool:
    """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary16 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

    """
    ...


def f16_lt(x: Float16, y: Float16) -> bool:
    """Tests if the first one is less than the second one expressed as IEEE 754 binary16 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

    """
    ...


def f16_eq_signaling(x: Float16, y: Float16) -> bool:
    """Tests if the first one is equal to the second one expressed as IEEE 754 binary16 floating points.

    The invalid exception flag is set for any NaN input, not just for signaling NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

    """
    ...


def f16_le_quiet(x: Float16, y: Float16) -> bool:
    """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary16 floating points.

    The invalid exception flag is not set for quiet NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

    """
    ...


def f16_lt_quiet(x: Float16, y: Float16) -> bool:
    """Tests if the first one is less than the second one expressed as IEEE 754 binary16 floating points.

    The invalid exception flag is not set for quiet NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

    """
    ...


def f16_is_signaling_nan(x: Float16) -> bool:
    """Tests if the IEEE 754 binary16 floating point is a signaling NaN.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is a signaling NaN, ``False`` otherwise.

    """
    ...


def f16_is_nan(x: Float16) -> bool:
    """Tests if the IEEE 754 binary16 floating point is a NaN.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is a NaN, ``False`` otherwise.

    """
    ...


def f16_is_inf(x: Float16) -> bool:
    """Tests if the IEEE 754 binary16 floating point is an infinity.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is an infinity, ``False`` otherwise.

    """
    ...


def bf16_to_f32(x: BFloat16) -> Float32:
    """Converts the 16-bit brain floating point to an IEEE 754 binary32 floating point.

    Args:
        x: The 16-bit brain floating point to be converted.

    Returns:
        The IEEE 754 binary32 floating point.

    """
    ...


def f32_to_bf16(x: Float32) -> BFloat16:
    """Converts the IEEE 754 binary32 floating point to a 16-bit brain floating point.

    Args:
        x: The IEEE 754 binary32 floating point to be converted.

    Returns:
        The 16-bit brain floating point.

    """
    ...


def bf16_is_signaling_nan(x: BFloat16) -> bool:
    """Tests if the 16-bit brain floating point is a signaling NaN.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is a signaling NaN, ``False`` otherwise.

    """
    ...


def bf16_is_nan(x: BFloat16) -> bool:
    """Tests if the 16-bit brain floating point is a NaN.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is a NaN, ``False`` otherwise.

    """
    ...


def bf16_is_inf(x: BFloat16) -> bool:
    """Tests if the 16-bit brain floating point is an infinity.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is an infinity, ``False`` otherwise.

    """
    ...


def f32_to_ui32(
    x: Float32, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
) -> UInt32:
    """Converts the IEEE 754 binary32 floating point to a 32-bit unsigned integer.

    Args:
        x: The IEEE 754 binary32 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 32-bit unsigned integer.

    """
    ...


def f32_to_ui64(
    x: Float32, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
) -> UInt64:
    """Converts the IEEE 754 binary32 floating point to a 64-bit unsigned integer.

    Args:
        x: The IEEE 754 binary32 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 64-bit unsigned integer.

    """
    ...


def f32_to_i32(
    x: Float32, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
) -> Int32:
    """Converts the IEEE 754 binary32 floating point to a 32-bit signed integer.

    Args:
        x: The IEEE 754 binary32 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 32-bit signed integer.

    """
    ...


def f32_to_i64(
    x: Float32, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
) -> Int64:
    """Converts the IEEE 754 binary32 floating point to a 64-bit signed integer.

    Args:
        x: The IEEE 754 binary32 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 64-bit signed integer.

    """
    ...


def f32_to_f16(x: Float32) -> Float16:
    """Converts the IEEE 754 binary32 floating point to a binary16 floating point.

    Args:
        x: The IEEE 754 binary32 floating point to be converted.

    Returns:
        The IEEE 754 binary16 floating point.

    """
    ...


def f32_to_f64(x: Float32) -> Float64:
    """Converts the IEEE 754 binary32 floating point to a binary64 floating point.

    Args:
        x: The IEEE 754 binary32 floating point to be converted.

    Returns:
        The IEEE 754 binary64 floating point.

    """
    ...


def f32_to_f128(x: Float32) -> Float128:
    """Converts the IEEE 754 binary32 floating point to a binary128 floating point.

    Args:
        x: The IEEE 754 binary32 floating point to be converted.

    Returns:
        The IEEE 754 binary128 floating point.

    """
    ...


def f32_round_to_int(
    x: Float32, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
) -> Float32:
    """Rounds the number expressed as an IEEE 754 binary32 floating point.

    Args:
        x: The IEEE 754 binary32 floating point to be rounded.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact rounding is unable.

    Returns:
        The resulted integer expressed as an IEEE 754 binary32 floating point.

    """
    ...


def f32_neg(x: Float32) -> Float32:
    """Negates the IEEE 754 binary32 floating point.

    Args:
        x: The floating point to be negated.

    Returns:
        The resulted number expressed as an IEEE 754 binary32 floating point (``-x``).

    """
    ...


def f32_add(x: Float32, y: Float32) -> Float32:
    """Adds the IEEE 754 binary32 floating points.

    Args:
        x: The floating point to be added.
        y: The floating point to add.

    Returns:
        The resulted number expressed as an IEEE 754 binary32 floating point (``x + y``).

    """
    ...


def f32_sub(x: Float32, y: Float32) -> Float32:
    """Subtracts the IEEE 754 binary32 floating points.

    Args:
        x: The floating point to be subtracted.
        y: The floating point to subtract.

    Returns:
        The resulted number expressed as an IEEE 754 binary32 floating point (``x - y``).

    """
    ...


def f32_mul(x: Float32, y: Float32) -> Float32:
    """Multiplies the IEEE 754 binary32 floating points.

    Args:
        x: The floating point to be multiplied.
        y: The floating point to multiply.

    Returns:
        The resulted number expressed as an IEEE 754 binary32 floating point (``x * y``).

    """
    ...


def f32_mul_add(x: Float32, y: Float32, z: Float32) -> Float32:
    """Multiplies and Adds the IEEE 754 binary32 floating points.

    Args:
        x: The floating point to be multiplied.
        y: The floating point to multiply.
        z: The floating point to add.

    Returns:
        The resulted number expressed as an IEEE 754 binary32 floating point (``x * y + z``).

    """
    ...


def f32_div(x: Float32, y: Float32) -> Float32:
    """Divides the IEEE 754 binary32 floating points.

    Args:
        x: The floating point to be divided.
        y: The floating point to divide.

    Returns:
        The resulted number expressed as an IEEE 754 binary32 floating point (``x / y``).

    """
    ...


def f32_rem(x: Float32, y: Float32) -> Float32:
    """Calculates a remainder by dividing the IEEE 754 binary32 floating points.

    Args:
        x: The floating point to be divided.
        y: The floating point to divide.

    Returns:
        The resulted number expressed as an IEEE 754 binary32 floating point (``x % y``).

    """
    ...


def f32_sqrt(x: Float32) -> Float32:
    """Calculates a square root of the IEEE 754 binary32 floating point.

    Args:
        x: The floating point whose square root is to be calculated.

    Returns:
        The resulted number expressed as an IEEE 754 binary32 floating point (``sqrt(x)``).

    """
    ...


def f32_eq(x: Float32, y: Float32) -> bool:
    """Tests if the first one is equal to the second one expressed as IEEE 754 binary32 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

    """
    ...


def f32_le(x: Float32, y: Float32) -> bool:
    """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary32 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

    """
    ...


def f32_lt(x: Float32, y: Float32) -> bool:
    """Tests if the first one is less than the second one expressed as IEEE 754 binary32 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

    """
    ...


def f32_eq_signaling(x: Float32, y: Float32) -> bool:
    """Tests if the first one is equal to the second one expressed as IEEE 754 binary32 floating points.

    The invalid exception flag is set for any NaN input, not just for signaling NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

    """
    ...


def f32_le_quiet(x: Float32, y: Float32) -> bool:
    """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary32 floating points.

    The invalid exception flag is not set for quiet NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

    """
    ...


def f32_lt_quiet(x: Float32, y: Float32) -> bool:
    """Tests if the first one is less than the second one expressed as IEEE 754 binary32 floating points.

    The invalid exception flag is not set for quiet NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

    """
    ...


def f32_is_signaling_nan(x: Float32) -> bool:
    """Tests if the IEEE 754 binary32 floating point is a signaling NaN.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is a signaling NaN, ``False`` otherwise.

    """
    ...


def f32_is_nan(x: Float32) -> bool:
    """Tests if the IEEE 754 binary32 floating point is a NaN.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is a NaN, ``False`` otherwise.

    """
    ...


def f32_is_inf(x: Float32) -> bool:
    """Tests if the IEEE 754 binary32 floating point is an infinity.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is an infinity, ``False`` otherwise.

    """
    ...


def f64_to_ui32(
    x: Float64, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
) -> UInt32:
    """Converts the IEEE 754 binary64 floating point to a 32-bit unsigned integer.

    Args:
        x: The IEEE 754 binary64 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 32-bit unsigned integer.

    """
    ...


def f64_to_ui64(
    x: Float64, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
) -> UInt64:
    """Converts the IEEE 754 binary64 floating point to a 64-bit unsigned integer.

    Args:
        x: The IEEE 754 binary64 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 64-bit unsigned integer.

    """
    ...


def f64_to_i32(
    x: Float64, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
) -> Int32:
    """Converts the IEEE 754 binary64 floating point to a 32-bit signed integer.

    Args:
        x: The IEEE 754 binary64 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 32-bit signed integer.

    """
    ...


def f64_to_i64(
    x: Float64, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
) -> Int64:
    """Converts the IEEE 754 binary64 floating point to a 64-bit signed integer.

    Args:
        x: The IEEE 754 binary64 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 64-bit signed integer.

    """
    ...


def f64_to_f16(x: Float64) -> Float16:
    """Converts the IEEE 754 binary64 floating point to a binary16 floating point.

    Args:
        x: The IEEE 754 binary64 floating point to be converted.

    Returns:
        The IEEE 754 binary16 floating point.

    """
    ...


def f64_to_f32(x: Float64) -> Float32:
    """Converts the IEEE 754 binary64 floating point to a binary32 floating point.

    Args:
        x: The IEEE 754 binary64 floating point to be converted.

    Returns:
        The IEEE 754 binary32 floating point.

    """
    ...


def f64_to_f128(x: Float64) -> Float128:
    """Converts the IEEE 754 binary64 floating point to a binary128 floating point.

    Args:
        x: The IEEE 754 binary64 floating point to be converted.

    Returns:
        The IEEE 754 binary128 floating point.

    """
    ...


def f64_round_to_int(
    x: Float64, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
) -> Float64:
    """Rounds the number expressed as an IEEE 754 binary64 floating point.

    Args:
        x: The IEEE 754 binary64 floating point to be rounded.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact rounding is unable.

    Returns:
        The resulted integer expressed as an IEEE 754 binary64 floating point.

    """
    ...


def f64_neg(x: Float64) -> Float64:
    """Negates the IEEE 754 binary64 floating point.

    Args:
        x: The floating point to be negated.

    Returns:
        The resulted number expressed as an IEEE 754 binary64 floating point (``-x``).

    """
    ...


def f64_add(x: Float64, y: Float64) -> Float64:
    """Adds the IEEE 754 binary64 floating points.

    Args:
        x: The floating point to be added.
        y: The floating point to add.

    Returns:
        The resulted number expressed as an IEEE 754 binary64 floating point (``x + y``).

    """
    ...


def f64_sub(x: Float64, y: Float64) -> Float64:
    """Subtracts the IEEE 754 binary64 floating points.

    Args:
        x: The floating point to be subtracted.
        y: The floating point to subtract.

    Returns:
        The resulted number expressed as an IEEE 754 binary64 floating point (``x - y``).

    """
    ...


def f64_mul(x: Float64, y: Float64) -> Float64:
    """Multiplies the IEEE 754 binary64 floating points.

    Args:
        x: The floating point to be multiplied.
        y: The floating point to multiply.

    Returns:
        The resulted number expressed as an IEEE 754 binary64 floating point (``x * y``).

    """
    ...


def f64_mul_add(x: Float64, y: Float64, z: Float64) -> Float64:
    """Multiplies and Adds the IEEE 754 binary64 floating points.

    Args:
        x: The floating point to be multiplied.
        y: The floating point to multiply.
        z: The floating point to add.

    Returns:
        The resulted number expressed as an IEEE 754 binary64 floating point (``x * y + z``).

    """
    ...


def f64_div(x: Float64, y: Float64) -> Float64:
    """Divides the IEEE 754 binary64 floating points.

    Args:
        x: The floating point to be divided.
        y: The floating point to divide.

    Returns:
        The resulted number expressed as an IEEE 754 binary64 floating point (``x / y``).

    """
    ...


def f64_rem(x: Float64, y: Float64) -> Float64:
    """Calculates a remainder by dividing the IEEE 754 binary64 floating points.

    Args:
        x: The floating point to be divided.
        y: The floating point to divide.

    Returns:
        The resulted number expressed as an IEEE 754 binary64 floating point (``x % y``).

    """
    ...


def f64_sqrt(x: Float64) -> Float64:
    """Calculates a square root of the IEEE 754 binary64 floating point.

    Args:
        x: The floating point whose square root is to be calculated.

    Returns:
        The resulted number expressed as an IEEE 754 binary64 floating point (``sqrt(x)``).

    """
    ...


def f64_eq(x: Float64, y: Float64) -> bool:
    """Tests if the first one is equal to the second one expressed as IEEE 754 binary64 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

    """
    ...


def f64_le(x: Float64, y: Float64) -> bool:
    """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary64 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

    """
    ...


def f64_lt(x: Float64, y: Float64) -> bool:
    """Tests if the first one is less than the second one expressed as IEEE 754 binary64 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

    """
    ...


def f64_eq_signaling(x: Float64, y: Float64) -> bool:
    """Tests if the first one is equal to the second one expressed as IEEE 754 binary64 floating points.

    The invalid exception flag is set for any NaN input, not just for signaling NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

    """
    ...


def f64_le_quiet(x: Float64, y: Float64) -> bool:
    """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary64 floating points.

    The invalid exception flag is not set for quiet NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

    """
    ...


def f64_lt_quiet(x: Float64, y: Float64) -> bool:
    """Tests if the first one is less than the second one expressed as IEEE 754 binary64 floating points.

    The invalid exception flag is not set for quiet NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

    """
    ...


def f64_is_signaling_nan(x: Float64) -> bool:
    """Tests if the IEEE 754 binary64 floating point is a signaling NaN.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is a signaling NaN, ``False`` otherwise.

    """
    ...


def f64_is_nan(x: Float64) -> bool:
    """Tests if the IEEE 754 binary64 floating point is a NaN.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is a NaN, ``False`` otherwise.

    """
    ...


def f64_is_inf(x: Float64) -> bool:
    """Tests if the IEEE 754 binary64 floating point is an infinity.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is an infinity, ``False`` otherwise.

    """
    ...


def f128_to_ui32(
    x: Float128, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
) -> UInt32:
    """Converts the IEEE 754 binary128 floating point to a 32-bit unsigned integer.

    Args:
        x: The IEEE 754 binary128 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 32-bit unsigned integer.

    """
    ...


def f128_to_ui64(
    x: Float128, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
) -> UInt64:
    """Converts the IEEE 754 binary128 floating point to a 64-bit unsigned integer.

    Args:
        x: The IEEE 754 binary128 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 64-bit unsigned integer.

    """
    ...


def f128_to_i32(
    x: Float128, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
) -> Int32:
    """Converts the IEEE 754 binary128 floating point to a 32-bit signed integer.

    Args:
        x: The IEEE 754 binary128 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 32-bit signed integer.

    """
    ...


def f128_to_i64(
    x: Float128, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
) -> Int64:
    """Converts the IEEE 754 binary128 floating point to a 64-bit signed integer.

    Args:
        x: The IEEE 754 binary128 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 64-bit signed integer.

    """
    ...


def f128_to_f16(x: Float128) -> Float16:
    """Converts the IEEE 754 binary128 floating point to a binary16 floating point.

    Args:
        x: The IEEE 754 binary128 floating point to be converted.

    Returns:
        The IEEE 754 binary16 floating point.

    """
    ...


def f128_to_f32(x: Float128) -> Float32:
    """Converts the IEEE 754 binary128 floating point to a binary32 floating point.

    Args:
        x: The IEEE 754 binary128 floating point to be converted.

    Returns:
        The IEEE 754 binary32 floating point.

    """
    ...


def f128_to_f64(x: Float128) -> Float64:
    """Converts the IEEE 754 binary128 floating point to a binary64 floating point.

    Args:
        x: The IEEE 754 binary128 floating point to be converted.

    Returns:
        The IEEE 754 binary64 floating point.

    """
    ...


def f128_round_to_int(
    x: Float128, rounding_mode: RoundingMode = get_rounding_mode(), exact: bool = True
) -> Float128:
    """Rounds the number expressed as an IEEE 754 binary128 floating point.

    Args:
        x: The IEEE 754 binary128 floating point to be rounded.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact rounding is unable.

    Returns:
        The resulted integer expressed as an IEEE 754 binary128 floating point.

    """
    ...


def f128_neg(x: Float128) -> Float128:
    """Negates the IEEE 754 binary128 floating point.

    Args:
        x: The floating point to be negated.

    Returns:
        The resulted number expressed as an IEEE 754 binary128 floating point (``-x``).

    """
    ...


def f128_add(x: Float128, y: Float128) -> Float128:
    """Adds the IEEE 754 binary128 floating points.

    Args:
        x: The floating point to be added.
        y: The floating point to add.

    Returns:
        The resulted number expressed as an IEEE 754 binary128 floating point (``x + y``).

    """
    ...


def f128_sub(x: Float128, y: Float128) -> Float128:
    """Subtracts the IEEE 754 binary128 floating points.

    Args:
        x: The floating point to be subtracted.
        y: The floating point to subtract.

    Returns:
        The resulted number expressed as an IEEE 754 binary128 floating point (``x - y``).

    """
    ...


def f128_mul(x: Float128, y: Float128) -> Float128:
    """Multiplies the IEEE 754 binary128 floating points.

    Args:
        x: The floating point to be multiplied.
        y: The floating point to multiply.

    Returns:
        The resulted number expressed as an IEEE 754 binary128 floating point (``x * y``).

    """
    ...


def f128_mul_add(x: Float128, y: Float128, z: Float128) -> Float128:
    """Multiplies and Adds the IEEE 754 binary128 floating points.

    Args:
        x: The floating point to be multiplied.
        y: The floating point to multiply.
        z: The floating point to add.

    Returns:
        The resulted number expressed as an IEEE 754 binary128 floating point (``x * y + z``).

    """
    ...


def f128_div(x: Float128, y: Float128) -> Float128:
    """Divides the IEEE 754 binary128 floating points.

    Args:
        x: The floating point to be divided.
        y: The floating point to divide.

    Returns:
        The resulted number expressed as an IEEE 754 binary128 floating point (``x / y``).

    """
    ...


def f128_rem(x: Float128, y: Float128) -> Float128:
    """Calculates a remainder by dividing the IEEE 754 binary128 floating points.

    Args:
        x: The floating point to be divided.
        y: The floating point to divide.

    Returns:
        The resulted number expressed as an IEEE 754 binary128 floating point (``x % y``).

    """
    ...


def f128_sqrt(x: Float128) -> Float128:
    """Calculates a square root of the IEEE 754 binary128 floating point.

    Args:
        x: The floating point whose square root is to be calculated.

    Returns:
        The resulted number expressed as an IEEE 754 binary128 floating point (``sqrt(x)``).

    """
    ...


def f128_eq(x: Float128, y: Float128) -> bool:
    """Tests if the first one is equal to the second one expressed as IEEE 754 binary128 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

    """
    ...


def f128_le(x: Float128, y: Float128) -> bool:
    """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary128 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

    """
    ...


def f128_lt(x: Float128, y: Float128) -> bool:
    """Tests if the first one is less than the second one expressed as IEEE 754 binary128 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

    """
    ...


def f128_eq_signaling(x: Float128, y: Float128) -> bool:
    """Tests if the first one is equal to the second one expressed as IEEE 754 binary128 floating points.

    The invalid exception flag is set for any NaN input, not just for signaling NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

    """
    ...


def f128_le_quiet(x: Float128, y: Float128) -> bool:
    """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary128 floating points.

    The invalid exception flag is not set for quiet NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

    """
    ...


def f128_lt_quiet(x: Float128, y: Float128) -> bool:
    """Tests if the first one is less than the second one expressed as IEEE 754 binary128 floating points.

    The invalid exception flag is not set for quiet NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

    """
    ...


def f128_is_signaling_nan(x: Float128) -> bool:
    """Tests if the IEEE 754 binary128 floating point is a signaling NaN.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is a signaling NaN, ``False`` otherwise.

    """
    ...


def f128_is_nan(x: Float128) -> bool:
    """Tests if the IEEE 754 binary128 floating point is a NaN.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is a NaN, ``False`` otherwise.

    """
    ...


def f128_is_inf(x: Float128) -> bool:
    """Tests if the IEEE 754 binary128 floating point is an infinity.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is an infinity, ``False`` otherwise.

    """
    ...
