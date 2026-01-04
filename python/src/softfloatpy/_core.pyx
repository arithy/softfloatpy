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

# cython: language_level=3
# cython: embedsignature=True

from typing import Self

from cython.view cimport array
from libc.stdint cimport (
    uint8_t, uint16_t, uint32_t, uint64_t,
    int32_t, int64_t,
    uint_fast8_t
)

cimport softfloat as sf


cdef extern from "internals.h":

    struct uint128:
        uint64_t v0
        uint64_t v64

    union ui32_f32:
        uint32_t ui
        sf.float32_t f

    union ui64_f64:
        uint64_t ui
        sf.float64_t f

    union ui128_f128:
        uint128 ui
        sf.float128_t f


cdef union _ui64_double:
    uint64_t ui
    double f


cpdef enum TininessMode:
    """The tininess detection modes.

    - ``BEFORE_ROUNDING``: Detecting tininess before rounding.
    - ``AFTER_ROUNDING``: Detecting tininess after rounding.

    """
    BEFORE_ROUNDING = 0
    AFTER_ROUNDING = 1


cpdef enum RoundingMode:
    """The rounding modes.

    - ``NEAR_EVEN``: Rounding to nearest, with ties to even.
    - ``NEAR_MAX_MAG``: Rounding to nearest, with ties to maximum magnitude (away from zero).
    - ``MIN_MAG``: Rounding to minimum magnitude (toward zero).
    - ``MIN``: Rounding to minimum (down).
    - ``MAX``: Rounding to maximum (up).

    """
    NEAR_EVEN = 0
    MIN_MAG = 1
    MIN = 2
    MAX = 3
    NEAR_MAX_MAG = 4


cpdef enum ExceptionFlag:
    """The floating-point exception flags.

    - ``INEXACT``: The exception set if the rounded value is different from the mathematically exact result of the operation.
    - ``UNDERFLOW``: The exception set if the rounded value is tiny and inexact.
    - ``OVERFLOW``: The exception set if the absolute value of the rounded value is too large to be represented.
    - ``INFINITE``: The exception set if the result is infinite given finite operands.
    - ``INVALID``: The exception set if a finite or infinite result cannot be returned.

    """
    INEXACT = 1
    UNDERFLOW = 2
    OVERFLOW = 4
    INFINITE = 8
    INVALID = 16


cdef class UInt32:
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

    cdef uint32_t _data
    """The native data."""

    cpdef uint32_t _get_data(self):  # To access the native data in another instance.
        return self._data

    @classmethod
    def size(cls) -> int:
        """Returns the native data size in bits.

        Returns:
            The native data size in bits, i.e. 32.

        """
        return 32

    @classmethod
    def from_bytes(cls, bytes src) -> UInt32:
        """Creates a new instance from the specified byte sequence.

        Args:
            src: The byte sequence representing the native data with big endian.
                 The length must be 4.

        Returns:
            A new instance created from the specified byte sequence.

        Raises:
            ValueError: If the length of bytes is not 4.

        """
        if len(src) != 4:
            raise ValueError('length of bytes must be 4')
        cdef const uint8_t[:] a = src
        cdef UInt32 o = UInt32()
        o._data = (
            (<uint32_t>a[0] << 24) |
            (<uint32_t>a[1] << 16) |
            (<uint32_t>a[2] << 8) |
            <uint32_t>a[3]
        )
        return o

    cpdef bytes to_bytes(self):
        """Returns the native data as a byte sequence.

        Returns:
            A byte sequence representing the native data with big endian.
            The length is 4.

        """
        cdef uint8_t a[4]
        a[0] = <uint8_t>(self._data >> 24)
        a[1] = <uint8_t>(self._data >> 16)
        a[2] = <uint8_t>(self._data >> 8)
        a[3] = <uint8_t>self._data
        return <bytes>a[:4]

    @classmethod
    def from_int(cls, src) -> UInt32:
        """Creates a new instance from the specified integer.

        Args:
            src: The integer from which a new instance is created.

        Returns:
            A new instance created from the specified integer.

        """
        cdef UInt32 o = UInt32()
        o._data = <uint32_t>src
        return o

    cpdef to_int(self):
        """Returns the native data as an integer.

        Returns:
            An integer that represents the native data.

        """
        return int(self._data)

    @classmethod
    def from_f16(cls, Float16 src, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True) -> UInt32:
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
        return f16_to_ui32(src, rounding_mode, exact)

    cpdef Float16 to_f16(self):
        """Converts the 32-bit unsigned integer to an IEEE 754 binary16 floating point.

        The result is the same as that of :func:`ui32_to_f16()`.

        Returns:
            The IEEE 754 binary16 floating point.

        """
        return ui32_to_f16(self)

    @classmethod
    def from_f32(cls, Float32 src, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True) -> UInt32:
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
        return f32_to_ui32(src, rounding_mode, exact)

    cpdef Float32 to_f32(self):
        """Converts the 32-bit unsigned integer to an IEEE 754 binary32 floating point.

        The result is the same as that of :func:`ui32_to_f32()`.

        Returns:
            The IEEE 754 binary32 floating point.

        """
        return ui32_to_f32(self)

    @classmethod
    def from_f64(cls, Float64 src, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True) -> UInt32:
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
        return f64_to_ui32(src, rounding_mode, exact)

    cpdef Float64 to_f64(self):
        """Converts the 32-bit unsigned integer to an IEEE 754 binary64 floating point.

        The result is the same as that of :func:`ui32_to_f64()`.

        Returns:
            The IEEE 754 binary64 floating point.

        """
        return ui32_to_f64(self)

    @classmethod
    def from_f128(cls, Float128 src, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True) -> UInt32:
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
        return f128_to_ui32(src, rounding_mode, exact)

    cpdef Float128 to_f128(self):
        """Converts the 32-bit unsigned integer to an IEEE 754 binary128 floating point.

        The result is the same as that of :func:`ui32_to_f128()`.

        Returns:
            The IEEE 754 binary128 floating point.

        """
        return ui32_to_f128(self)

    def __str__(self) -> str:
        return str(int(self._data))

    def __pos__(self) -> Self:
        return _make_uint32(self._data)

    def __neg__(self) -> Self:
        if self._data != 0:
            raise ValueError("cannot be negated")
        return _make_uint32(self._data)

    def __invert__(self) -> Self:
        return _make_uint32(~self._data)

    def __add__(self, other: Self) -> Self:
        return _make_uint32(self._data + other._get_data())

    def __sub__(self, other: Self) -> Self:
        return _make_uint32(self._data - other._get_data())

    def __mul__(self, other: Self) -> Self:
        return _make_uint32(self._data * other._get_data())

    def __floordiv__(self, other: Self) -> Self:
        return _make_uint32(self._data // other._get_data())

    def __mod__(self, other: Self) -> Self:
        return _make_uint32(self._data % other._get_data())

    def __lshift__(self, other: Self) -> Self:
        if other._get_data() < 0:
            raise ValueError("negative shift count")
        return _make_uint32(self._data << other._get_data())

    def __rshift__(self, other: Self) -> Self:
        if other._get_data() < 0:
            raise ValueError("negative shift count")
        return _make_uint32(self._data >> other._get_data())

    def __and__(self, other: Self) -> Self:
        return _make_uint32(self._data & other._get_data())

    def __or__(self, other: Self) -> Self:
        return _make_uint32(self._data | other._get_data())

    def __xor__(self, other: Self) -> Self:
        return _make_uint32(self._data ^ other._get_data())

    def __lt__(self, other: Self) -> bool:
        return self._data < other._get_data()

    def __le__(self, other: Self) -> bool:
        return self._data <= other._get_data()

    def __gt__(self, other: Self) -> bool:
        return self._data > other._get_data()

    def __ge__(self, other: Self) -> bool:
        return self._data >= other._get_data()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            raise ValueError('type mismatch')
        return self._data == other._get_data()

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            raise ValueError('type mismatch')
        return self._data != other._get_data()

    def __iadd__(self, other: Self) -> Self:
        return self.__add__(other)

    def __isub__(self, other: Self) -> Self:
        return self.__sub__(other)

    def __imul__(self, other: Self) -> Self:
        return self.__mul__(other)

    def __ifloordiv__(self, other: Self) -> Self:
        return self.__floordiv__(other)

    def __imod__(self, other: Self) -> Self:
        return self.__mod__(other)

    def __ilshift__(self, other: Self) -> Self:
        return self.__lshift__(other)

    def __irshift__(self, other: Self) -> Self:
        return self.__rshift__(other)

    def __iand__(self, other: Self) -> Self:
        return self.__and__(other)

    def __ior__(self, other: Self) -> Self:
        return self.__or__(other)

    def __ixor__(self, other: Self) -> Self:
        return self.__xor__(other)


cdef class UInt64:
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

    cdef uint64_t _data
    """The native data."""

    cpdef uint64_t _get_data(self):  # To access the native data in another instance.
        return self._data

    @classmethod
    def size(cls) -> int:
        """Returns the native data size in bits.

        Returns:
            The native data size in bits, i.e. 64.

        """
        return 64

    @classmethod
    def from_bytes(cls, bytes src) -> UInt64:
        """Creates a new instance from the specified byte sequence.

        Args:
            src: The byte sequence representing the native data with big endian.
                 The length must be 8.

        Returns:
            A new instance created from the specified byte sequence.

        Raises:
            ValueError: If the length of bytes is not 8.

        """
        if len(src) != 8:
            raise ValueError('length of bytes must be 8')
        cdef const uint8_t[:] a = src
        cdef UInt64 o = UInt64()
        o._data = (
            (<uint64_t>a[0] << 56) |
            (<uint64_t>a[1] << 48) |
            (<uint64_t>a[2] << 40) |
            (<uint64_t>a[3] << 32) |
            (<uint64_t>a[4] << 24) |
            (<uint64_t>a[5] << 16) |
            (<uint64_t>a[6] << 8) |
            <uint64_t>a[7]
        )
        return o

    cpdef bytes to_bytes(self):
        """Returns the native data as a byte sequence.

        Returns:
            A byte sequence representing the native data with big endian.
            The length is 8.

        """
        cdef uint8_t a[8]
        a[0] = <uint8_t>(self._data >> 56)
        a[1] = <uint8_t>(self._data >> 48)
        a[2] = <uint8_t>(self._data >> 40)
        a[3] = <uint8_t>(self._data >> 32)
        a[4] = <uint8_t>(self._data >> 24)
        a[5] = <uint8_t>(self._data >> 16)
        a[6] = <uint8_t>(self._data >> 8)
        a[7] = <uint8_t>self._data
        return <bytes>a[:8]

    @classmethod
    def from_int(cls, src) -> UInt64:
        """Creates a new instance from the specified integer.

        Args:
            src: The integer from which a new instance is created.

        Returns:
            A new instance created from the specified integer.

        """
        cdef UInt64 o = UInt64()
        o._data = <uint64_t>src
        return o

    cpdef to_int(self):
        """Returns the native data as an integer.

        Returns:
            An integer that represents the native data.

        """
        return int(self._data)

    @classmethod
    def from_f16(cls, Float16 src, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True) -> UInt64:
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
        return f16_to_ui64(src, rounding_mode, exact)

    cpdef Float16 to_f16(self):
        """Converts the 64-bit unsigned integer to an IEEE 754 binary16 floating point.

        The result is the same as that of :func:`ui64_to_f16()`.

        Returns:
            The IEEE 754 binary16 floating point.

        """
        return ui64_to_f16(self)

    @classmethod
    def from_f32(cls, Float32 src, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True) -> UInt64:
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
        return f32_to_ui64(src, rounding_mode, exact)

    cpdef Float32 to_f32(self):
        """Converts the 64-bit unsigned integer to an IEEE 754 binary32 floating point.

        The result is the same as that of :func:`ui64_to_f32()`.

        Returns:
            The IEEE 754 binary32 floating point.

        """
        return ui64_to_f32(self)

    @classmethod
    def from_f64(cls, Float64 src, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True) -> UInt64:
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
        return f64_to_ui64(src, rounding_mode, exact)

    cpdef Float64 to_f64(self):
        """Converts the 64-bit unsigned integer to an IEEE 754 binary64 floating point.

        The result is the same as that of :func:`ui64_to_f64()`.

        Returns:
            The IEEE 754 binary64 floating point.

        """
        return ui64_to_f64(self)

    @classmethod
    def from_f128(cls, Float128 src, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True) -> UInt64:
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
        return f128_to_ui64(src, rounding_mode, exact)

    cpdef Float128 to_f128(self):
        """Converts the 64-bit unsigned integer to an IEEE 754 binary128 floating point.

        The result is the same as that of :func:`ui64_to_f128()`.

        Returns:
            The IEEE 754 binary128 floating point.

        """
        return ui64_to_f128(self)

    def __str__(self) -> str:
        return str(int(self._data))

    def __pos__(self) -> Self:
        return _make_uint64(self._data)

    def __neg__(self) -> Self:
        if self._data != 0:
            raise ValueError("cannot be negated")
        return _make_uint32(self._data)

    def __invert__(self) -> Self:
        return _make_uint64(~self._data)

    def __add__(self, other: Self) -> Self:
        return _make_uint64(self._data + other._get_data())

    def __sub__(self, other: Self) -> Self:
        return _make_uint64(self._data - other._get_data())

    def __mul__(self, other: Self) -> Self:
        return _make_uint64(self._data * other._get_data())

    def __floordiv__(self, other: Self) -> Self:
        return _make_uint64(self._data // other._get_data())

    def __mod__(self, other: Self) -> Self:
        return _make_uint64(self._data % other._get_data())

    def __lshift__(self, other: Self) -> Self:
        if other._get_data() < 0:
            raise ValueError("negative shift count")
        return _make_uint64(self._data << other._get_data())

    def __rshift__(self, other: Self) -> Self:
        if other._get_data() < 0:
            raise ValueError("negative shift count")
        return _make_uint64(self._data >> other._get_data())

    def __and__(self, other: Self) -> Self:
        return _make_uint64(self._data & other._get_data())

    def __or__(self, other: Self) -> Self:
        return _make_uint64(self._data | other._get_data())

    def __xor__(self, other: Self) -> Self:
        return _make_uint64(self._data ^ other._get_data())

    def __lt__(self, other: Self) -> bool:
        return self._data < other._get_data()

    def __le__(self, other: Self) -> bool:
        return self._data <= other._get_data()

    def __gt__(self, other: Self) -> bool:
        return self._data > other._get_data()

    def __ge__(self, other: Self) -> bool:
        return self._data >= other._get_data()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            raise ValueError('type mismatch')
        return self._data == other._get_data()

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            raise ValueError('type mismatch')
        return self._data != other._get_data()

    def __iadd__(self, other: Self) -> Self:
        return self.__add__(other)

    def __isub__(self, other: Self) -> Self:
        return self.__sub__(other)

    def __imul__(self, other: Self) -> Self:
        return self.__mul__(other)

    def __ifloordiv__(self, other: Self) -> Self:
        return self.__floordiv__(other)

    def __imod__(self, other: Self) -> Self:
        return self.__mod__(other)

    def __ilshift__(self, other: Self) -> Self:
        return self.__lshift__(other)

    def __irshift__(self, other: Self) -> Self:
        return self.__rshift__(other)

    def __iand__(self, other: Self) -> Self:
        return self.__and__(other)

    def __ior__(self, other: Self) -> Self:
        return self.__or__(other)

    def __ixor__(self, other: Self) -> Self:
        return self.__xor__(other)


cdef class Int32:
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

    cdef int32_t _data
    """The native data."""

    cpdef int32_t _get_data(self):  # To access the native data in another instance.
        return self._data

    @classmethod
    def size(cls) -> int:
        """Returns the native data size in bits.

        Returns:
            The native data size in bits, i.e. 32.

        """
        return 32

    @classmethod
    def from_bytes(cls, bytes src) -> Int32:
        """Creates a new instance from the specified byte sequence.

        Args:
            src: The byte sequence representing the native data with big endian.
                 The length must be 4.

        Returns:
            A new instance created from the specified byte sequence.

        Raises:
            ValueError: If the length of bytes is not 4.

        """
        if len(src) != 4:
            raise ValueError('length of bytes must be 4')
        cdef const uint8_t[:] a = src
        cdef Int32 o = Int32()
        o._data = (
            (<int32_t>a[0] << 24) |
            (<int32_t>a[1] << 16) |
            (<int32_t>a[2] << 8) |
            <int32_t>a[3]
        )
        return o

    cpdef bytes to_bytes(self):
        """Returns the native data as a byte sequence.

        Returns:
            A byte sequence representing the native data with big endian.
            The length is 4.

        """
        cdef uint8_t a[4]
        a[0] = <uint8_t>(self._data >> 24)
        a[1] = <uint8_t>(self._data >> 16)
        a[2] = <uint8_t>(self._data >> 8)
        a[3] = <uint8_t>self._data
        return <bytes>a[:4]

    @classmethod
    def from_int(cls, src) -> Int32:
        """Creates a new instance from the specified integer.

        Args:
            src: The integer from which a new instance is created.

        Returns:
            A new instance created from the specified integer.

        """
        cdef Int32 o = Int32()
        o._data = <int32_t>src
        return o

    cpdef to_int(self):
        """Returns the native data as an integer.

        Returns:
            An integer that represents the native data.

        """
        return int(self._data)

    @classmethod
    def from_f16(cls, Float16 src, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True) -> Int32:
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
        return f16_to_i32(src, rounding_mode, exact)

    cpdef Float16 to_f16(self):
        """Converts the 32-bit signed integer to an IEEE 754 binary16 floating point.

        The result is the same as that of :func:`i32_to_f16()`.

        Returns:
            The IEEE 754 binary16 floating point.

        """
        return i32_to_f16(self)

    @classmethod
    def from_f32(cls, Float32 src, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True) -> Int32:
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
        return f32_to_i32(src, rounding_mode, exact)

    cpdef Float32 to_f32(self):
        """Converts the 32-bit signed integer to an IEEE 754 binary32 floating point.

        The result is the same as that of :func:`i32_to_f32()`.

        Returns:
            The IEEE 754 binary32 floating point.

        """
        return i32_to_f32(self)

    @classmethod
    def from_f64(cls, Float64 src, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True) -> Int32:
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
        return f64_to_i32(src, rounding_mode, exact)

    cpdef Float64 to_f64(self):
        """Converts the 32-bit signed integer to an IEEE 754 binary64 floating point.

        The result is the same as that of :func:`i32_to_f64()`.

        Returns:
            The IEEE 754 binary64 floating point.

        """
        return i32_to_f64(self)

    @classmethod
    def from_f128(cls, Float128 src, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True) -> Int32:
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
        return f128_to_i32(src, rounding_mode, exact)

    cpdef Float128 to_f128(self):
        """Converts the 32-bit signed integer to an IEEE 754 binary128 floating point.

        The result is the same as that of :func:`i32_to_f128()`.

        Returns:
            The IEEE 754 binary128 floating point.

        """
        return i32_to_f128(self)

    def __str__(self) -> str:
        return str(int(self._data))

    def __pos__(self) -> Self:
        return _make_int32(self._data)

    def __neg__(self) -> Self:
        return _make_int32(-self._data)

    def __invert__(self) -> Self:
        return _make_int32(~self._data)

    def __add__(self, other: Self) -> Self:
        return _make_int32(self._data + other._get_data())

    def __sub__(self, other: Self) -> Self:
        return _make_int32(self._data - other._get_data())

    def __mul__(self, other: Self) -> Self:
        return _make_int32(self._data * other._get_data())

    def __floordiv__(self, other: Self) -> Self:
        return _make_int32(self._data // other._get_data())

    def __mod__(self, other: Self) -> Self:
        return _make_int32(self._data % other._get_data())

    def __lshift__(self, other: Self) -> Self:
        if other._get_data() < 0:
            raise ValueError("negative shift count")
        return _make_int32(self._data << other._get_data())

    def __rshift__(self, other: Self) -> Self:
        if other._get_data() < 0:
            raise ValueError("negative shift count")
        return _make_int32(self._data >> other._get_data())

    def __and__(self, other: Self) -> Self:
        return _make_int32(self._data & other._get_data())

    def __or__(self, other: Self) -> Self:
        return _make_int32(self._data | other._get_data())

    def __xor__(self, other: Self) -> Self:
        return _make_int32(self._data ^ other._get_data())

    def __lt__(self, other: Self) -> bool:
        return self._data < other._get_data()

    def __le__(self, other: Self) -> bool:
        return self._data <= other._get_data()

    def __gt__(self, other: Self) -> bool:
        return self._data > other._get_data()

    def __ge__(self, other: Self) -> bool:
        return self._data >= other._get_data()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            raise ValueError('type mismatch')
        return self._data == other._get_data()

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            raise ValueError('type mismatch')
        return self._data != other._get_data()

    def __iadd__(self, other: Self) -> Self:
        return self.__add__(other)

    def __isub__(self, other: Self) -> Self:
        return self.__sub__(other)

    def __imul__(self, other: Self) -> Self:
        return self.__mul__(other)

    def __ifloordiv__(self, other: Self) -> Self:
        return self.__floordiv__(other)

    def __imod__(self, other: Self) -> Self:
        return self.__mod__(other)

    def __ilshift__(self, other: Self) -> Self:
        return self.__lshift__(other)

    def __irshift__(self, other: Self) -> Self:
        return self.__rshift__(other)

    def __iand__(self, other: Self) -> Self:
        return self.__and__(other)

    def __ior__(self, other: Self) -> Self:
        return self.__or__(other)

    def __ixor__(self, other: Self) -> Self:
        return self.__xor__(other)


cdef class Int64:
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

    cdef int64_t _data
    """The native data."""

    cpdef int64_t _get_data(self):  # To access the native data in another instance.
        return self._data

    @classmethod
    def size(cls) -> int:
        """Returns the native data size in bits.

        Returns:
            The native data size in bits, i.e. 64.

        """
        return 64

    @classmethod
    def from_bytes(cls, bytes src) -> Int64:
        """Creates a new instance from the specified byte sequence.

        Args:
            src: The byte sequence representing the native data with big endian.
                 The length must be 8.

        Returns:
            A new instance created from the specified byte sequence.

        Raises:
            ValueError: If the length of bytes is not 8.

        """
        if len(src) != 8:
            raise ValueError('length of bytes must be 8')
        cdef const uint8_t[:] a = src
        cdef Int64 o = Int64()
        o._data = (
            (<int64_t>a[0] << 56) |
            (<int64_t>a[1] << 48) |
            (<int64_t>a[2] << 40) |
            (<int64_t>a[3] << 32) |
            (<int64_t>a[4] << 24) |
            (<int64_t>a[5] << 16) |
            (<int64_t>a[6] << 8) |
            <int64_t>a[7]
        )
        return o

    cpdef bytes to_bytes(self):
        """Returns the native data as a byte sequence.

        Returns:
            A byte sequence representing the native data with big endian.
            The length is 8.

        """
        cdef uint8_t a[8]
        a[0] = <uint8_t>(self._data >> 56)
        a[1] = <uint8_t>(self._data >> 48)
        a[2] = <uint8_t>(self._data >> 40)
        a[3] = <uint8_t>(self._data >> 32)
        a[4] = <uint8_t>(self._data >> 24)
        a[5] = <uint8_t>(self._data >> 16)
        a[6] = <uint8_t>(self._data >> 8)
        a[7] = <uint8_t>self._data
        return <bytes>a[:8]

    @classmethod
    def from_int(cls, src) -> Int64:
        """Creates a new instance from the specified integer.

        Args:
            src: The integer from which a new instance is created.

        Returns:
            A new instance created from the specified integer.

        """
        cdef Int64 o = Int64()
        o._data = <int64_t>src
        return o

    cpdef to_int(self):
        """Returns the native data as an integer.

        Returns:
            An integer that represents the native data.

        """
        return int(self._data)

    @classmethod
    def from_f16(cls, Float16 src, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True) -> Int64:
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
        return f16_to_i64(src, rounding_mode, exact)

    cpdef Float16 to_f16(self):
        """Converts the 64-bit signed integer to an IEEE 754 binary16 floating point.

        The result is the same as that of :func:`i64_to_f16()`.

        Returns:
            The IEEE 754 binary16 floating point.

        """
        return i64_to_f16(self)

    @classmethod
    def from_f32(cls, Float32 src, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True) -> Int64:
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
        return f32_to_i64(src, rounding_mode, exact)

    cpdef Float32 to_f32(self):
        """Converts the 64-bit signed integer to an IEEE 754 binary32 floating point.

        The result is the same as that of :func:`i64_to_f32()`.

        Returns:
            The IEEE 754 binary32 floating point.

        """
        return i64_to_f32(self)

    @classmethod
    def from_f64(cls, Float64 src, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True) -> Int64:
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
        return f64_to_i64(src, rounding_mode, exact)

    cpdef Float64 to_f64(self):
        """Converts the 64-bit signed integer to an IEEE 754 binary64 floating point.

        The result is the same as that of :func:`i64_to_f64()`.

        Returns:
            The IEEE 754 binary64 floating point.

        """
        return i64_to_f64(self)

    @classmethod
    def from_f128(cls, Float128 src, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True) -> Int64:
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
        return f128_to_i64(src, rounding_mode, exact)

    cpdef Float128 to_f128(self):
        """Converts the 64-bit signed integer to an IEEE 754 binary128 floating point.

        The result is the same as that of :func:`i64_to_f128()`.

        Returns:
            The IEEE 754 binary128 floating point.

        """
        return i64_to_f128(self)

    def __str__(self) -> str:
        return str(int(self._data))

    def __pos__(self) -> Self:
        return _make_int64(self._data)

    def __neg__(self) -> Self:
        return _make_int64(-self._data)

    def __invert__(self) -> Self:
        return _make_int64(~self._data)

    def __add__(self, other: Self) -> Self:
        return _make_int64(self._data + other._get_data())

    def __sub__(self, other: Self) -> Self:
        return _make_int64(self._data - other._get_data())

    def __mul__(self, other: Self) -> Self:
        return _make_int64(self._data * other._get_data())

    def __floordiv__(self, other: Self) -> Self:
        return _make_int64(self._data // other._get_data())

    def __mod__(self, other: Self) -> Self:
        return _make_int64(self._data % other._get_data())

    def __lshift__(self, other: Self) -> Self:
        if other._get_data() < 0:
            raise ValueError("negative shift count")
        return _make_int64(self._data << other._get_data())

    def __rshift__(self, other: Self) -> Self:
        if other._get_data() < 0:
            raise ValueError("negative shift count")
        return _make_int64(self._data >> other._get_data())

    def __and__(self, other: Self) -> Self:
        return _make_int64(self._data & other._get_data())

    def __or__(self, other: Self) -> Self:
        return _make_int64(self._data | other._get_data())

    def __xor__(self, other: Self) -> Self:
        return _make_int64(self._data ^ other._get_data())

    def __lt__(self, other: Self) -> bool:
        return self._data < other._get_data()

    def __le__(self, other: Self) -> bool:
        return self._data <= other._get_data()

    def __gt__(self, other: Self) -> bool:
        return self._data > other._get_data()

    def __ge__(self, other: Self) -> bool:
        return self._data >= other._get_data()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            raise ValueError('type mismatch')
        return self._data == other._get_data()

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            raise ValueError('type mismatch')
        return self._data != other._get_data()

    def __iadd__(self, other: Self) -> Self:
        return self.__add__(other)

    def __isub__(self, other: Self) -> Self:
        return self.__sub__(other)

    def __imul__(self, other: Self) -> Self:
        return self.__mul__(other)

    def __ifloordiv__(self, other: Self) -> Self:
        return self.__floordiv__(other)

    def __imod__(self, other: Self) -> Self:
        return self.__mod__(other)

    def __ilshift__(self, other: Self) -> Self:
        return self.__lshift__(other)

    def __irshift__(self, other: Self) -> Self:
        return self.__rshift__(other)

    def __iand__(self, other: Self) -> Self:
        return self.__and__(other)

    def __ior__(self, other: Self) -> Self:
        return self.__or__(other)

    def __ixor__(self, other: Self) -> Self:
        return self.__xor__(other)


cdef class BFloat16:
    """A 16-bit brain floating point.

    The object is immutable.

    No operators are supported.

    """

    cdef sf.bfloat16_t _data
    """The native data."""

    cpdef sf.bfloat16_t _get_data(self):  # To access the native data in another instance.
        return self._data

    @classmethod
    def size(cls) -> int:
        """Returns the native data size in bits.

        Returns:
            The native data size in bits, i.e. 16.

        """
        return 16

    @classmethod
    def from_bytes(cls, bytes src) -> BFloat16:
        """Creates a new instance from the specified byte sequence.

        Args:
            src: The byte sequence representing the native data with big endian.
                 The length must be 2.

        Returns:
            A new instance created from the specified byte sequence.

        Raises:
            ValueError: If the length of bytes is not 2.

        """
        if len(src) != 2:
            raise ValueError('length of bytes must be 2')
        cdef const uint8_t[:] a = src
        cdef BFloat16 o = BFloat16()
        o._data.v = (<uint16_t>a[0] << 8) | <uint16_t>a[1]
        return o

    cpdef bytes to_bytes(self):
        """Returns the native data as a byte sequence.

        Returns:
            A byte sequence representing the native data with big endian.
            The length is 2.

        """
        cdef uint8_t a[2]
        a[0] = <uint8_t>(self._data.v >> 8)
        a[1] = <uint8_t>self._data.v
        return <bytes>a[:2]

    @classmethod
    def from_float(cls, double src) -> BFloat16:
        """Creates a new instance from the specified floating point.

        Args:
            src: The floating point from which a new instance is created.

        Returns:
            A new instance created from the specified floating point.

        """
        cdef _ui64_double t
        cdef sf.float64_t f
        t.f = src
        f.v = t.ui
        cdef BFloat16 o = BFloat16()
        o._data = sf.f32_to_bf16(sf.f64_to_f32(f))
        return o

    cpdef double to_float(self):
        """Returns the native data as a floating point.

        Returns:
            A floating point that represents the native data.

        """
        cdef _ui64_double t
        t.ui = sf.f32_to_f64(sf.bf16_to_f32(self._data)).v
        return t.f

    @classmethod
    def from_f32(cls, Float32 src) -> BFloat16:
        """Creates a new instance from the specified IEEE 754 binary32 floating point.

        The result is the same as that of :func:`f32_to_bf16()`.

        Args:
            src: The IEEE 754 binary32 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary32 floating point.

        """
        return f32_to_bf16(src)

    cpdef Float32 to_f32(self):
        """Converts the 16-bit brain floating point to an IEEE 754 binary32 floating point.

        The result is the same as that of :func:`bf16_to_f32()`.

        Returns:
            The IEEE 754 binary32 floating point.

        """
        return bf16_to_f32(self)

    cpdef bool is_signaling_nan(self):
        """Tests if the 16-bit brain floating point is a signaling NaN.

        The result is the same as that of :func:`bf16_is_signaling_nan()`.

        Returns:
            ``True`` if the floating point is a signaling NaN, ``False`` otherwise.

        """
        return bf16_is_signaling_nan(self)

    cpdef bool is_nan(self):
        """Tests if the 16-bit brain floating point is a NaN.

        The result is the same as that of :func:`bf16_is_nan()`.

        Returns:
            ``True`` if the floating point is a NaN, ``False`` otherwise.

        """
        return bf16_is_nan(self)

    cpdef bool is_inf(self):
        """Tests if the 16-bit brain floating point is an infinity.

        The result is the same as that of :func:`bf16_is_inf()`.

        Returns:
            ``True`` if the floating point is an infinity, ``False`` otherwise.

        """
        return bf16_is_inf(self)

    def __str__(self) -> str:
        return str(self.to_float())


cdef class Float16:
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

    cdef sf.float16_t _data
    """The native data."""

    cpdef sf.float16_t _get_data(self):  # To access the native data in another instance.
        return self._data

    @classmethod
    def size(cls) -> int:
        """Returns the native data size in bits.

        Returns:
            The native data size in bits, i.e. 16.

        """
        return 16

    @classmethod
    def from_bytes(cls, bytes src) -> Float16:
        """Creates a new instance from the specified byte sequence.

        Args:
            src: The byte sequence representing the native data with big endian.
                 The length must be 2.

        Returns:
            A new instance created from the specified byte sequence.

        Raises:
            ValueError: If the length of bytes is not 2.

        """
        if len(src) != 2:
            raise ValueError('length of bytes must be 2')
        cdef const uint8_t[:] a = src
        cdef Float16 o = Float16()
        o._data.v = (<uint16_t>a[0] << 8) | <uint16_t>a[1]
        return o

    cpdef bytes to_bytes(self):
        """Returns the native data as a byte sequence.

        Returns:
            A byte sequence representing the native data with big endian.
            The length is 2.

        """
        cdef uint8_t a[2]
        a[0] = <uint8_t>(self._data.v >> 8)
        a[1] = <uint8_t>self._data.v
        return <bytes>a[:2]

    @classmethod
    def from_float(cls, double src) -> Float16:
        """Creates a new instance from the specified floating point.

        Args:
            src: The floating point from which a new instance is created.

        Returns:
            A new instance created from the specified floating point.

        """
        cdef _ui64_double t
        cdef sf.float64_t f
        t.f = src
        f.v = t.ui
        cdef Float16 o = Float16()
        o._data = sf.f64_to_f16(f)
        return o

    cpdef double to_float(self):
        """Returns the native data as a floating point.

        Returns:
            A floating point that represents the native data.

        """
        cdef _ui64_double t
        t.ui = sf.f16_to_f64(self._data).v
        return t.f

    @classmethod
    def from_ui32(cls, UInt32 src) -> Float16:
        """Creates a new instance from the specified 32-bit unsigned integer.

        The result is the same as that of :func:`ui32_to_f16()`.

        Args:
            src: The 32-bit unsigned integer from which a new instance is created.

        Returns:
            A new instance created from the specified 32-bit unsigned integer.

        """
        return ui32_to_f16(src)

    cpdef UInt32 to_ui32(
        self, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
    ):
        """Converts the IEEE 754 binary16 floating point to a 32-bit unsigned integer.

        The result is the same as that of :func:`f16_to_ui32()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 32-bit unsigned integer.

        """
        return f16_to_ui32(self, rounding_mode, exact)

    @classmethod
    def from_ui64(cls, UInt64 src) -> Float16:
        """Creates a new instance from the specified 64-bit unsigned integer.

        The result is the same as that of :func:`ui64_to_f16()`.

        Args:
            src: The 64-bit unsigned integer from which a new instance is created.

        Returns:
            A new instance created from the specified 64-bit unsigned integer.

        """
        return ui64_to_f16(src)

    cpdef UInt64 to_ui64(
        self, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
    ):
        """Converts the IEEE 754 binary16 floating point to a 64-bit unsigned integer.

        The result is the same as that of :func:`f16_to_ui64()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 64-bit unsigned integer.

        """
        return f16_to_ui64(self, rounding_mode, exact)

    @classmethod
    def from_i32(cls, Int32 src) -> Float16:
        """Creates a new instance from the specified 32-bit signed integer.

        The result is the same as that of :func:`i32_to_f16()`.

        Args:
            src: The 32-bit signed integer from which a new instance is created.

        Returns:
            A new instance created from the specified 32-bit signed integer.

        """
        return i32_to_f16(src)

    cpdef Int32 to_i32(
        self, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
    ):
        """Converts the IEEE 754 binary16 floating point to a 32-bit signed integer.

        The result is the same as that of :func:`f16_to_i32()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 32-bit signed integer.

        """
        return f16_to_i32(self, rounding_mode, exact)

    @classmethod
    def from_i64(cls, Int64 src) -> Float16:
        """Creates a new instance from the specified 64-bit signed integer.

        The result is the same as that of :func:`i64_to_f16()`.

        Args:
            src: The 64-bit signed integer from which a new instance is created.

        Returns:
            A new instance created from the specified 64-bit signed integer.

        """
        return i64_to_f16(src)

    cpdef Int64 to_i64(
        self, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
    ):
        """Converts the IEEE 754 binary16 floating point to a 64-bit signed integer.

        The result is the same as that of :func:`f16_to_i64()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 64-bit signed integer.

        """
        return f16_to_i64(self, rounding_mode, exact)

    @classmethod
    def from_f16(cls, Float16 src) -> Float16:
        """Creates a new instance from the specified IEEE 754 binary16 floating point.

        The result is a copy of the specified instance.

        Args:
            src: The IEEE 754 binary16 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary16 floating point.

        """
        return src.__pos__()

    cpdef Float16 to_f16(self):
        """Converts the IEEE 754 binary16 floating point to a binary16 floating point.

        The result is a copy.

        Returns:
            The IEEE 754 binary16 floating point.

        """
        return self.__pos__()

    @classmethod
    def from_f32(cls, Float32 src) -> Float16:
        """Creates a new instance from the specified IEEE 754 binary32 floating point.

        The result is the same as that of :func:`f32_to_f16()`.

        Args:
            src: The IEEE 754 binary32 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary32 floating point.

        """
        return f32_to_f16(src)

    cpdef Float32 to_f32(self):
        """Converts the IEEE 754 binary16 floating point to a binary32 floating point.

        The result is the same as that of :func:`f16_to_f32()`.

        Returns:
            The IEEE 754 binary32 floating point.

        """
        return f16_to_f32(self)

    @classmethod
    def from_f64(cls, Float64 src) -> Float16:
        """Creates a new instance from the specified IEEE 754 binary64 floating point.

        The result is the same as that of :func:`f64_to_f16()`.

        Args:
            src: The IEEE 754 binary64 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary64 floating point.

        """
        return f64_to_f16(src)

    cpdef Float64 to_f64(self):
        """Converts the IEEE 754 binary16 floating point to a binary64 floating point.

        The result is the same as that of :func:`f16_to_f64()`.

        Returns:
            The IEEE 754 binary64 floating point.

        """
        return f16_to_f64(self)

    @classmethod
    def from_f128(cls, Float128 src) -> Float16:
        """Creates a new instance from the specified IEEE 754 binary128 floating point.

        The result is the same as that of :func:`f128_to_f16()`.

        Args:
            src: The IEEE 754 binary128 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary128 floating point.

        """
        return f128_to_f16(src)

    cpdef Float128 to_f128(self):
        """Converts the IEEE 754 binary16 floating point to a binary128 floating point.

        The result is the same as that of :func:`f16_to_f128()`.

        Returns:
            The IEEE 754 binary128 floating point.

        """
        return f16_to_f128(self)

    cpdef Float16 round_to_int(
        self, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
    ):
        """Rounds the number.

        The result is the same as that of :func:`f16_round_to_int()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact rounding is unable.

        Returns:
            The resulted integer.

        """
        return f16_round_to_int(self, rounding_mode, exact)

    cpdef Float16 neg(self):
        """Negates the IEEE 754 binary16 floating point.

        The result is the same as that of :func:`f16_neg()`.

        Returns:
            The resulted number (``-x``).

        """
        return f16_neg(self)

    @classmethod
    def add(cls, Float16 x, Float16 y) -> Float16:
        """Adds the IEEE 754 binary16 floating points.

        The result is the same as that of :func:`f16_add()`.

        Args:
            x: The floating point to be added.
            y: The floating point to add.

        Returns:
            The resulted number (``x + y``).

        """
        return f16_add(x, y)

    @classmethod
    def sub(cls, Float16 x, Float16 y) -> Float16:
        """Subtracts the IEEE 754 binary16 floating points.

        The result is the same as that of :func:`f16_sub()`.

        Args:
            x: The floating point to be subtracted.
            y: The floating point to subtract.

        Returns:
            The resulted number (``x - y``).

        """
        return f16_sub(x, y)

    @classmethod
    def mul(cls, Float16 x, Float16 y) -> Float16:
        """Multiplies the IEEE 754 binary16 floating points.

        The result is the same as that of :func:`f16_mul()`.

        Args:
            x: The floating point to be multiplied.
            y: The floating point to multiply.

        Returns:
            The resulted number (``x * y``).

        """
        return f16_mul(x, y)

    @classmethod
    def mul_add(cls, Float16 x, Float16 y, Float16 z) -> Float16:
        """Multiplies and Adds the IEEE 754 binary16 floating points.

        The result is the same as that of :func:`f16_mul_add()`.

        Args:
            x: The floating point to be multiplied.
            y: The floating point to multiply.
            z: The floating point to add.

        Returns:
            The resulted number (``x * y + z``).

        """
        return f16_mul_add(x, y, z)

    @classmethod
    def div(cls, Float16 x, Float16 y) -> Float16:
        """Divides the IEEE 754 binary16 floating points.

        The result is the same as that of :func:`f16_div()`.

        Args:
            x: The floating point to be divided.
            y: The floating point to divide.

        Returns:
            The resulted number (``x / y``).

        """
        return f16_div(x, y)

    @classmethod
    def rem(cls, Float16 x, Float16 y) -> Float16:
        """Calculates a remainder by dividing the IEEE 754 binary16 floating points.

        The result is the same as that of :func:`f16_rem()`.

        Args:
            x: The floating point to be divided.
            y: The floating point to divide.

        Returns:
            The resulted number (``x % y``).

        """
        return f16_rem(x, y)

    cpdef Float16 sqrt(self):
        """Calculates a square root of the IEEE 754 binary16 floating point.

        The result is the same as that of :func:`f16_sqrt()`.

        Returns:
            The resulted number (``sqrt(x)``).

        """
        return f16_sqrt(self)

    @classmethod
    def eq(cls, Float16 x, Float16 y) -> bool:
        """Tests if the first one is equal to the second one expressed as IEEE 754 binary16 floating points.

        The result is the same as that of :func:`f16_eq()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

        """
        return f16_eq(x, y)

    @classmethod
    def le(cls, Float16 x, Float16 y) -> bool:
        """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary16 floating points.

        The result is the same as that of :func:`f16_le()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

        """
        return f16_le(x, y)

    @classmethod
    def lt(cls, Float16 x, Float16 y) -> bool:
        """Tests if the first one is less than the second one expressed as IEEE 754 binary16 floating points.

        The result is the same as that of :func:`f16_lt()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

        """
        return f16_lt(x, y)

    @classmethod
    def eq_signaling(cls, Float16 x, Float16 y) -> bool:
        """Tests if the first one is equal to the second one expressed as IEEE 754 binary16 floating points.

        The invalid exception flag is set for any NaN input, not just for signaling NaNs.

        The result is the same as that of :func:`f16_eq_signaling()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

        """
        return f16_eq_signaling(x, y)

    @classmethod
    def le_quiet(cls, Float16 x, Float16 y) -> bool:
        """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary16 floating points.

        The invalid exception flag is not set for quiet NaNs.

        The result is the same as that of :func:`f16_le_quiet()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

        """
        return f16_le_quiet(x, y)

    @classmethod
    def lt_quiet(cls, Float16 x, Float16 y) -> bool:
        """Tests if the first one is less than the second one expressed as IEEE 754 binary16 floating points.

        The invalid exception flag is not set for quiet NaNs.

        The result is the same as that of :func:`f16_lt_quiet()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

        """
        return f16_lt_quiet(x, y)

    cpdef bool is_signaling_nan(self):
        """Tests if the IEEE 754 binary16 floating point is a signaling NaN.

        The result is the same as that of :func:`f16_is_signaling_nan()`.

        Returns:
            ``True`` if the floating point is a signaling NaN, ``False`` otherwise.

        """
        return f16_is_signaling_nan(self)

    cpdef bool is_nan(self):
        """Tests if the IEEE 754 binary16 floating point is a NaN.

        The result is the same as that of :func:`f16_is_nan()`.

        Returns:
            ``True`` if the floating point is a NaN, ``False`` otherwise.

        """
        return f16_is_nan(self)

    cpdef bool is_inf(self):
        """Tests if the IEEE 754 binary16 floating point is an infinity.

        The result is the same as that of :func:`f16_is_inf()`.

        Returns:
            ``True`` if the floating point is an infinity, ``False`` otherwise.

        """
        return f16_is_inf(self)

    def __str__(self) -> str:
        return str(self.to_float())

    def __pos__(self) -> Self:
        return _make_float16(self._data)

    def __neg__(self) -> Self:
        return f16_neg(self)

    def __add__(self, other: Self) -> Self:
        return f16_add(self, other)

    def __sub__(self, other: Self) -> Self:
        return f16_sub(self, other)

    def __mul__(self, other: Self) -> Self:
        return f16_mul(self, other)

    def __truediv__(self, other: Self) -> Self:
        return f16_div(self, other)

    def __floordiv__(self, other: Self) -> Self:
        return f16_round_to_int(f16_div(self, other), RoundingMode.MIN)

    def __mod__(self, other: Self) -> Self:
        return f16_sub(self, f16_mul(other, f16_round_to_int(f16_div(self, other), RoundingMode.MIN)))

    def __lt__(self, other: Self) -> bool:
        return f16_lt(self, other)

    def __le__(self, other: Self) -> bool:
        return f16_le(self, other)

    def __gt__(self, other: Self) -> bool:
        return f16_lt(other, self)

    def __ge__(self, other: Self) -> bool:
        return f16_le(other, self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            raise ValueError('type mismatch')
        return f16_eq(self, other)

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            raise ValueError('type mismatch')
        return not f16_eq(self, other)

    def __iadd__(self, other: Self) -> Self:
        return self.__add__(other)

    def __isub__(self, other: Self) -> Self:
        return self.__sub__(other)

    def __imul__(self, other: Self) -> Self:
        return self.__mul__(other)

    def __itruediv__(self, other: Self) -> Self:
        return self.__truediv__(other)

    def __ifloordiv__(self, other: Self) -> Self:
        return self.__floordiv__(other)

    def __imod__(self, other: Self) -> Self:
        return self.__mod__(other)


cdef class Float32:
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

    cdef sf.float32_t _data
    """The native data."""

    cpdef sf.float32_t _get_data(self):  # To access the native data in another instance.
        return self._data

    @classmethod
    def size(cls) -> int:
        """Returns the native data size in bits.

        Returns:
            The native data size in bits, i.e. 32.

        """
        return 32

    @classmethod
    def from_bytes(cls, bytes src) -> Float32:
        """Creates a new instance from the specified byte sequence.

        Args:
            src: The byte sequence representing the native data with big endian.
                 The length must be 4.

        Returns:
            A new instance created from the specified byte sequence.

        Raises:
            ValueError: If the length of bytes is not 4.

        """
        if len(src) != 4:
            raise ValueError('length of bytes must be 4')
        cdef const uint8_t[:] a = src
        cdef Float32 o = Float32()
        o._data.v = (
            (<uint32_t>a[0] << 24) |
            (<uint32_t>a[1] << 16) |
            (<uint32_t>a[2] << 8) |
            <uint32_t>a[3]
        )
        return o

    cpdef bytes to_bytes(self):
        """Returns the native data as a byte sequence.

        Returns:
            A byte sequence representing the native data with big endian.
            The length is 4.

        """
        cdef uint8_t a[4]
        a[0] = <uint8_t>(self._data.v >> 24)
        a[1] = <uint8_t>(self._data.v >> 16)
        a[2] = <uint8_t>(self._data.v >> 8)
        a[3] = <uint8_t>self._data.v
        return <bytes>a[:4]

    @classmethod
    def from_float(cls, double src) -> Float32:
        """Creates a new instance from the specified floating point.

        Args:
            src: The floating point from which a new instance is created.

        Returns:
            A new instance created from the specified floating point.

        """
        cdef _ui64_double t
        cdef sf.float64_t f
        t.f = src
        f.v = t.ui
        cdef Float32 o = Float32()
        o._data = sf.f64_to_f32(f)
        return o

    cpdef double to_float(self):
        """Returns the native data as a floating point.

        Returns:
            A floating point that represents the native data.

        """
        cdef _ui64_double t
        t.ui = sf.f32_to_f64(self._data).v
        return t.f

    @classmethod
    def from_bf16(cls, BFloat16 src) -> Float32:
        """Creates a new instance from the 16-bit brain floating point.

        The result is the same as that of :func:`bf16_to_f32()`.

        Args:
            src: The 16-bit brain floating point from which a new instance is created.

        Returns:
            A new instance created from the specified 16-bit brain floating point.

        """
        return bf16_to_f32(src)

    cpdef BFloat16 to_bf16(self):
        """Converts the IEEE 754 binary32 floating point to a 16-bit brain floating point.

        The result is the same as that of :func:`f32_to_bf16()`.

        Returns:
            The 16-bit brain floating point.

        """
        return f32_to_bf16(self)

    @classmethod
    def from_ui32(cls, UInt32 src) -> Float32:
        """Creates a new instance from the specified 32-bit unsigned integer.

        The result is the same as that of :func:`ui32_to_f32()`.

        Args:
            src: The 32-bit unsigned integer from which a new instance is created.

        Returns:
            A new instance created from the specified 32-bit unsigned integer.

        """
        return ui32_to_f32(src)

    cpdef UInt32 to_ui32(
        self, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
    ):
        """Converts the IEEE 754 binary32 floating point to a 32-bit unsigned integer.

        The result is the same as that of :func:`f32_to_ui32()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 32-bit unsigned integer.

        """
        return f32_to_ui32(self, rounding_mode, exact)

    @classmethod
    def from_ui64(cls, UInt64 src) -> Float32:
        """Creates a new instance from the specified 64-bit unsigned integer.

        The result is the same as that of :func:`ui64_to_f32()`.

        Args:
            src: The 64-bit unsigned integer from which a new instance is created.

        Returns:
            A new instance created from the specified 64-bit unsigned integer.

        """
        return ui64_to_f32(src)

    cpdef UInt64 to_ui64(
        self, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
    ):
        """Converts the IEEE 754 binary32 floating point to a 64-bit unsigned integer.

        The result is the same as that of :func:`f32_to_ui64()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 64-bit unsigned integer.

        """
        return f32_to_ui64(self, rounding_mode, exact)

    @classmethod
    def from_i32(cls, Int32 src) -> Float32:
        """Creates a new instance from the specified 32-bit signed integer.

        The result is the same as that of :func:`i32_to_f32()`.

        Args:
            src: The 32-bit signed integer from which a new instance is created.

        Returns:
            A new instance created from the specified 32-bit signed integer.

        """
        return i32_to_f32(src)

    cpdef Int32 to_i32(
        self, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
    ):
        """Converts the IEEE 754 binary32 floating point to a 32-bit signed integer.

        The result is the same as that of :func:`f32_to_i32()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 32-bit signed integer.

        """
        return f32_to_i32(self, rounding_mode, exact)

    @classmethod
    def from_i64(cls, Int64 src) -> Float32:
        """Creates a new instance from the specified 64-bit signed integer.

        The result is the same as that of :func:`i64_to_f32()`.

        Args:
            src: The 64-bit signed integer from which a new instance is created.

        Returns:
            A new instance created from the specified 64-bit signed integer.

        """
        return i64_to_f32(src)

    cpdef Int64 to_i64(
        self, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
    ):
        """Converts the IEEE 754 binary32 floating point to a 64-bit signed integer.

        The result is the same as that of :func:`f32_to_i64()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 64-bit signed integer.

        """
        return f32_to_i64(self, rounding_mode, exact)

    @classmethod
    def from_f16(cls, Float16 src) -> Float32:
        """Creates a new instance from the specified IEEE 754 binary16 floating point.

        The result is the same as that of :func:`f16_to_f32()`.

        Args:
            src: The IEEE 754 binary16 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary16 floating point.

        """
        return f16_to_f32(src)

    cpdef Float16 to_f16(self):
        """Converts the IEEE 754 binary32 floating point to a binary16 floating point.

        The result is the same as that of :func:`f32_to_f16()`.

        Returns:
            The IEEE 754 binary16 floating point.

        """
        return f32_to_f16(self)

    @classmethod
    def from_f32(cls, Float32 src) -> Float32:
        """Creates a new instance from the specified IEEE 754 binary32 floating point.

        The result is a copy of the specified instance.

        Args:
            src: The IEEE 754 binary32 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary32 floating point.

        """
        return src.__pos__()

    cpdef Float32 to_f32(self):
        """Converts the IEEE 754 binary32 floating point to a binary32 floating point.

        The result is a copy.

        Returns:
            The IEEE 754 binary32 floating point.

        """
        return self.__pos__()

    @classmethod
    def from_f64(cls, Float64 src) -> Float32:
        """Creates a new instance from the specified IEEE 754 binary64 floating point.

        The result is the same as that of :func:`f64_to_f32()`.

        Args:
            src: The IEEE 754 binary64 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary64 floating point.

        """
        return f64_to_f32(src)

    cpdef Float64 to_f64(self):
        """Converts the IEEE 754 binary32 floating point to a binary64 floating point.

        The result is the same as that of :func:`f32_to_f64()`.

        Returns:
            The IEEE 754 binary64 floating point.

        """
        return f32_to_f64(self)

    @classmethod
    def from_f128(cls, Float128 src) -> Float32:
        """Creates a new instance from the specified IEEE 754 binary128 floating point.

        The result is the same as that of :func:`f128_to_f32()`.

        Args:
            src: The IEEE 754 binary128 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary128 floating point.

        """
        return f128_to_f32(src)

    cpdef Float128 to_f128(self):
        """Converts the IEEE 754 binary32 floating point to a binary128 floating point.

        The result is the same as that of :func:`f32_to_f128()`.

        Returns:
            The IEEE 754 binary128 floating point.

        """
        return f32_to_f128(self)

    cpdef Float32 round_to_int(
        self, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
    ):
        """Rounds the number.

        The result is the same as that of :func:`f32_round_to_int()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact rounding is unable.

        Returns:
            The resulted integer.

        """
        return f32_round_to_int(self, rounding_mode, exact)

    cpdef Float32 neg(self):
        """Negates the IEEE 754 binary32 floating point.

        The result is the same as that of :func:`f32_neg()`.

        Returns:
            The resulted number (``-x``).

        """
        return f32_neg(self)

    @classmethod
    def add(cls, Float32 x, Float32 y) -> Float32:
        """Adds the IEEE 754 binary32 floating points.

        The result is the same as that of :func:`f32_add()`.

        Args:
            x: The floating point to be added.
            y: The floating point to add.

        Returns:
            The resulted number (``x + y``).

        """
        return f32_add(x, y)

    @classmethod
    def sub(cls, Float32 x, Float32 y) -> Float32:
        """Subtracts the IEEE 754 binary32 floating points.

        The result is the same as that of :func:`f32_sub()`.

        Args:
            x: The floating point to be subtracted.
            y: The floating point to subtract.

        Returns:
            The resulted number (``x - y``).

        """
        return f32_sub(x, y)

    @classmethod
    def mul(cls, Float32 x, Float32 y) -> Float32:
        """Multiplies the IEEE 754 binary32 floating points.

        The result is the same as that of :func:`f32_mul()`.

        Args:
            x: The floating point to be multiplied.
            y: The floating point to multiply.

        Returns:
            The resulted number (``x * y``).

        """
        return f32_mul(x, y)

    @classmethod
    def mul_add(cls, Float32 x, Float32 y, Float32 z) -> Float32:
        """Multiplies and Adds the IEEE 754 binary32 floating points.

        The result is the same as that of :func:`f32_mul_add()`.

        Args:
            x: The floating point to be multiplied.
            y: The floating point to multiply.
            z: The floating point to add.

        Returns:
            The resulted number (``x * y + z``).

        """
        return f32_mul_add(x, y, z)

    @classmethod
    def div(cls, Float32 x, Float32 y) -> Float32:
        """Divides the IEEE 754 binary32 floating points.

        The result is the same as that of :func:`f32_div()`.

        Args:
            x: The floating point to be divided.
            y: The floating point to divide.

        Returns:
            The resulted number (``x / y``).

        """
        return f32_div(x, y)

    @classmethod
    def rem(cls, Float32 x, Float32 y) -> Float32:
        """Calculates a remainder by dividing the IEEE 754 binary32 floating points.

        The result is the same as that of :func:`f32_rem()`.

        Args:
            x: The floating point to be divided.
            y: The floating point to divide.

        Returns:
            The resulted number (``x % y``).

        """
        return f32_rem(x, y)

    cpdef Float32 sqrt(self):
        """Calculates a square root of the IEEE 754 binary32 floating point.

        The result is the same as that of :func:`f32_sqrt()`.

        Returns:
            The resulted number (``sqrt(x)``).

        """
        return f32_sqrt(self)

    @classmethod
    def eq(cls, Float32 x, Float32 y) -> bool:
        """Tests if the first one is equal to the second one expressed as IEEE 754 binary32 floating points.

        The result is the same as that of :func:`f32_eq()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

        """
        return f32_eq(x, y)

    @classmethod
    def le(cls, Float32 x, Float32 y) -> bool:
        """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary32 floating points.

        The result is the same as that of :func:`f32_le()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

        """
        return f32_le(x, y)

    @classmethod
    def lt(cls, Float32 x, Float32 y) -> bool:
        """Tests if the first one is less than the second one expressed as IEEE 754 binary32 floating points.

        The result is the same as that of :func:`f32_lt()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

        """
        return f32_lt(x, y)

    @classmethod
    def eq_signaling(cls, Float32 x, Float32 y) -> bool:
        """Tests if the first one is equal to the second one expressed as IEEE 754 binary32 floating points.

        The invalid exception flag is set for any NaN input, not just for signaling NaNs.

        The result is the same as that of :func:`f32_eq_signaling()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

        """
        return f32_eq_signaling(x, y)

    @classmethod
    def le_quiet(cls, Float32 x, Float32 y) -> bool:
        """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary32 floating points.

        The invalid exception flag is not set for quiet NaNs.

        The result is the same as that of :func:`f32_le_quiet()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

        """
        return f32_le_quiet(x, y)

    @classmethod
    def lt_quiet(cls, Float32 x, Float32 y) -> bool:
        """Tests if the first one is less than the second one expressed as IEEE 754 binary32 floating points.

        The invalid exception flag is not set for quiet NaNs.

        The result is the same as that of :func:`f32_lt_quiet()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

        """
        return f32_lt_quiet(x, y)

    cpdef bool is_signaling_nan(self):
        """Tests if the IEEE 754 binary32 floating point is a signaling NaN.

        The result is the same as that of :func:`f32_is_signaling_nan()`.

        Returns:
            ``True`` if the floating point is a signaling NaN, ``False`` otherwise.

        """
        return f32_is_signaling_nan(self)

    cpdef bool is_nan(self):
        """Tests if the IEEE 754 binary32 floating point is a NaN.

        The result is the same as that of :func:`f32_is_nan()`.

        Returns:
            ``True`` if the floating point is a NaN, ``False`` otherwise.

        """
        return f32_is_nan(self)

    cpdef bool is_inf(self):
        """Tests if the IEEE 754 binary32 floating point is an infinity.

        The result is the same as that of :func:`f32_is_inf()`.

        Returns:
            ``True`` if the floating point is an infinity, ``False`` otherwise.

        """
        return f32_is_inf(self)

    def __str__(self) -> str:
        return str(self.to_float())

    def __pos__(self) -> Self:
        return _make_float32(self._data)

    def __neg__(self) -> Self:
        return f32_neg(self)

    def __add__(self, other: Self) -> Self:
        return f32_add(self, other)

    def __sub__(self, other: Self) -> Self:
        return f32_sub(self, other)

    def __mul__(self, other: Self) -> Self:
        return f32_mul(self, other)

    def __truediv__(self, other: Self) -> Self:
        return f32_div(self, other)

    def __floordiv__(self, other: Self) -> Self:
        return f32_round_to_int(f32_div(self, other), RoundingMode.MIN)

    def __mod__(self, other: Self) -> Self:
        return f32_sub(self, f32_mul(other, f32_round_to_int(f32_div(self, other), RoundingMode.MIN)))

    def __lt__(self, other: Self) -> bool:
        return f32_lt(self, other)

    def __le__(self, other: Self) -> bool:
        return f32_le(self, other)

    def __gt__(self, other: Self) -> bool:
        return f32_lt(other, self)

    def __ge__(self, other: Self) -> bool:
        return f32_le(other, self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            raise ValueError('type mismatch')
        return f32_eq(self, other)

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            raise ValueError('type mismatch')
        return not f32_eq(self, other)

    def __iadd__(self, other: Self) -> Self:
        return self.__add__(other)

    def __isub__(self, other: Self) -> Self:
        return self.__sub__(other)

    def __imul__(self, other: Self) -> Self:
        return self.__mul__(other)

    def __itruediv__(self, other: Self) -> Self:
        return self.__truediv__(other)

    def __ifloordiv__(self, other: Self) -> Self:
        return self.__floordiv__(other)

    def __imod__(self, other: Self) -> Self:
        return self.__mod__(other)


cdef class Float64:
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

    cdef sf.float64_t _data
    """The native data."""

    cpdef sf.float64_t _get_data(self):  # To access the native data in another instance.
        return self._data

    @classmethod
    def size(cls) -> int:
        """Returns the native data size in bits.

        Returns:
            The native data size in bits, i.e. 64.

        """
        return 64

    @classmethod
    def from_bytes(cls, bytes src) -> Float64:
        """Creates a new instance from the specified byte sequence.

        Args:
            src: The byte sequence representing the native data with big endian.
                 The length must be 8.

        Returns:
            A new instance created from the specified byte sequence.

        Raises:
            ValueError: If the length of bytes is not 8.

        """
        if len(src) != 8:
            raise ValueError('length of bytes must be 8')
        cdef const uint8_t[:] a = src
        cdef Float64 o = Float64()
        o._data.v = (
            (<uint64_t>a[0] << 56) |
            (<uint64_t>a[1] << 48) |
            (<uint64_t>a[2] << 40) |
            (<uint64_t>a[3] << 32) |
            (<uint64_t>a[4] << 24) |
            (<uint64_t>a[5] << 16) |
            (<uint64_t>a[6] << 8) |
            <uint64_t>a[7]
        )
        return o

    cpdef bytes to_bytes(self):
        """Returns the native data as a byte sequence.

        Returns:
            A byte sequence representing the native data with big endian.
            The length is 8.

        """
        cdef uint8_t a[8]
        a[0] = <uint8_t>(self._data.v >> 56)
        a[1] = <uint8_t>(self._data.v >> 48)
        a[2] = <uint8_t>(self._data.v >> 40)
        a[3] = <uint8_t>(self._data.v >> 32)
        a[4] = <uint8_t>(self._data.v >> 24)
        a[5] = <uint8_t>(self._data.v >> 16)
        a[6] = <uint8_t>(self._data.v >> 8)
        a[7] = <uint8_t>self._data.v
        return <bytes>a[:8]

    @classmethod
    def from_float(cls, double src) -> Float64:
        """Creates a new instance from the specified floating point.

        Args:
            src: The floating point from which a new instance is created.

        Returns:
            A new instance created from the specified floating point.

        """
        cdef _ui64_double t
        t.f = src
        cdef Float64 o = Float64()
        o._data.v = t.ui
        return o

    cpdef double to_float(self):
        """Returns the native data as a floating point.

        Returns:
            A floating point that represents the native data.

        """
        cdef _ui64_double t
        t.ui = self._data.v
        return t.f

    @classmethod
    def from_ui32(cls, UInt32 src) -> Float64:
        """Creates a new instance from the specified 32-bit unsigned integer.

        The result is the same as that of :func:`ui32_to_f64()`.

        Args:
            src: The 32-bit unsigned integer from which a new instance is created.

        Returns:
            A new instance created from the specified 32-bit unsigned integer.

        """
        return ui32_to_f64(src)

    cpdef UInt32 to_ui32(
        self, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
    ):
        """Converts the IEEE 754 binary64 floating point to a 32-bit unsigned integer.

        The result is the same as that of :func:`f64_to_ui32()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 32-bit unsigned integer.

        """
        return f64_to_ui32(self, rounding_mode, exact)

    @classmethod
    def from_ui64(cls, UInt64 src) -> Float64:
        """Creates a new instance from the specified 64-bit unsigned integer.

        The result is the same as that of :func:`ui64_to_f64()`.

        Args:
            src: The 64-bit unsigned integer from which a new instance is created.

        Returns:
            A new instance created from the specified 64-bit unsigned integer.

        """
        return ui64_to_f64(src)

    cpdef UInt64 to_ui64(
        self, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
    ):
        """Converts the IEEE 754 binary64 floating point to a 64-bit unsigned integer.

        The result is the same as that of :func:`f64_to_ui64()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 64-bit unsigned integer.

        """
        return f64_to_ui64(self, rounding_mode, exact)

    @classmethod
    def from_i32(cls, Int32 src) -> Float64:
        """Creates a new instance from the specified 32-bit signed integer.

        The result is the same as that of :func:`i32_to_f64()`.

        Args:
            src: The 32-bit signed integer from which a new instance is created.

        Returns:
            A new instance created from the specified 32-bit signed integer.

        """
        return i32_to_f64(src)

    cpdef Int32 to_i32(
        self, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
    ):
        """Converts the IEEE 754 binary64 floating point to a 32-bit signed integer.

        The result is the same as that of :func:`f64_to_i32()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 32-bit signed integer.

        """
        return f64_to_i32(self, rounding_mode, exact)

    @classmethod
    def from_i64(cls, Int64 src) -> Float64:
        """Creates a new instance from the specified 64-bit signed integer.

        The result is the same as that of :func:`i64_to_f64()`.

        Args:
            src: The 64-bit signed integer from which a new instance is created.

        Returns:
            A new instance created from the specified 64-bit signed integer.

        """
        return i64_to_f64(src)

    cpdef Int64 to_i64(
        self, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
    ):
        """Converts the IEEE 754 binary64 floating point to a 64-bit signed integer.

        The result is the same as that of :func:`f64_to_i64()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 64-bit signed integer.

        """
        return f64_to_i64(self, rounding_mode, exact)

    @classmethod
    def from_f16(cls, Float16 src) -> Float64:
        """Creates a new instance from the specified IEEE 754 binary16 floating point.

        The result is the same as that of :func:`f16_to_f64()`.

        Args:
            src: The IEEE 754 binary16 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary16 floating point.

        """
        return f16_to_f64(src)

    cpdef Float16 to_f16(self):
        """Converts the IEEE 754 binary64 floating point to a binary16 floating point.

        The result is the same as that of :func:`f64_to_f16()`.

        Returns:
            The IEEE 754 binary16 floating point.

        """
        return f64_to_f16(self)

    @classmethod
    def from_f32(cls, Float32 src) -> Float64:
        """Creates a new instance from the specified IEEE 754 binary32 floating point.

        The result is the same as that of :func:`f32_to_f64()`.

        Args:
            src: The IEEE 754 binary32 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary32 floating point.

        """
        return f32_to_f64(src)

    cpdef Float32 to_f32(self):
        """Converts the IEEE 754 binary64 floating point to a binary32 floating point.

        The result is the same as that of :func:`f64_to_f32()`.

        Returns:
            The IEEE 754 binary32 floating point.

        """
        return f64_to_f32(self)

    @classmethod
    def from_f64(cls, Float64 src) -> Float64:
        """Creates a new instance from the specified IEEE 754 binary64 floating point.

        The result is a copy of the specified instance.

        Args:
            src: The IEEE 754 binary64 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary64 floating point.

        """
        return src.__pos__()

    cpdef Float64 to_f64(self):
        """Converts the IEEE 754 binary64 floating point to a binary64 floating point.

        The result is a copy.

        Returns:
            The IEEE 754 binary64 floating point.

        """
        return self.__pos__()

    @classmethod
    def from_f128(cls, Float128 src) -> Float64:
        """Creates a new instance from the specified IEEE 754 binary128 floating point.

        The result is the same as that of :func:`f128_to_f64()`.

        Args:
            src: The IEEE 754 binary128 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary128 floating point.

        """
        return f128_to_f64(src)

    cpdef Float128 to_f128(self):
        """Converts the IEEE 754 binary64 floating point to a binary128 floating point.

        The result is the same as that of :func:`f64_to_f128()`.

        Returns:
            The IEEE 754 binary128 floating point.

        """
        return f64_to_f128(self)

    cpdef Float64 round_to_int(
        self, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
    ):
        """Rounds the number.

        The result is the same as that of :func:`f64_round_to_int()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact rounding is unable.

        Returns:
            The resulted integer.

        """
        return f64_round_to_int(self, rounding_mode, exact)

    cpdef Float64 neg(self):
        """Negates the IEEE 754 binary64 floating point.

        The result is the same as that of :func:`f64_neg()`.

        Returns:
            The resulted number (``-x``).

        """
        return f64_neg(self)

    @classmethod
    def add(cls, Float64 x, Float64 y) -> Float64:
        """Adds the IEEE 754 binary64 floating points.

        The result is the same as that of :func:`f64_add()`.

        Args:
            x: The floating point to be added.
            y: The floating point to add.

        Returns:
            The resulted number (``x + y``).

        """
        return f64_add(x, y)

    @classmethod
    def sub(cls, Float64 x, Float64 y) -> Float64:
        """Subtracts the IEEE 754 binary64 floating points.

        The result is the same as that of :func:`f64_sub()`.

        Args:
            x: The floating point to be subtracted.
            y: The floating point to subtract.

        Returns:
            The resulted number (``x - y``).

        """
        return f64_sub(x, y)

    @classmethod
    def mul(cls, Float64 x, Float64 y) -> Float64:
        """Multiplies the IEEE 754 binary64 floating points.

        The result is the same as that of :func:`f64_mul()`.

        Args:
            x: The floating point to be multiplied.
            y: The floating point to multiply.

        Returns:
            The resulted number (``x * y``).

        """
        return f64_mul(x, y)

    @classmethod
    def mul_add(cls, Float64 x, Float64 y, Float64 z) -> Float64:
        """Multiplies and Adds the IEEE 754 binary64 floating points.

        The result is the same as that of :func:`f64_mul_add()`.

        Args:
            x: The floating point to be multiplied.
            y: The floating point to multiply.
            z: The floating point to add.

        Returns:
            The resulted number (``x * y + z``).

        """
        return f64_mul_add(x, y, z)

    @classmethod
    def div(cls, Float64 x, Float64 y) -> Float64:
        """Divides the IEEE 754 binary64 floating points.

        The result is the same as that of :func:`f64_div()`.

        Args:
            x: The floating point to be divided.
            y: The floating point to divide.

        Returns:
            The resulted number (``x / y``).

        """
        return f64_div(x, y)

    @classmethod
    def rem(cls, Float64 x, Float64 y) -> Float64:
        """Calculates a remainder by dividing the IEEE 754 binary64 floating points.

        The result is the same as that of :func:`f64_rem()`.

        Args:
            x: The floating point to be divided.
            y: The floating point to divide.

        Returns:
            The resulted number (``x % y``).

        """
        return f64_rem(x, y)

    cpdef Float64 sqrt(self):
        """Calculates a square root of the IEEE 754 binary64 floating point.

        The result is the same as that of :func:`f64_sqrt()`.

        Returns:
            The resulted number (``sqrt(x)``).

        """
        return f64_sqrt(self)

    @classmethod
    def eq(cls, Float64 x, Float64 y) -> bool:
        """Tests if the first one is equal to the second one expressed as IEEE 754 binary64 floating points.

        The result is the same as that of :func:`f64_eq()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

        """
        return f64_eq(x, y)

    @classmethod
    def le(cls, Float64 x, Float64 y) -> bool:
        """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary64 floating points.

        The result is the same as that of :func:`f64_le()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

        """
        return f64_le(x, y)

    @classmethod
    def lt(cls, Float64 x, Float64 y) -> bool:
        """Tests if the first one is less than the second one expressed as IEEE 754 binary64 floating points.

        The result is the same as that of :func:`f64_lt()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

        """
        return f64_lt(x, y)

    @classmethod
    def eq_signaling(cls, Float64 x, Float64 y) -> bool:
        """Tests if the first one is equal to the second one expressed as IEEE 754 binary64 floating points.

        The invalid exception flag is set for any NaN input, not just for signaling NaNs.

        The result is the same as that of :func:`f64_eq_signaling()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

        """
        return f64_eq_signaling(x, y)

    @classmethod
    def le_quiet(cls, Float64 x, Float64 y) -> bool:
        """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary64 floating points.

        The invalid exception flag is not set for quiet NaNs.

        The result is the same as that of :func:`f64_le_quiet()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

        """
        return f64_le_quiet(x, y)

    @classmethod
    def lt_quiet(cls, Float64 x, Float64 y) -> bool:
        """Tests if the first one is less than the second one expressed as IEEE 754 binary64 floating points.

        The invalid exception flag is not set for quiet NaNs.

        The result is the same as that of :func:`f64_lt_quiet()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

        """
        return f64_lt_quiet(x, y)

    cpdef bool is_signaling_nan(self):
        """Tests if the IEEE 754 binary64 floating point is a signaling NaN.

        The result is the same as that of :func:`f64_is_signaling_nan()`.

        Returns:
            ``True`` if the floating point is a signaling NaN, ``False`` otherwise.

        """
        return f64_is_signaling_nan(self)

    cpdef bool is_nan(self):
        """Tests if the IEEE 754 binary64 floating point is a NaN.

        The result is the same as that of :func:`f64_is_nan()`.

        Returns:
            ``True`` if the floating point is a NaN, ``False`` otherwise.

        """
        return f64_is_nan(self)

    cpdef bool is_inf(self):
        """Tests if the IEEE 754 binary64 floating point is an infinity.

        The result is the same as that of :func:`f64_is_inf()`.

        Returns:
            ``True`` if the floating point is an infinity, ``False`` otherwise.

        """
        return f64_is_inf(self)

    def __str__(self) -> str:
        return str(self.to_float())

    def __pos__(self) -> Self:
        return _make_float64(self._data)

    def __neg__(self) -> Self:
        return f64_neg(self)

    def __add__(self, other: Self) -> Self:
        return f64_add(self, other)

    def __sub__(self, other: Self) -> Self:
        return f64_sub(self, other)

    def __mul__(self, other: Self) -> Self:
        return f64_mul(self, other)

    def __truediv__(self, other: Self) -> Self:
        return f64_div(self, other)

    def __floordiv__(self, other: Self) -> Self:
        return f64_round_to_int(f64_div(self, other), RoundingMode.MIN)

    def __mod__(self, other: Self) -> Self:
        return f64_sub(self, f64_mul(other, f64_round_to_int(f64_div(self, other), RoundingMode.MIN)))

    def __lt__(self, other: Self) -> bool:
        return f64_lt(self, other)

    def __le__(self, other: Self) -> bool:
        return f64_le(self, other)

    def __gt__(self, other: Self) -> bool:
        return f64_lt(other, self)

    def __ge__(self, other: Self) -> bool:
        return f64_le(other, self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            raise ValueError('type mismatch')
        return f64_eq(self, other)

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            raise ValueError('type mismatch')
        return not f64_eq(self, other)

    def __iadd__(self, other: Self) -> Self:
        return self.__add__(other)

    def __isub__(self, other: Self) -> Self:
        return self.__sub__(other)

    def __imul__(self, other: Self) -> Self:
        return self.__mul__(other)

    def __itruediv__(self, other: Self) -> Self:
        return self.__truediv__(other)

    def __ifloordiv__(self, other: Self) -> Self:
        return self.__floordiv__(other)

    def __imod__(self, other: Self) -> Self:
        return self.__mod__(other)


cdef class Float128:
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

    cdef sf.float128_t _data
    """The native data."""

    cpdef sf.float128_t _get_data(self):  # To access the native data in another instance.
        return self._data

    @classmethod
    def size(cls) -> int:
        """Returns the native data size in bits.

        Returns:
            The native data size in bits, i.e. 128.

        """
        return 128

    @classmethod
    def from_bytes(cls, bytes src) -> Float128:
        """Creates a new instance from the specified byte sequence.

        Args:
            src: The byte sequence representing the native data with big endian.
                 The length must be 16.

        Returns:
            A new instance created from the specified byte sequence.

        Raises:
            ValueError: If the length of bytes is not 16.

        """
        if len(src) != 16:
            raise ValueError('length of bytes must be 16')
        cdef const uint8_t[:] a = src
        cdef ui128_f128 t
        t.ui.v0 = (
            (<uint64_t>a[0] << 56) |
            (<uint64_t>a[1] << 48) |
            (<uint64_t>a[2] << 40) |
            (<uint64_t>a[3] << 32) |
            (<uint64_t>a[4] << 24) |
            (<uint64_t>a[5] << 16) |
            (<uint64_t>a[6] << 8) |
            <uint64_t>a[7]
        )
        t.ui.v64 = (
            (<uint64_t>a[8] << 56) |
            (<uint64_t>a[9] << 48) |
            (<uint64_t>a[10] << 40) |
            (<uint64_t>a[11] << 32) |
            (<uint64_t>a[12] << 24) |
            (<uint64_t>a[13] << 16) |
            (<uint64_t>a[14] << 8) |
            <uint64_t>a[15]
        )
        cdef Float128 o = Float128()
        o._data = t.f
        return o

    cpdef bytes to_bytes(self):
        """Returns the native data as a byte sequence.

        Returns:
            A byte sequence representing the native data with big endian.
            The length is 16.

        """
        cdef uint8_t a[16]
        cdef ui128_f128 t
        t.f = self._data
        a[0] = <uint8_t>(t.ui.v0 >> 56)
        a[1] = <uint8_t>(t.ui.v0 >> 48)
        a[2] = <uint8_t>(t.ui.v0 >> 40)
        a[3] = <uint8_t>(t.ui.v0 >> 32)
        a[4] = <uint8_t>(t.ui.v0 >> 24)
        a[5] = <uint8_t>(t.ui.v0 >> 16)
        a[6] = <uint8_t>(t.ui.v0 >> 8)
        a[7] = <uint8_t>t.ui.v0
        a[8] = <uint8_t>(t.ui.v64 >> 56)
        a[9] = <uint8_t>(t.ui.v64 >> 48)
        a[10] = <uint8_t>(t.ui.v64 >> 40)
        a[11] = <uint8_t>(t.ui.v64 >> 32)
        a[12] = <uint8_t>(t.ui.v64 >> 24)
        a[13] = <uint8_t>(t.ui.v64 >> 16)
        a[14] = <uint8_t>(t.ui.v64 >> 8)
        a[15] = <uint8_t>t.ui.v64
        return <bytes>a[:16]

    @classmethod
    def from_float(cls, double src) -> Float128:
        """Creates a new instance from the specified floating point.

        Args:
            src: The floating point from which a new instance is created.

        Returns:
            A new instance created from the specified floating point.

        Note:
            Cannot create an instance with a number that an IEEE 754 binary64
            floating point is unable to express.

        """
        cdef _ui64_double t
        cdef sf.float64_t f
        t.f = src
        f.v = t.ui
        cdef Float128 o = Float128()
        o._data = sf.f64_to_f128(f)
        return o

    cpdef double to_float(self):
        """Returns the native data as a floating point.

        Returns:
            A floating point that represents the native data.

        Note:
            Cannot return the exact number if it is unable to be expressed
            as an IEEE 754 binary64 floating point.

        """
        cdef _ui64_double t
        t.ui = sf.f128_to_f64(self._data).v
        return t.f

    @classmethod
    def from_ui32(cls, UInt32 src) -> Float128:
        """Creates a new instance from the specified 32-bit unsigned integer.

        The result is the same as that of :func:`ui32_to_f128()`.

        Args:
            src: The 32-bit unsigned integer from which a new instance is created.

        Returns:
            A new instance created from the specified 32-bit unsigned integer.

        """
        return ui32_to_f128(src)

    cpdef UInt32 to_ui32(
        self, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
    ):
        """Converts the IEEE 754 binary128 floating point to a 32-bit unsigned integer.

        The result is the same as that of :func:`f128_to_ui32()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 32-bit unsigned integer.

        """
        return f128_to_ui32(self, rounding_mode, exact)

    @classmethod
    def from_ui64(cls, UInt64 src) -> Float128:
        """Creates a new instance from the specified 64-bit unsigned integer.

        The result is the same as that of :func:`ui64_to_f128()`.

        Args:
            src: The 64-bit unsigned integer from which a new instance is created.

        Returns:
            A new instance created from the specified 64-bit unsigned integer.

        """
        return ui64_to_f128(src)

    cpdef UInt64 to_ui64(
        self, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
    ):
        """Converts the IEEE 754 binary128 floating point to a 64-bit unsigned integer.

        The result is the same as that of :func:`f128_to_ui64()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 64-bit unsigned integer.

        """
        return f128_to_ui64(self, rounding_mode, exact)

    @classmethod
    def from_i32(cls, Int32 src) -> Float128:
        """Creates a new instance from the specified 32-bit signed integer.

        The result is the same as that of :func:`i32_to_f128()`.

        Args:
            src: The 32-bit signed integer from which a new instance is created.

        Returns:
            A new instance created from the specified 32-bit signed integer.

        """
        return i32_to_f128(src)

    cpdef Int32 to_i32(
        self, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
    ):
        """Converts the IEEE 754 binary128 floating point to a 32-bit signed integer.

        The result is the same as that of :func:`f128_to_i32()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 32-bit signed integer.

        """
        return f128_to_i32(self, rounding_mode, exact)

    @classmethod
    def from_i64(cls, Int64 src) -> Float128:
        """Creates a new instance from the specified 64-bit signed integer.

        The result is the same as that of :func:`i64_to_f128()`.

        Args:
            src: The 64-bit signed integer from which a new instance is created.

        Returns:
            A new instance created from the specified 64-bit signed integer.

        """
        return i64_to_f128(src)

    cpdef Int64 to_i64(
        self, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
    ):
        """Converts the IEEE 754 binary128 floating point to a 64-bit signed integer.

        The result is the same as that of :func:`f128_to_i64()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact conversion is unable.

        Returns:
            The 64-bit signed integer.

        """
        return f128_to_i64(self, rounding_mode, exact)

    @classmethod
    def from_f16(cls, Float16 src) -> Float128:
        """Creates a new instance from the specified IEEE 754 binary16 floating point.

        The result is the same as that of :func:`f16_to_f128()`.

        Args:
            src: The IEEE 754 binary16 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary16 floating point.

        """
        return f16_to_f128(src)

    cpdef Float16 to_f16(self):
        """Converts the IEEE 754 binary128 floating point to a binary16 floating point.

        The result is the same as that of :func:`f128_to_f16()`.

        Returns:
            The IEEE 754 binary16 floating point.

        """
        return f128_to_f16(self)

    @classmethod
    def from_f32(cls, Float32 src) -> Float128:
        """Creates a new instance from the specified IEEE 754 binary32 floating point.

        The result is the same as that of :func:`f32_to_f128()`.

        Args:
            src: The IEEE 754 binary32 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary32 floating point.

        """
        return f32_to_f128(src)

    cpdef Float32 to_f32(self):
        """Converts the IEEE 754 binary128 floating point to a binary32 floating point.

        The result is the same as that of :func:`f128_to_f32()`.

        Returns:
            The IEEE 754 binary32 floating point.

        """
        return f128_to_f32(self)

    @classmethod
    def from_f64(cls, Float64 src) -> Float128:
        """Creates a new instance from the specified IEEE 754 binary64 floating point.

        The result is the same as that of :func:`f64_to_f128()`.

        Args:
            src: The IEEE 754 binary64 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary64 floating point.

        """
        return f64_to_f128(src)

    cpdef Float64 to_f64(self):
        """Converts the IEEE 754 binary128 floating point to a binary64 floating point.

        The result is the same as that of :func:`f128_to_f64()`.

        Returns:
            The IEEE 754 binary64 floating point.

        """
        return f128_to_f64(self)

    @classmethod
    def from_f128(cls, Float128 src) -> Float128:
        """Creates a new instance from the specified IEEE 754 binary128 floating point.

        The result is a copy of the specified instance.

        Args:
            src: The IEEE 754 binary128 floating point from which a new instance is created.

        Returns:
            A new instance created from the specified IEEE 754 binary128 floating point.

        """
        return src.__pos__()

    cpdef Float128 to_f128(self):
        """Converts the IEEE 754 binary128 floating point to a binary128 floating point.

        The result is a copy.

        Returns:
            The IEEE 754 binary128 floating point.

        """
        return self.__pos__()

    cpdef Float128 round_to_int(
        self, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
    ):
        """Rounds the number.

        The result is the same as that of :func:`f128_round_to_int()`.

        Args:
            rounding_mode: The rounding mode.
            exact: If ``True`` is specified, the floating-point exception flags are to be set
                   when exact rounding is unable.

        Returns:
            The resulted integer.

        """
        return f128_round_to_int(self, rounding_mode, exact)

    cpdef Float128 neg(self):
        """Negates the IEEE 754 binary128 floating point.

        The result is the same as that of :func:`f128_neg()`.

        Returns:
            The resulted number (``-x``).

        """
        return f128_neg(self)

    @classmethod
    def add(cls, Float128 x, Float128 y) -> Float128:
        """Adds the IEEE 754 binary128 floating points.

        The result is the same as that of :func:`f128_add()`.

        Args:
            x: The floating point to be added.
            y: The floating point to add.

        Returns:
            The resulted number (``x + y``).

        """
        return f128_add(x, y)

    @classmethod
    def sub(cls, Float128 x, Float128 y) -> Float128:
        """Subtracts the IEEE 754 binary128 floating points.

        The result is the same as that of :func:`f128_sub()`.

        Args:
            x: The floating point to be subtracted.
            y: The floating point to subtract.

        Returns:
            The resulted number (``x - y``).

        """
        return f128_sub(x, y)

    @classmethod
    def mul(cls, Float128 x, Float128 y) -> Float128:
        """Multiplies the IEEE 754 binary128 floating points.

        The result is the same as that of :func:`f128_mul()`.

        Args:
            x: The floating point to be multiplied.
            y: The floating point to multiply.

        Returns:
            The resulted number (``x * y``).

        """
        return f128_mul(x, y)

    @classmethod
    def mul_add(cls, Float128 x, Float128 y, Float128 z) -> Float128:
        """Multiplies and Adds the IEEE 754 binary128 floating points.

        The result is the same as that of :func:`f128_mul_add()`.

        Args:
            x: The floating point to be multiplied.
            y: The floating point to multiply.
            z: The floating point to add.

        Returns:
            The resulted number (``x * y + z``).

        """
        return f128_mul_add(x, y, z)

    @classmethod
    def div(cls, Float128 x, Float128 y) -> Float128:
        """Divides the IEEE 754 binary128 floating points.

        The result is the same as that of :func:`f128_div()`.

        Args:
            x: The floating point to be divided.
            y: The floating point to divide.

        Returns:
            The resulted number (``x / y``).

        """
        return f128_div(x, y)

    @classmethod
    def rem(cls, Float128 x, Float128 y) -> Float128:
        """Calculates a remainder by dividing the IEEE 754 binary128 floating points.

        The result is the same as that of :func:`f128_rem()`.

        Args:
            x: The floating point to be divided.
            y: The floating point to divide.

        Returns:
            The resulted number (``x % y``).

        """
        return f128_rem(x, y)

    cpdef Float128 sqrt(self):
        """Calculates a square root of the IEEE 754 binary128 floating point.

        The result is the same as that of :func:`f128_sqrt()`.

        Returns:
            The resulted number (``sqrt(x)``).

        """
        return f128_sqrt(self)

    @classmethod
    def eq(cls, Float128 x, Float128 y) -> bool:
        """Tests if the first one is equal to the second one expressed as IEEE 754 binary128 floating points.

        The result is the same as that of :func:`f128_eq()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

        """
        return f128_eq(x, y)

    @classmethod
    def le(cls, Float128 x, Float128 y) -> bool:
        """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary128 floating points.

        The result is the same as that of :func:`f128_le()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

        """
        return f128_le(x, y)

    @classmethod
    def lt(cls, Float128 x, Float128 y) -> bool:
        """Tests if the first one is less than the second one expressed as IEEE 754 binary128 floating points.

        The result is the same as that of :func:`f128_lt()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

        """
        return f128_lt(x, y)

    @classmethod
    def eq_signaling(cls, Float128 x, Float128 y) -> bool:
        """Tests if the first one is equal to the second one expressed as IEEE 754 binary128 floating points.

        The invalid exception flag is set for any NaN input, not just for signaling NaNs.

        The result is the same as that of :func:`f128_eq_signaling()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

        """
        return f128_eq_signaling(x, y)

    @classmethod
    def le_quiet(cls, Float128 x, Float128 y) -> bool:
        """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary128 floating points.

        The invalid exception flag is not set for quiet NaNs.

        The result is the same as that of :func:`f128_le_quiet()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

        """
        return f128_le_quiet(x, y)

    @classmethod
    def lt_quiet(cls, Float128 x, Float128 y) -> bool:
        """Tests if the first one is less than the second one expressed as IEEE 754 binary128 floating points.

        The invalid exception flag is not set for quiet NaNs.

        The result is the same as that of :func:`f128_lt_quiet()`.

        Args:
            x: The first floating point to be compared.
            y: The second floating point to be compared.

        Returns:
            ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

        """
        return f128_lt_quiet(x, y)

    cpdef bool is_signaling_nan(self):
        """Tests if the IEEE 754 binary128 floating point is a signaling NaN.

        The result is the same as that of :func:`f128_is_signaling_nan()`.

        Returns:
            ``True`` if the floating point is a signaling NaN, ``False`` otherwise.

        """
        return f128_is_signaling_nan(self)

    cpdef bool is_nan(self):
        """Tests if the IEEE 754 binary128 floating point is a NaN.

        The result is the same as that of :func:`f128_is_nan()`.

        Returns:
            ``True`` if the floating point is a NaN, ``False`` otherwise.

        """
        return f128_is_nan(self)

    cpdef bool is_inf(self):
        """Tests if the IEEE 754 binary128 floating point is an infinity.

        The result is the same as that of :func:`f128_is_inf()`.

        Returns:
            ``True`` if the floating point is an infinity, ``False`` otherwise.

        """
        return f128_is_inf(self)

    def __str__(self) -> str:
        return str(self.to_float())

    def __pos__(self) -> Self:
        return _make_float128(self._data)

    def __neg__(self) -> Self:
        return f128_neg(self)

    def __add__(self, other: Self) -> Self:
        return f128_add(self, other)

    def __sub__(self, other: Self) -> Self:
        return f128_sub(self, other)

    def __mul__(self, other: Self) -> Self:
        return f128_mul(self, other)

    def __truediv__(self, other: Self) -> Self:
        return f128_div(self, other)

    def __floordiv__(self, other: Self) -> Self:
        return f128_round_to_int(f128_div(self, other), RoundingMode.MIN)

    def __mod__(self, other: Self) -> Self:
        return f128_sub(self, f128_mul(other, f128_round_to_int(f128_div(self, other), RoundingMode.MIN)))

    def __lt__(self, other: Self) -> bool:
        return f128_lt(self, other)

    def __le__(self, other: Self) -> bool:
        return f128_le(self, other)

    def __gt__(self, other: Self) -> bool:
        return f128_lt(other, self)

    def __ge__(self, other: Self) -> bool:
        return f128_le(other, self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            raise ValueError('type mismatch')
        return f128_eq(self, other)

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            raise ValueError('type mismatch')
        return not f128_eq(self, other)

    def __iadd__(self, other: Self) -> Self:
        return self.__add__(other)

    def __isub__(self, other: Self) -> Self:
        return self.__sub__(other)

    def __imul__(self, other: Self) -> Self:
        return self.__mul__(other)

    def __itruediv__(self, other: Self) -> Self:
        return self.__truediv__(other)

    def __ifloordiv__(self, other: Self) -> Self:
        return self.__floordiv__(other)

    def __imod__(self, other: Self) -> Self:
        return self.__mod__(other)


cpdef void set_tininess_mode(TininessMode mode):
    """Sets the tininess detection mode.

    Args:
        mode: The tininess detection mode to be set.

    """
    sf.softfloat_detectTininess = <uint_fast8_t>mode


cpdef TininessMode get_tininess_mode():
    """Returns the current tininess detection mode.

    Returns:
        The current tininess detection mode.

    """
    return <TininessMode>sf.softfloat_detectTininess


cpdef void set_rounding_mode(RoundingMode mode):
    """Sets the rounding mode.

    Args:
        mode: The rounding mode to be set.

    """
    sf.softfloat_roundingMode = <uint_fast8_t>mode


cpdef RoundingMode get_rounding_mode():
    """Returns the current rounding mode.

    Returns:
        The current rounding mode.

    """
    return <RoundingMode>sf.softfloat_roundingMode


cpdef void set_exception_flags(flags):
    """Sets the floating-point exception flags.

    Args:
        flags: The floating-point exception flags to be set.

    """
    sf.softfloat_exceptionFlags = <uint_fast8_t>flags


cpdef get_exception_flags():
    """Returns the current floating-point exception flags.

    Returns:
        The current floating-point exception flags.

    """
    return int(sf.softfloat_exceptionFlags)


cpdef bool test_exception_flags(flags):
    """Tests the floating-point exception flags.

    Args:
        flags: The floating-point exception flags to be tested.

    Returns:
        ``True`` if any of the specified exception flags is nonzero, ``False`` otherwise.

    """
    return (sf.softfloat_exceptionFlags & <uint_fast8_t>flags) != 0


cdef UInt32 _make_uint32(uint32_t src):
    cdef UInt32 i = UInt32()
    i._data = src
    return i


cdef UInt64 _make_uint64(uint64_t src):
    cdef UInt64 i = UInt64()
    i._data = src
    return i


cdef Int32 _make_int32(int32_t src):
    cdef Int32 i = Int32()
    i._data = src
    return i


cdef Int64 _make_int64(int64_t src):
    cdef Int64 i = Int64()
    i._data = src
    return i


cdef BFloat16 _make_bfloat16(sf.bfloat16_t src):
    cdef BFloat16 f = BFloat16()
    f._data = src
    return f


cdef Float16 _make_float16(sf.float16_t src):
    cdef Float16 f = Float16()
    f._data = src
    return f


cdef Float32 _make_float32(sf.float32_t src):
    cdef Float32 f = Float32()
    f._data = src
    return f


cdef Float64 _make_float64(sf.float64_t src):
    cdef Float64 f = Float64()
    f._data = src
    return f


cdef Float128 _make_float128(sf.float128_t src):
    cdef Float128 f = Float128()
    f._data = src
    return f


cpdef Float16 ui32_to_f16(UInt32 x):
    """Converts the 32-bit unsigned integer to an IEEE 754 binary16 floating point.

    Args:
        x: The 32-bit unsigned integer to be converted.

    Returns:
        The IEEE 754 binary16 floating point.

    """
    return _make_float16(sf.ui32_to_f16(x._data))


cpdef Float32 ui32_to_f32(UInt32 x):
    """Converts the 32-bit unsigned integer to an IEEE 754 binary32 floating point.

    Args:
        x: The 32-bit unsigned integer to be converted.

    Returns:
        The IEEE 754 binary32 floating point.

    """
    return _make_float32(sf.ui32_to_f32(x._data))


cpdef Float64 ui32_to_f64(UInt32 x):
    """Converts the 32-bit unsigned integer to an IEEE 754 binary64 floating point.

    Args:
        x: The 32-bit unsigned integer to be converted.

    Returns:
        The IEEE 754 binary64 floating point.

    """
    return _make_float64(sf.ui32_to_f64(x._data))


cpdef Float128 ui32_to_f128(UInt32 x):
    """Converts the 32-bit unsigned integer to an IEEE 754 binary128 floating point.

    Args:
        x: The 32-bit unsigned integer to be converted.

    Returns:
        The IEEE 754 binary128 floating point.

    """
    return _make_float128(sf.ui32_to_f128(x._data))


cpdef Float16 ui64_to_f16(UInt64 x):
    """Converts the 64-bit unsigned integer to an IEEE 754 binary16 floating point.

    Args:
        x: The 64-bit unsigned integer to be converted.

    Returns:
        The IEEE 754 binary16 floating point.

    """
    return _make_float16(sf.ui64_to_f16(x._data))


cpdef Float32 ui64_to_f32(UInt64 x):
    """Converts the 64-bit unsigned integer to an IEEE 754 binary32 floating point.

    Args:
        x: The 64-bit unsigned integer to be converted.

    Returns:
        The IEEE 754 binary32 floating point.

    """
    return _make_float32(sf.ui64_to_f32(x._data))


cpdef Float64 ui64_to_f64(UInt64 x):
    """Converts the 64-bit unsigned integer to an IEEE 754 binary64 floating point.

    Args:
        x: The 64-bit unsigned integer to be converted.

    Returns:
        The IEEE 754 binary64 floating point.

    """
    return _make_float64(sf.ui64_to_f64(x._data))


cpdef Float128 ui64_to_f128(UInt64 x):
    """Converts the 64-bit unsigned integer to an IEEE 754 binary128 floating point.

    Args:
        x: The 64-bit unsigned integer to be converted.

    Returns:
        The IEEE 754 binary128 floating point.

    """
    return _make_float128(sf.ui64_to_f128(x._data))


cpdef Float16 i32_to_f16(Int32 x):
    """Converts the 32-bit signed integer to an IEEE 754 binary16 floating point.

    Args:
        x: The 32-bit signed integer to be converted.

    Returns:
        The IEEE 754 binary16 floating point.

    """
    return _make_float16(sf.i32_to_f16(x._data))


cpdef Float32 i32_to_f32(Int32 x):
    """Converts the 32-bit signed integer to an IEEE 754 binary32 floating point.

    Args:
        x: The 32-bit signed integer to be converted.

    Returns:
        The IEEE 754 binary32 floating point.

    """
    return _make_float32(sf.i32_to_f32(x._data))


cpdef Float64 i32_to_f64(Int32 x):
    """Converts the 32-bit signed integer to an IEEE 754 binary64 floating point.

    Args:
        x: The 32-bit signed integer to be converted.

    Returns:
        The IEEE 754 binary64 floating point.

    """
    return _make_float64(sf.i32_to_f64(x._data))


cpdef Float128 i32_to_f128(Int32 x):
    """Converts the 32-bit signed integer to an IEEE 754 binary128 floating point.

    Args:
        x: The 32-bit signed integer to be converted.

    Returns:
        The IEEE 754 binary128 floating point.

    """
    return _make_float128(sf.i32_to_f128(x._data))


cpdef Float16 i64_to_f16(Int64 x):
    """Converts the 64-bit signed integer to an IEEE 754 binary16 floating point.

    Args:
        x: The 64-bit signed integer to be converted.

    Returns:
        The IEEE 754 binary16 floating point.

    """
    return _make_float16(sf.i64_to_f16(x._data))


cpdef Float32 i64_to_f32(Int64 x):
    """Converts the 64-bit signed integer to an IEEE 754 binary32 floating point.

    Args:
        x: The 64-bit signed integer to be converted.

    Returns:
        The IEEE 754 binary32 floating point.

    """
    return _make_float32(sf.i64_to_f32(x._data))


cpdef Float64 i64_to_f64(Int64 x):
    """Converts the 64-bit signed integer to an IEEE 754 binary64 floating point.

    Args:
        x: The 64-bit signed integer to be converted.

    Returns:
        The IEEE 754 binary64 floating point.

    """
    return _make_float64(sf.i64_to_f64(x._data))


cpdef Float128 i64_to_f128(Int64 x):
    """Converts the 64-bit signed integer to an IEEE 754 binary128 floating point.

    Args:
        x: The 64-bit signed integer to be converted.

    Returns:
        The IEEE 754 binary128 floating point.

    """
    return _make_float128(sf.i64_to_f128(x._data))


cpdef UInt32 f16_to_ui32(
    Float16 x, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
):
    """Converts the IEEE 754 binary16 floating point to a 32-bit unsigned integer.

    Args:
        x: The IEEE 754 binary16 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 32-bit unsigned integer.

    """
    return _make_uint32(sf.f16_to_ui32(x._data, rounding_mode, exact))


cpdef UInt64 f16_to_ui64(
    Float16 x, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
):
    """Converts the IEEE 754 binary16 floating point to a 64-bit unsigned integer.

    Args:
        x: The IEEE 754 binary16 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 64-bit unsigned integer.

    """
    return _make_uint64(sf.f16_to_ui64(x._data, rounding_mode, exact))


cpdef Int32 f16_to_i32(
    Float16 x, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
):
    """Converts the IEEE 754 binary16 floating point to a 32-bit signed integer.

    Args:
        x: The IEEE 754 binary16 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 32-bit signed integer.

    """
    return _make_int32(sf.f16_to_i32(x._data, rounding_mode, exact))


cpdef Int64 f16_to_i64(
    Float16 x, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
):
    """Converts the IEEE 754 binary16 floating point to a 64-bit signed integer.

    Args:
        x: The IEEE 754 binary16 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 64-bit signed integer.

    """
    return _make_int64(sf.f16_to_i64(x._data, rounding_mode, exact))


cpdef Float32 f16_to_f32(Float16 x):
    """Converts the IEEE 754 binary16 floating point to a binary32 floating point.

    Args:
        x: The IEEE 754 binary16 floating point to be converted.

    Returns:
        The IEEE 754 binary32 floating point.

    """
    return _make_float32(sf.f16_to_f32(x._data))


cpdef Float64 f16_to_f64(Float16 x):
    """Converts the IEEE 754 binary16 floating point to a binary64 floating point.

    Args:
        x: The IEEE 754 binary16 floating point to be converted.

    Returns:
        The IEEE 754 binary64 floating point.

    """
    return _make_float64(sf.f16_to_f64(x._data))


cpdef Float128 f16_to_f128(Float16 x):
    """Converts the IEEE 754 binary16 floating point to a binary128 floating point.

    Args:
        x: The IEEE 754 binary16 floating point to be converted.

    Returns:
        The IEEE 754 binary128 floating point.

    """
    return _make_float128(sf.f16_to_f128(x._data))


cpdef Float16 f16_round_to_int(
    Float16 x, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
):
    """Rounds the number expressed as an IEEE 754 binary16 floating point.

    Args:
        x: The IEEE 754 binary16 floating point to be rounded.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact rounding is unable.

    Returns:
        The resulted integer expressed as an IEEE 754 binary16 floating point.

    """
    return _make_float16(sf.f16_roundToInt(x._data, rounding_mode, exact))


cpdef Float16 f16_neg(Float16 x):
    """Negates the IEEE 754 binary16 floating point.

    Args:
        x: The floating point to be negated.

    Returns:
        The resulted number expressed as an IEEE 754 binary16 floating point (``-x``).

    """
    cdef Float16 f = Float16()
    f._data.v = x._data.v ^ <uint16_t>0x8000
    return f


cpdef Float16 f16_add(Float16 x, Float16 y):
    """Adds the IEEE 754 binary16 floating points.

    Args:
        x: The floating point to be added.
        y: The floating point to add.

    Returns:
        The resulted number expressed as an IEEE 754 binary16 floating point (``x + y``).

    """
    return _make_float16(sf.f16_add(x._data, y._data))


cpdef Float16 f16_sub(Float16 x, Float16 y):
    """Subtracts the IEEE 754 binary16 floating points.

    Args:
        x: The floating point to be subtracted.
        y: The floating point to subtract.

    Returns:
        The resulted number expressed as an IEEE 754 binary16 floating point (``x - y``).

    """
    return _make_float16(sf.f16_sub(x._data, y._data))


cpdef Float16 f16_mul(Float16 x, Float16 y):
    """Multiplies the IEEE 754 binary16 floating points.

    Args:
        x: The floating point to be multiplied.
        y: The floating point to multiply.

    Returns:
        The resulted number expressed as an IEEE 754 binary16 floating point (``x * y``).

    """
    return _make_float16(sf.f16_mul(x._data, y._data))


cpdef Float16 f16_mul_add(Float16 x, Float16 y, Float16 z):
    """Multiplies and Adds the IEEE 754 binary16 floating points.

    Args:
        x: The floating point to be multiplied.
        y: The floating point to multiply.
        z: The floating point to add.

    Returns:
        The resulted number expressed as an IEEE 754 binary16 floating point (``x * y + z``).

    """
    return _make_float16(sf.f16_mulAdd(x._data, y._data, z._data))


cpdef Float16 f16_div(Float16 x, Float16 y):
    """Divides the IEEE 754 binary16 floating points.

    Args:
        x: The floating point to be divided.
        y: The floating point to divide.

    Returns:
        The resulted number expressed as an IEEE 754 binary16 floating point (``x / y``).

    """
    return _make_float16(sf.f16_div(x._data, y._data))


cpdef Float16 f16_rem(Float16 x, Float16 y):
    """Calculates a remainder by dividing the IEEE 754 binary16 floating points.

    Args:
        x: The floating point to be divided.
        y: The floating point to divide.

    Returns:
        The resulted number expressed as an IEEE 754 binary16 floating point (``x % y``).

    """
    return _make_float16(sf.f16_rem(x._data, y._data))


cpdef Float16 f16_sqrt(Float16 x):
    """Calculates a square root of the IEEE 754 binary16 floating point.

    Args:
        x: The floating point whose square root is to be calculated.

    Returns:
        The resulted number expressed as an IEEE 754 binary16 floating point (``sqrt(x)``).

    """
    return _make_float16(sf.f16_sqrt(x._data))


cpdef bool f16_eq(Float16 x, Float16 y):
    """Tests if the first one is equal to the second one expressed as IEEE 754 binary16 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

    """
    return sf.f16_eq(x._data, y._data)


cpdef bool f16_le(Float16 x, Float16 y):
    """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary16 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

    """
    return sf.f16_le(x._data, y._data)


cpdef bool f16_lt(Float16 x, Float16 y):
    """Tests if the first one is less than the second one expressed as IEEE 754 binary16 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

    """
    return sf.f16_lt(x._data, y._data)


cpdef bool f16_eq_signaling(Float16 x, Float16 y):
    """Tests if the first one is equal to the second one expressed as IEEE 754 binary16 floating points.

    The invalid exception flag is set for any NaN input, not just for signaling NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

    """
    return sf.f16_eq_signaling(x._data, y._data)


cpdef bool f16_le_quiet(Float16 x, Float16 y):
    """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary16 floating points.

    The invalid exception flag is not set for quiet NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

    """
    return sf.f16_le_quiet(x._data, y._data)


cpdef bool f16_lt_quiet(Float16 x, Float16 y):
    """Tests if the first one is less than the second one expressed as IEEE 754 binary16 floating points.

    The invalid exception flag is not set for quiet NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

    """
    return sf.f16_lt_quiet(x._data, y._data)


cpdef bool f16_is_signaling_nan(Float16 x):
    """Tests if the IEEE 754 binary16 floating point is a signaling NaN.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is a signaling NaN, ``False`` otherwise.

    """
    return sf.f16_isSignalingNaN(x._data)


cpdef bool f16_is_nan(Float16 x):
    """Tests if the IEEE 754 binary16 floating point is a NaN.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is a NaN, ``False`` otherwise.

    """
    return (~x._data.v & 0x7c00) == 0 and (x._data.v & 0x03ff) != 0


cpdef bool f16_is_inf(Float16 x):
    """Tests if the IEEE 754 binary16 floating point is an infinity.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is an infinity, ``False`` otherwise.

    """
    return (x._data.v & 0x7fff) == 0x7c00


cpdef Float32 bf16_to_f32(BFloat16 x):
    """Converts the 16-bit brain floating point to an IEEE 754 binary32 floating point.

    Args:
        x: The 16-bit brain floating point to be converted.

    Returns:
        The IEEE 754 binary32 floating point.

    """
    return _make_float32(sf.bf16_to_f32(x._data))


cpdef BFloat16 f32_to_bf16(Float32 x):
    """Converts the IEEE 754 binary32 floating point to a 16-bit brain floating point.

    Args:
        x: The IEEE 754 binary32 floating point to be converted.

    Returns:
        The 16-bit brain floating point.

    """
    return _make_bfloat16(sf.f32_to_bf16(x._data))


cpdef bool bf16_is_signaling_nan(BFloat16 x):
    """Tests if the 16-bit brain floating point is a signaling NaN.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is a signaling NaN, ``False`` otherwise.

    """
    return sf.bf16_isSignalingNaN(x._data)


cpdef bool bf16_is_nan(BFloat16 x):
    """Tests if the 16-bit brain floating point is a NaN.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is a NaN, ``False`` otherwise.

    """
    return (~x._data.v & 0x7f80) == 0 and (x._data.v & 0x007f) != 0


cpdef bool bf16_is_inf(BFloat16 x):
    """Tests if the 16-bit brain floating point is an infinity.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is an infinity, ``False`` otherwise.

    """
    return (x._data.v & 0x7fff) == 0x7f80


cpdef UInt32 f32_to_ui32(
    Float32 x, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
):
    """Converts the IEEE 754 binary32 floating point to a 32-bit unsigned integer.

    Args:
        x: The IEEE 754 binary32 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 32-bit unsigned integer.

    """
    return _make_uint32(sf.f32_to_ui32(x._data, rounding_mode, exact))


cpdef UInt64 f32_to_ui64(
    Float32 x, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
):
    """Converts the IEEE 754 binary32 floating point to a 64-bit unsigned integer.

    Args:
        x: The IEEE 754 binary32 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 64-bit unsigned integer.

    """
    return _make_uint64(sf.f32_to_ui64(x._data, rounding_mode, exact))


cpdef Int32 f32_to_i32(
    Float32 x, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
):
    """Converts the IEEE 754 binary32 floating point to a 32-bit signed integer.

    Args:
        x: The IEEE 754 binary32 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 32-bit signed integer.

    """
    return _make_int32(sf.f32_to_i32(x._data, rounding_mode, exact))


cpdef Int64 f32_to_i64(
    Float32 x, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
):
    """Converts the IEEE 754 binary32 floating point to a 64-bit signed integer.

    Args:
        x: The IEEE 754 binary32 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 64-bit signed integer.

    """
    return _make_int64(sf.f32_to_i64(x._data, rounding_mode, exact))


cpdef Float16 f32_to_f16(Float32 x):
    """Converts the IEEE 754 binary32 floating point to a binary16 floating point.

    Args:
        x: The IEEE 754 binary32 floating point to be converted.

    Returns:
        The IEEE 754 binary16 floating point.

    """
    return _make_float16(sf.f32_to_f16(x._data))


cpdef Float64 f32_to_f64(Float32 x):
    """Converts the IEEE 754 binary32 floating point to a binary64 floating point.

    Args:
        x: The IEEE 754 binary32 floating point to be converted.

    Returns:
        The IEEE 754 binary64 floating point.

    """
    return _make_float64(sf.f32_to_f64(x._data))


cpdef Float128 f32_to_f128(Float32 x):
    """Converts the IEEE 754 binary32 floating point to a binary128 floating point.

    Args:
        x: The IEEE 754 binary32 floating point to be converted.

    Returns:
        The IEEE 754 binary128 floating point.

    """
    return _make_float128(sf.f32_to_f128(x._data))


cpdef Float32 f32_round_to_int(
    Float32 x, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
):
    """Rounds the number expressed as an IEEE 754 binary32 floating point.

    Args:
        x: The IEEE 754 binary32 floating point to be rounded.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact rounding is unable.

    Returns:
        The resulted integer expressed as an IEEE 754 binary32 floating point.

    """
    return _make_float32(sf.f32_roundToInt(x._data, rounding_mode, exact))


cpdef Float32 f32_neg(Float32 x):
    """Negates the IEEE 754 binary32 floating point.

    Args:
        x: The floating point to be negated.

    Returns:
        The resulted number expressed as an IEEE 754 binary32 floating point (``-x``).

    """
    cdef Float32 f = Float32()
    f._data.v = x._data.v ^ <uint32_t>0x80000000
    return f


cpdef Float32 f32_add(Float32 x, Float32 y):
    """Adds the IEEE 754 binary32 floating points.

    Args:
        x: The floating point to be added.
        y: The floating point to add.

    Returns:
        The resulted number expressed as an IEEE 754 binary32 floating point (``x + y``).

    """
    return _make_float32(sf.f32_add(x._data, y._data))


cpdef Float32 f32_sub(Float32 x, Float32 y):
    """Subtracts the IEEE 754 binary32 floating points.

    Args:
        x: The floating point to be subtracted.
        y: The floating point to subtract.

    Returns:
        The resulted number expressed as an IEEE 754 binary32 floating point (``x - y``).

    """
    return _make_float32(sf.f32_sub(x._data, y._data))


cpdef Float32 f32_mul(Float32 x, Float32 y):
    """Multiplies the IEEE 754 binary32 floating points.

    Args:
        x: The floating point to be multiplied.
        y: The floating point to multiply.

    Returns:
        The resulted number expressed as an IEEE 754 binary32 floating point (``x * y``).

    """
    return _make_float32(sf.f32_mul(x._data, y._data))


cpdef Float32 f32_mul_add(Float32 x, Float32 y, Float32 z):
    """Multiplies and Adds the IEEE 754 binary32 floating points.

    Args:
        x: The floating point to be multiplied.
        y: The floating point to multiply.
        z: The floating point to add.

    Returns:
        The resulted number expressed as an IEEE 754 binary32 floating point (``x * y + z``).

    """
    return _make_float32(sf.f32_mulAdd(x._data, y._data, z._data))


cpdef Float32 f32_div(Float32 x, Float32 y):
    """Divides the IEEE 754 binary32 floating points.

    Args:
        x: The floating point to be divided.
        y: The floating point to divide.

    Returns:
        The resulted number expressed as an IEEE 754 binary32 floating point (``x / y``).

    """
    return _make_float32(sf.f32_div(x._data, y._data))


cpdef Float32 f32_rem(Float32 x, Float32 y):
    """Calculates a remainder by dividing the IEEE 754 binary32 floating points.

    Args:
        x: The floating point to be divided.
        y: The floating point to divide.

    Returns:
        The resulted number expressed as an IEEE 754 binary32 floating point (``x % y``).

    """
    return _make_float32(sf.f32_rem(x._data, y._data))


cpdef Float32 f32_sqrt(Float32 x):
    """Calculates a square root of the IEEE 754 binary32 floating point.

    Args:
        x: The floating point whose square root is to be calculated.

    Returns:
        The resulted number expressed as an IEEE 754 binary32 floating point (``sqrt(x)``).

    """
    return _make_float32(sf.f32_sqrt(x._data))


cpdef bool f32_eq(Float32 x, Float32 y):
    """Tests if the first one is equal to the second one expressed as IEEE 754 binary32 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

    """
    return sf.f32_eq(x._data, y._data)


cpdef bool f32_le(Float32 x, Float32 y):
    """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary32 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

    """
    return sf.f32_le(x._data, y._data)


cpdef bool f32_lt(Float32 x, Float32 y):
    """Tests if the first one is less than the second one expressed as IEEE 754 binary32 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

    """
    return sf.f32_lt(x._data, y._data)


cpdef bool f32_eq_signaling(Float32 x, Float32 y):
    """Tests if the first one is equal to the second one expressed as IEEE 754 binary32 floating points.

    The invalid exception flag is set for any NaN input, not just for signaling NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

    """
    return sf.f32_eq_signaling(x._data, y._data)


cpdef bool f32_le_quiet(Float32 x, Float32 y):
    """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary32 floating points.

    The invalid exception flag is not set for quiet NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

    """
    return sf.f32_le_quiet(x._data, y._data)


cpdef bool f32_lt_quiet(Float32 x, Float32 y):
    """Tests if the first one is less than the second one expressed as IEEE 754 binary32 floating points.

    The invalid exception flag is not set for quiet NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

    """
    return sf.f32_lt_quiet(x._data, y._data)


cpdef bool f32_is_signaling_nan(Float32 x):
    """Tests if the IEEE 754 binary32 floating point is a signaling NaN.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is a signaling NaN, ``False`` otherwise.

    """
    return sf.f32_isSignalingNaN(x._data)


cpdef bool f32_is_nan(Float32 x):
    """Tests if the IEEE 754 binary32 floating point is a NaN.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is a NaN, ``False`` otherwise.

    """
    return (~x._data.v & 0x7f800000) == 0 and (x._data.v & 0x007fffff) != 0


cpdef bool f32_is_inf(Float32 x):
    """Tests if the IEEE 754 binary32 floating point is an infinity.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is an infinity, ``False`` otherwise.

    """
    return (x._data.v & 0x7fffffff) == 0x7f800000


cpdef UInt32 f64_to_ui32(
    Float64 x, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
):
    """Converts the IEEE 754 binary64 floating point to a 32-bit unsigned integer.

    Args:
        x: The IEEE 754 binary64 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 32-bit unsigned integer.

    """
    return _make_uint32(sf.f64_to_ui32(x._data, rounding_mode, exact))


cpdef UInt64 f64_to_ui64(
    Float64 x, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
):
    """Converts the IEEE 754 binary64 floating point to a 64-bit unsigned integer.

    Args:
        x: The IEEE 754 binary64 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 64-bit unsigned integer.

    """
    return _make_uint64(sf.f64_to_ui64(x._data, rounding_mode, exact))


cpdef Int32 f64_to_i32(
    Float64 x, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
):
    """Converts the IEEE 754 binary64 floating point to a 32-bit signed integer.

    Args:
        x: The IEEE 754 binary64 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 32-bit signed integer.

    """
    return _make_int32(sf.f64_to_i32(x._data, rounding_mode, exact))


cpdef Int64 f64_to_i64(
    Float64 x, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
):
    """Converts the IEEE 754 binary64 floating point to a 64-bit signed integer.

    Args:
        x: The IEEE 754 binary64 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 64-bit signed integer.

    """
    return _make_int64(sf.f64_to_i64(x._data, rounding_mode, exact))


cpdef Float16 f64_to_f16(Float64 x):
    """Converts the IEEE 754 binary64 floating point to a binary16 floating point.

    Args:
        x: The IEEE 754 binary64 floating point to be converted.

    Returns:
        The IEEE 754 binary16 floating point.

    """
    return _make_float16(sf.f64_to_f16(x._data))


cpdef Float32 f64_to_f32(Float64 x):
    """Converts the IEEE 754 binary64 floating point to a binary32 floating point.

    Args:
        x: The IEEE 754 binary64 floating point to be converted.

    Returns:
        The IEEE 754 binary32 floating point.

    """
    return _make_float32(sf.f64_to_f32(x._data))


cpdef Float128 f64_to_f128(Float64 x):
    """Converts the IEEE 754 binary64 floating point to a binary128 floating point.

    Args:
        x: The IEEE 754 binary64 floating point to be converted.

    Returns:
        The IEEE 754 binary128 floating point.

    """
    return _make_float128(sf.f64_to_f128(x._data))


cpdef Float64 f64_round_to_int(
    Float64 x, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
):
    """Rounds the number expressed as an IEEE 754 binary64 floating point.

    Args:
        x: The IEEE 754 binary64 floating point to be rounded.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact rounding is unable.

    Returns:
        The resulted integer expressed as an IEEE 754 binary64 floating point.

    """
    return _make_float64(sf.f64_roundToInt(x._data, rounding_mode, exact))


cpdef Float64 f64_neg(Float64 x):
    """Negates the IEEE 754 binary64 floating point.

    Args:
        x: The floating point to be negated.

    Returns:
        The resulted number expressed as an IEEE 754 binary64 floating point (``-x``).

    """
    cdef Float64 f = Float64()
    f._data.v = x._data.v ^ <uint64_t>0x80000000_00000000
    return f


cpdef Float64 f64_add(Float64 x, Float64 y):
    """Adds the IEEE 754 binary64 floating points.

    Args:
        x: The floating point to be added.
        y: The floating point to add.

    Returns:
        The resulted number expressed as an IEEE 754 binary64 floating point (``x + y``).

    """
    return _make_float64(sf.f64_add(x._data, y._data))


cpdef Float64 f64_sub(Float64 x, Float64 y):
    """Subtracts the IEEE 754 binary64 floating points.

    Args:
        x: The floating point to be subtracted.
        y: The floating point to subtract.

    Returns:
        The resulted number expressed as an IEEE 754 binary64 floating point (``x - y``).

    """
    return _make_float64(sf.f64_sub(x._data, y._data))


cpdef Float64 f64_mul(Float64 x, Float64 y):
    """Multiplies the IEEE 754 binary64 floating points.

    Args:
        x: The floating point to be multiplied.
        y: The floating point to multiply.

    Returns:
        The resulted number expressed as an IEEE 754 binary64 floating point (``x * y``).

    """
    return _make_float64(sf.f64_mul(x._data, y._data))


cpdef Float64 f64_mul_add(Float64 x, Float64 y, Float64 z):
    """Multiplies and Adds the IEEE 754 binary64 floating points.

    Args:
        x: The floating point to be multiplied.
        y: The floating point to multiply.
        z: The floating point to add.

    Returns:
        The resulted number expressed as an IEEE 754 binary64 floating point (``x * y + z``).

    """
    return _make_float64(sf.f64_mulAdd(x._data, y._data, z._data))


cpdef Float64 f64_div(Float64 x, Float64 y):
    """Divides the IEEE 754 binary64 floating points.

    Args:
        x: The floating point to be divided.
        y: The floating point to divide.

    Returns:
        The resulted number expressed as an IEEE 754 binary64 floating point (``x / y``).

    """
    return _make_float64(sf.f64_div(x._data, y._data))


cpdef Float64 f64_rem(Float64 x, Float64 y):
    """Calculates a remainder by dividing the IEEE 754 binary64 floating points.

    Args:
        x: The floating point to be divided.
        y: The floating point to divide.

    Returns:
        The resulted number expressed as an IEEE 754 binary64 floating point (``x % y``).

    """
    return _make_float64(sf.f64_rem(x._data, y._data))


cpdef Float64 f64_sqrt(Float64 x):
    """Calculates a square root of the IEEE 754 binary64 floating point.

    Args:
        x: The floating point whose square root is to be calculated.

    Returns:
        The resulted number expressed as an IEEE 754 binary64 floating point (``sqrt(x)``).

    """
    return _make_float64(sf.f64_sqrt(x._data))


cpdef bool f64_eq(Float64 x, Float64 y):
    """Tests if the first one is equal to the second one expressed as IEEE 754 binary64 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

    """
    return sf.f64_eq(x._data, y._data)


cpdef bool f64_le(Float64 x, Float64 y):
    """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary64 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

    """
    return sf.f64_le(x._data, y._data)


cpdef bool f64_lt(Float64 x, Float64 y):
    """Tests if the first one is less than the second one expressed as IEEE 754 binary64 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

    """
    return sf.f64_lt(x._data, y._data)


cpdef bool f64_eq_signaling(Float64 x, Float64 y):
    """Tests if the first one is equal to the second one expressed as IEEE 754 binary64 floating points.

    The invalid exception flag is set for any NaN input, not just for signaling NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

    """
    return sf.f64_eq_signaling(x._data, y._data)


cpdef bool f64_le_quiet(Float64 x, Float64 y):
    """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary64 floating points.

    The invalid exception flag is not set for quiet NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

    """
    return sf.f64_le_quiet(x._data, y._data)


cpdef bool f64_lt_quiet(Float64 x, Float64 y):
    """Tests if the first one is less than the second one expressed as IEEE 754 binary64 floating points.

    The invalid exception flag is not set for quiet NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

    """
    return sf.f64_lt_quiet(x._data, y._data)


cpdef bool f64_is_signaling_nan(Float64 x):
    """Tests if the IEEE 754 binary64 floating point is a signaling NaN.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is a signaling NaN, ``False`` otherwise.

    """
    return sf.f64_isSignalingNaN(x._data)


cpdef bool f64_is_nan(Float64 x):
    """Tests if the IEEE 754 binary64 floating point is a NaN.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is a NaN, ``False`` otherwise.

    """
    return (~x._data.v & 0x7ff00000_00000000) == 0 and (x._data.v & 0x000fffff_ffffffff) != 0


cpdef bool f64_is_inf(Float64 x):
    """Tests if the IEEE 754 binary64 floating point is an infinity.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is an infinity, ``False`` otherwise.

    """
    return (x._data.v & 0x7fffffff_ffffffff) == 0x7ff00000_00000000


cpdef UInt32 f128_to_ui32(
    Float128 x, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
):
    """Converts the IEEE 754 binary128 floating point to a 32-bit unsigned integer.

    Args:
        x: The IEEE 754 binary128 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 32-bit unsigned integer.

    """
    return _make_uint32(sf.f128_to_ui32(x._data, rounding_mode, exact))


cpdef UInt64 f128_to_ui64(
    Float128 x, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
):
    """Converts the IEEE 754 binary128 floating point to a 64-bit unsigned integer.

    Args:
        x: The IEEE 754 binary128 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 64-bit unsigned integer.

    """
    return _make_uint64(sf.f128_to_ui64(x._data, rounding_mode, exact))


cpdef Int32 f128_to_i32(
    Float128 x, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
):
    """Converts the IEEE 754 binary128 floating point to a 32-bit signed integer.

    Args:
        x: The IEEE 754 binary128 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 32-bit signed integer.

    """
    return _make_int32(sf.f128_to_i32(x._data, rounding_mode, exact))


cpdef Int64 f128_to_i64(
    Float128 x, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
):
    """Converts the IEEE 754 binary128 floating point to a 64-bit signed integer.

    Args:
        x: The IEEE 754 binary128 floating point to be converted.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact conversion is unable.

    Returns:
        The 64-bit signed integer.

    """
    return _make_int64(sf.f128_to_i64(x._data, rounding_mode, exact))


cpdef Float16 f128_to_f16(Float128 x):
    """Converts the IEEE 754 binary128 floating point to a binary16 floating point.

    Args:
        x: The IEEE 754 binary128 floating point to be converted.

    Returns:
        The IEEE 754 binary16 floating point.

    """
    return _make_float16(sf.f128_to_f16(x._data))


cpdef Float32 f128_to_f32(Float128 x):
    """Converts the IEEE 754 binary128 floating point to a binary32 floating point.

    Args:
        x: The IEEE 754 binary128 floating point to be converted.

    Returns:
        The IEEE 754 binary32 floating point.

    """
    return _make_float32(sf.f128_to_f32(x._data))


cpdef Float64 f128_to_f64(Float128 x):
    """Converts the IEEE 754 binary128 floating point to a binary64 floating point.

    Args:
        x: The IEEE 754 binary128 floating point to be converted.

    Returns:
        The IEEE 754 binary64 floating point.

    """
    return _make_float64(sf.f128_to_f64(x._data))


cpdef Float128 f128_round_to_int(
    Float128 x, RoundingMode rounding_mode = get_rounding_mode(), bool exact = True
):
    """Rounds the number expressed as an IEEE 754 binary128 floating point.

    Args:
        x: The IEEE 754 binary128 floating point to be rounded.
        rounding_mode: The rounding mode.
        exact: If ``True`` is specified, the floating-point exception flags are to be set
               when exact rounding is unable.

    Returns:
        The resulted integer expressed as an IEEE 754 binary128 floating point.

    """
    return _make_float128(sf.f128_roundToInt(x._data, rounding_mode, exact))


cpdef Float128 f128_neg(Float128 x):
    """Negates the IEEE 754 binary128 floating point.

    Args:
        x: The floating point to be negated.

    Returns:
        The resulted number expressed as an IEEE 754 binary128 floating point (``-x``).

    """
    cdef ui128_f128 t
    t.f = x._data
    t.ui.v0 ^= <uint64_t>0x80000000_00000000
    cdef Float128 o = Float128()
    o._data = t.f
    return o


cpdef Float128 f128_add(Float128 x, Float128 y):
    """Adds the IEEE 754 binary128 floating points.

    Args:
        x: The floating point to be added.
        y: The floating point to add.

    Returns:
        The resulted number expressed as an IEEE 754 binary128 floating point (``x + y``).

    """
    return _make_float128(sf.f128_add(x._data, y._data))


cpdef Float128 f128_sub(Float128 x, Float128 y):
    """Subtracts the IEEE 754 binary128 floating points.

    Args:
        x: The floating point to be subtracted.
        y: The floating point to subtract.

    Returns:
        The resulted number expressed as an IEEE 754 binary128 floating point (``x - y``).

    """
    return _make_float128(sf.f128_sub(x._data, y._data))


cpdef Float128 f128_mul(Float128 x, Float128 y):
    """Multiplies the IEEE 754 binary128 floating points.

    Args:
        x: The floating point to be multiplied.
        y: The floating point to multiply.

    Returns:
        The resulted number expressed as an IEEE 754 binary128 floating point (``x * y``).

    """
    return _make_float128(sf.f128_mul(x._data, y._data))


cpdef Float128 f128_mul_add(Float128 x, Float128 y, Float128 z):
    """Multiplies and Adds the IEEE 754 binary128 floating points.

    Args:
        x: The floating point to be multiplied.
        y: The floating point to multiply.
        z: The floating point to add.

    Returns:
        The resulted number expressed as an IEEE 754 binary128 floating point (``x * y + z``).

    """
    return _make_float128(sf.f128_mulAdd(x._data, y._data, z._data))


cpdef Float128 f128_div(Float128 x, Float128 y):
    """Divides the IEEE 754 binary128 floating points.

    Args:
        x: The floating point to be divided.
        y: The floating point to divide.

    Returns:
        The resulted number expressed as an IEEE 754 binary128 floating point (``x / y``).

    """
    return _make_float128(sf.f128_div(x._data, y._data))


cpdef Float128 f128_rem(Float128 x, Float128 y):
    """Calculates a remainder by dividing the IEEE 754 binary128 floating points.

    Args:
        x: The floating point to be divided.
        y: The floating point to divide.

    Returns:
        The resulted number expressed as an IEEE 754 binary128 floating point (``x % y``).

    """
    return _make_float128(sf.f128_rem(x._data, y._data))


cpdef Float128 f128_sqrt(Float128 x):
    """Calculates a square root of the IEEE 754 binary128 floating point.

    Args:
        x: The floating point whose square root is to be calculated.

    Returns:
        The resulted number expressed as an IEEE 754 binary128 floating point (``sqrt(x)``).

    """
    return _make_float128(sf.f128_sqrt(x._data))


cpdef bool f128_eq(Float128 x, Float128 y):
    """Tests if the first one is equal to the second one expressed as IEEE 754 binary128 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

    """
    return sf.f128_eq(x._data, y._data)


cpdef bool f128_le(Float128 x, Float128 y):
    """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary128 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

    """
    return sf.f128_le(x._data, y._data)


cpdef bool f128_lt(Float128 x, Float128 y):
    """Tests if the first one is less than the second one expressed as IEEE 754 binary128 floating points.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

    """
    return sf.f128_lt(x._data, y._data)


cpdef bool f128_eq_signaling(Float128 x, Float128 y):
    """Tests if the first one is equal to the second one expressed as IEEE 754 binary128 floating points.

    The invalid exception flag is set for any NaN input, not just for signaling NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is equal to the second one, ``False`` otherwise (``x == y``).

    """
    return sf.f128_eq_signaling(x._data, y._data)


cpdef bool f128_le_quiet(Float128 x, Float128 y):
    """Tests if the first one is less than or equal to the second one expressed as IEEE 754 binary128 floating points.

    The invalid exception flag is not set for quiet NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than or equal to the second one, ``False`` otherwise (``x <= y``).

    """
    return sf.f128_le_quiet(x._data, y._data)


cpdef bool f128_lt_quiet(Float128 x, Float128 y):
    """Tests if the first one is less than the second one expressed as IEEE 754 binary128 floating points.

    The invalid exception flag is not set for quiet NaNs.

    Args:
        x: The first floating point to be compared.
        y: The second floating point to be compared.

    Returns:
        ``True`` if the first one is less than the second one, ``False`` otherwise (``x < y``).

    """
    return sf.f128_lt_quiet(x._data, y._data)


cpdef bool f128_is_signaling_nan(Float128 x):
    """Tests if the IEEE 754 binary128 floating point is a signaling NaN.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is a signaling NaN, ``False`` otherwise.

    """
    return sf.f128_isSignalingNaN(x._data)


cpdef bool f128_is_nan(Float128 x):
    """Tests if the IEEE 754 binary128 floating point is a NaN.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is a NaN, ``False`` otherwise.

    """
    cdef ui128_f128 t
    t.f = x._data
    return (~t.ui.v0 & 0x7fff0000_00000000) == 0 and (
        (t.ui.v0 & 0x0000ffff_ffffffff) != 0 or t.ui.v64 != 0
    )


cpdef bool f128_is_inf(Float128 x):
    """Tests if the IEEE 754 binary128 floating point is an infinity.

    Args:
        x: The floating point to be tested.

    Returns:
        ``True`` if the floating point is an infinity, ``False`` otherwise.

    """
    cdef ui128_f128 t
    t.f = x._data
    return (t.ui.v0 & 0x7fffffff_ffffffff) == 0x7fff0000_00000000 and t.ui.v64 == 0
