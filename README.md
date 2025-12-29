# SoftFloatPy

## Overview

**SoftFloatPy** is a Python binding of *Berkeley SoftFloat*.

[Berkeley SoftFloat](https://www.jhauser.us/arithmetic/SoftFloat.html) is a software implementation of binary floating-point functionalities that conforms to the IEEE Standard for Floating-Point Arithmetic (IEEE 754 and the succeeding standards).

As of December 2025, the latest release is Berkeley SoftFloat 3e, and SoftFloatPy supports the following features in it.
- four binary floating-point formats:
    - 16-bit half-precision format (the binary16 format),
    - 32-bit single-precision format (the binary32 format),
    - 64-bit double-precision format (the binary64 format),
    - 128-bit quadruple-precision format (the binary128 format).
- addition, subtraction, multiplication, division, remainder, fused multiply-add, and square root.
- round to an integer value.
- comparisons.
- conversions between the supported floating-point formats.
- conversions between the 32-bit single-precision format and the 16-bit brain floating-point format (bfloat16).

The following features that are not in the IEEE 754 standard are excluded from SoftFloatPy support.
- 80-bit extended format (known as x86 extended-precision format).
- odd-rounding mode (known as jamming).

SoftFloatPy requires Python 3.11 or later. The build configuration provided in the [SoftFloatPy repository](https://github.com/arithy/softfloatpy) assumes platforms with 64-bit integer arithmetic support.

**Links related to Berkeley SoftFloat:**
- https://www.jhauser.us/arithmetic/SoftFloat.html
- https://github.com/ucb-bar/berkeley-softfloat-3

## Installation

### Release Version

You can install the release version by the following command.

```sh
$ python -m pip install softfloatpy
```

### Development Version

You can install the development version by the following commands.

```sh
$ cd softfloatpy  # the repository root directory
$ make req
$ make clean
$ make dist
$ python -m pip install --no-index --find-links=./dist softfloatpy
```

## Usage

### Import of Module

To use SoftFloatPy in a Python script, import `softfloatpy` module. An example is shown below.
```py
import softfloatpy as sf
```

### Creation of Objects

To use the SoftFloatPy functions, you need to create objects of the classes shown below.
- Floating-Point Classes:
    - [`Float16`](https://arithy.github.io/softfloatpy/apidoc/softfloatpy.html#softfloatpy.Float16)
    - [`Float32`](https://arithy.github.io/softfloatpy/apidoc/softfloatpy.html#softfloatpy.Float32)
    - [`Float64`](https://arithy.github.io/softfloatpy/apidoc/softfloatpy.html#softfloatpy.Float64)
    - [`Float128`](https://arithy.github.io/softfloatpy/apidoc/softfloatpy.html#softfloatpy.Float128)
    - [`BFloat16`](https://arithy.github.io/softfloatpy/apidoc/softfloatpy.html#softfloatpy.BFloat16)
- Fixed-Bit Integer Classes:
    - [`UInt32`](https://arithy.github.io/softfloatpy/apidoc/softfloatpy.html#softfloatpy.UInt32)
    - [`UInt64`](https://arithy.github.io/softfloatpy/apidoc/softfloatpy.html#softfloatpy.UInt64)
    - [`Int32`](https://arithy.github.io/softfloatpy/apidoc/softfloatpy.html#softfloatpy.Int32)
    - [`Int64`](https://arithy.github.io/softfloatpy/apidoc/softfloatpy.html#softfloatpy.Int64)

For the floating-point classes, there are two options to create its object.
- Creation from `bytes`. Below is an example of `Float16` with the value 1.0.
  ```py
  f = sf.Float16.from_bytes(b'\x3c\x00')  # The byte order is big-endian.
  ```
- Creation from `float`. Below is an example of `Float16` with the value 1.0.
  ```py
  f = sf.Float16.from_float(1.0)
  ```

For the fixed-bit integer classes, there are two options to create its object.
- Creation from `bytes`. Below is an example of `Int32` with the value 1.
  ```py
  i = sf.Int32.from_bytes(b'\x00\x00\x00\x01')  # The byte order is big-endian.
  ```
- Creation from `int`. Below is an example of `Int32` with the value 1.
  ```py
  i = sf.Int32.from_int(1)
  ```

### Arithmetic of Objects

Once creating the objects, you can use the functions described in the [SoftFloat documentation](https://www.jhauser.us/arithmetic/SoftFloat-3/doc/SoftFloat.html).
  ```py
  a = sf.Float16.from_float(2.0)
  b = sf.Float16.from_float(-3.0)
  c = sf.f16_mul(a, b)
  ```

In addition, you can use Python operators for arithmetic operations such as `+`, `-`, `*`, `//`, and `%`. For more details, see the [SoftFloatPy API documentation](https://arithy.github.io/softfloatpy/apidoc/softfloatpy.html).
  ```py
  a = sf.Float16.from_float(2.0)
  b = sf.Float16.from_float(-3.0)
  c = a * b
  ```

### Conversion to Built-in Types

You can retrieve the value retained in an object as a Python built-in type.

For the floating-point classes, there are two built-in types to retrieve the value.
- Retrieve the value as `bytes`. Below is an example of `Float16`.
  ```py
  print(f.to_bytes())
  # -> b'\x3c\x00'  (The byte order is big-endian.)
  ```
- Retrieve the value as `float`. Below is an example of `Float16`.
  ```py
  print(f.to_float())
  # -> 1.0
  ```

For the fixed-bit integer classes, there are two built-in types to retrieve the value.
- Retrieve the value as `bytes`. Below is an example of `Int32`.
  ```py
  print(i.to_bytes())
  # -> b'\x00\x00\x00\x01'  (The byte order is big-endian.)
  ```
- Retrieve the value as `int`. Below is an example of `Int32`.
  ```py
  print(i.to_int())
  # -> 1
  ```

The other option is to create a string representation of the object value using `str()` or formatted strings.
  ```py
  print(f'value: {f}')
  # -> value: 1.0
  ```

### Setting of Rounding Mode

You can set and get the default rounding mode using the functions below.
- [`set_rounding_mode()`](https://arithy.github.io/softfloatpy/apidoc/softfloatpy.html#softfloatpy.set_rounding_mode)
- [`get_rounding_mode()`](https://arithy.github.io/softfloatpy/apidoc/softfloatpy.html#softfloatpy.get_rounding_mode)

The rounding mode is a thread-local property.

### Check of Floating-Point Exceptions

You can set, get, and test the floating point exceptions using the functions below.
- [`set_exception_flags()`](https://arithy.github.io/softfloatpy/apidoc/softfloatpy.html#softfloatpy.set_exception_flags)
- [`get_exception_flags()`](https://arithy.github.io/softfloatpy/apidoc/softfloatpy.html#softfloatpy.get_exception_flags)
- [`test_exception_flags()`](https://arithy.github.io/softfloatpy/apidoc/softfloatpy.html#softfloatpy.test_exception_flags)

The floating-point exceptions are thread-local properties.
