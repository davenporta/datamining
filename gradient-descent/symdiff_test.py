#!/usr/bin/env python3

"""Test module `symdiff` using doctests

Set up

>>> from symdiff import *
>>> from math import pi, e

Variables

>>> x, y, z = Var('x'), Var('y'), Var('z')
>>> x, y, z
(x, y, z)

Equality and Operator Overloading

>>> Var('x') == x != y
True
>>> Plus(x, 2) == x + 2
True
>>> Plus(2, x) == 2 + x
True
>>> 2 + x != 2 + y
True
>>> 2 + x != 2 * x
True
>>> x ** 2
(x ** 2)
>>> 2 - x
(2 + (-1 * x))
>>> 2 / x
(2 * (x ** -1))
>>> x += 2
Traceback (most recent call last):
  ...
NotImplementedError: Variables are immutable

String

>>> 2 + (x * y)
(2 + (x * y))
>>> Cos(Sin(x + y))
cos(sin(x + y))

Evaluation

>>> f = 0 + 1 + x + y + 1
>>> f, simplify(f), f(), f()()
((((1 + x) + y) + 1), (2 + x + y), (2 + x + y), (2 + x + y))
>>> f(y=0), f(y=2), f(x=2, y=3)
((2 + x), (4 + x), 7)

>>> simplify(1 * x * 1)
x
>>> (x * (y + -2))(y=2)
0
>>> cos(x)(x=pi)
-1.0
>>> simplify(x ** 1)
x
>>> simplify(x ** 0)
1

Differentiation

>>> gradient(f, x, y, z)
[1, 1, 0]
>>> diff(x * x * z, x)
((x + x) * z)
>>> diff(cos(x), x)
(-1 * sin(x))
>>> diff(x ** 3, x)
(3 * (x ** 2))
>>> diff(x ** 1, x)
1
>>> diff(x / y, y)
(-1 * (y ** -2) * x)

Functions

>>> f = Function(x * y + z, x, y, z)
>>> f(1, 2, 3)
5
>>> diff(f, x)
Function(y, x, y, z)

"""

if __name__ == '__main__':
    import doctest

    doctest.testmod()
