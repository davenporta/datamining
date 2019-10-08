#!/usr/bin/env python3

import math
import numpy as np
import operator
from functools import reduce
from abc import ABCMeta, abstractmethod, abstractstaticmethod

"""Implement symbolic differentiation

This module performs symbolic differentiation and expression simplification on
 objects called expressions. Expressions include variables and combinations
 thereof using functions like addition and cosine.

Sample usage looks like this:

    >>> x, y = Var('x'), Var('y')  # Construction variables
    >>> f = (1 + x^2 + y) * 3  # Constructing functions
    >>> f(x=0, y=1)  # Evaluation
    6
    >>> f(x=0)  # Partial evaluation
    (1 + y) * 3
    >>> diff(f, x)  # Differentiation
    6 * x
    >>> diff(f, x)(x=2)  # Evaluation of derivative
    12
    >>> gradient(f, x, y)  # Taking gradients
    [6 * x, 3]

If you want to be able to write f(1, 2) instead of f(x=1, f=2), you can use the
 Function class as so:


    >>> f = Function(x * y + z, x, y, z)
    >>> f(1, 2, 3)
    5
    >>> diff(f, x)
    Function(y, x, y, z)

More examples of usage can be found in the `symdiff_test.py` module, which uses doctests.

"""


def is_exp(obj):
    return isinstance(obj, Expression)


def is_const(obj):
    return not is_exp(obj)


def evaluate(obj, *args, **kwargs):
    """Partially or fully evaluate and simplify an object

    There are two ways of passing in the value of variables. The first way is
     using `kwargs`, which should be a dictionary from variable names to
      values. For example, evaluate(f, x=2, y=3).

    The other way is to use `args`. For example, evaluate(f, 2, 3). This
     way is only applicable for expressions that are Functions.

    Do not use both `args` and `kwargs` at the same time.
    """

    if is_const(obj):
        return obj

    elif type(obj) == Function:
        return obj._evaluate(*args, **kwargs)

    else:
        assert not args, ValueError(
            "Cannot pass in `vals_list` for non-Functions")
        return obj._evaluate(**kwargs)


def simplify(obj):
    return evaluate(obj)


def diff(obj, var):
    """Differentiate an object with respect to a variable and simplify"""
    assert type(var) == Var
    return simplify(obj._diff(var) if is_exp(obj) else 0)


def gradient(obj, *vars):
    """Return the gradient of an expression that is the function of `vars`

    `vals` should be a list of variables in the order in which the
     partial derivatives should be returned. For example:

    >>> x, y, z = Var('x'), Var('y'), Var('z')
    >>> f = x + y * z
    >>> gradient(f, y, x)
    [z, 1]

    If `obj` is a function, then `vars` should not be passed in.
    """
    if type(obj) == Function:
        assert not vars, ValueError("`vars` unnecessary for gradients of Functions")
        vars = obj.vars
    return [diff(obj, var) for var in vars]


def _equal(a, b):
    """Compare equality, guarding against numpy matrices

    Numpy arrays raise an error on being compared to a constant, necessitating
     this method.
    """
    return type(a) != np.ndarray and a == b


""" Core Classes """


class Expression(metaclass=ABCMeta):
    "Represent an expression that can be differentiated and evaluated"

    """ Core Functionality """

    @abstractmethod
    def _evaluate(self, **vals):
        pass

    @abstractmethod
    def _diff(self, var):
        pass

    def __eq__(self, other):
        return type(self) == type(other) and self.__dict__ == other.__dict__

    @abstractmethod
    def __repr__(self):
        pass

    """ Syntax Sugar """

    IMMUTABLE_ERROR_MESSAGE = "Variables are immutable"

    def __call__(self, *args, **kwargs):
        return evaluate(self, *args, **kwargs)

    def __pos__(self):
        return self

    def __neg__(self):
        return -1 * self

    def __add__(self, other):
        return Plus(self, other)

    def __radd__(self, other):
        return Plus(other, self)

    def __iadd__(self, other):
        raise NotImplementedError(self.IMMUTABLE_ERROR_MESSAGE)

    def __mul__(self, other):
        return Multiply(self, other)

    def __rmul__(self, other):
        return Multiply(other, self)

    def __imul__(self, other):
        raise NotImplementedError(self.IMMUTABLE_ERROR_MESSAGE)

    def __pow__(self, power, modulo=None):
        if modulo is not None:
            return NotImplemented
        return Power(self, power)

    def __rpow__(self, other):
        return Power(other, self)

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return other + -self

    def __isub__(self, other):
        raise NotImplementedError(self.IMMUTABLE_ERROR_MESSAGE)

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def __itruediv__(self, other):
        raise NotImplementedError(self.IMMUTABLE_ERROR_MESSAGE)


class Var(Expression):
    """Represent variable with respect to which an expression can be differentiated and evaluated"""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def _evaluate(self, **vals):
        return vals.get(self.name, self)

    def _diff(self, var):
        return int(self == var)


class Function(Expression):
    """Represent expression that knows the order of its variables

    >>> x, y = Var('x'), Var('y')
    >>> f = Function(x + y, x, y)
    >>> f(1, 2)
    3
    """

    def __init__(self, exp, *vars):
        self.exp = exp
        self.vars = vars

    def __repr__(self):
        return "Function({}, {})".format(self.exp, ", ".join(map(str, self.vars)))

    def _evaluate(self, *vals_list, **vals_dict):
        assert not (vals_list and vals_dict), ValueError(
            "Cannot pass in both `vals_list` and `vals_dict` simultaneously"
        )

        if vals_list:
            assert len(vals_list) == len(self.vars), ValueError(
                "Incorrect number of arguments passed in")
            vals_dict = {var.name: val for var, val in zip(self.vars, vals_list)}

        new_exp = evaluate(self.exp, **vals_dict)
        if is_const(new_exp):
            return new_exp

        return Function(new_exp, *self.vars)

    def _diff(self, var):
        return Function(self.exp._diff(var), *self.vars)


""" Abstract Helper Classes """


class PolynaryInfixOperator(Expression, metaclass=ABCMeta):
    """Represent infix operator taking any number of arguments

    Overriding classes must define the class constants:
    * `SYMBOL`: string representing the infix operator
    * `IDENTITY` operator’s identity (`None` if N/A)
    * `OPERATOR`: operator function
    """

    def __init__(self, *args):
        self.args = args

    def __repr__(self):
        return "(" + " {} ".format(self.SYMBOL).join(map(str, self.args)) + ")"

    def _evaluate(self, **vals):
        new_args = [evaluate(arg, **vals) for arg in self.args]

        # If one of the arguments is the the same polynary infix operator
        #  then squash its arguments (e.g. Plus(1, Plus(2, 3)) becomes Plus(1, 2, 3))
        squashed_args = []
        for arg in new_args:
            if isinstance(arg, self.__class__):
                squashed_args.extend(arg.args)
            else:
                squashed_args.append(arg)

        # Combine all constants and bring the constant up front
        #  (e.g. 1 + x + 2 becomes 3 + x)
        combined_args = (
            [reduce(self.OPERATOR, list(filter(is_const, squashed_args)), self.IDENTITY)]  # Combined constants
            + [arg for arg in squashed_args if is_exp(arg)])  # Non-constant arguments

        # Remove any identities
        try:
            non_zero_args = [arg for arg in combined_args if not _equal(arg, self.IDENTITY)]
        except:
            print([type(arg) for arg in combined_args])
            raise

        # Return as simple a value as possible
        if not non_zero_args:
            return self.IDENTITY  # empty sum or empty product or the like
        elif len(non_zero_args) == 1:
            return non_zero_args[0]  # constant return value

        return self.__class__(*non_zero_args)


class UnaryFunction(Expression, metaclass=ABCMeta):
    """Represent differentiable function taking one argument

    Overriding classes must define the following constants/methods:
    * `SYMBOL`: string representing the function
    * `_evaluate_helper`: static method that takes a constant and returns a constant
    """

    def __init__(self, arg):
        self.arg = arg

    def __repr__(self):
        arg_repr = repr(self.arg)
        if arg_repr[0] != '(':
            arg_repr = '(' + arg_repr + ')'
        return "{}{}".format(self.SYMBOL, arg_repr)

    @abstractstaticmethod
    def _evaluate_helper(self, arg):
        pass

    def _evaluate(self, **vals):
        new_arg = evaluate(self.arg, **vals)
        if is_const(new_arg):
            return self._evaluate_helper(new_arg)
        return self.__class__(new_arg)


""" Basic Operations """


class Plus(PolynaryInfixOperator):
    SYMBOL = "+"
    IDENTITY = 0
    OPERATOR = operator.add

    # TODO: In `_evaluate`, combine repeated expressions with multiplication

    def _diff(self, var):
        return sum(
            diff(arg, var)
            for arg in self.args
        )


class Multiply(PolynaryInfixOperator):
    SYMBOL = "*"
    IDENTITY = 1
    OPERATOR = operator.mul

    def _evaluate(self, **vals):
        obj = super()._evaluate(**vals)

        # Zero Product Property
        if isinstance(obj, Multiply) and 0 in obj.args:
            return 0

        # TODO: Combine repeated expressions with exponentiation

        return obj

    def _diff(self, var):
        return sum(
            diff(arg, var) * Multiply(*self.args[:i], *self.args[i + 1:])
            for i, arg in enumerate(self.args)
        )


class Power(Expression):
    def __init__(self, base, exp):
        self.base = base
        self.exp = exp

    def __repr__(self):
        return "({} ** {})".format(self.base, self.exp)

    def _evaluate(self, **vals):
        base = evaluate(self.base, **vals)
        exp = evaluate(self.exp, **vals)

        if _equal(exp, 0):
            return 1
        elif _equal(exp, 1):
            return base

        if _equal(base, 0):
            return 0
        elif _equal(base, 1):
            return 1

        return base ** exp

    def _diff(self, var):
        return self.base ** (self.exp - 1) * (
            self.exp * diff(self.base, var)
            + self.base * ln(self.base) * diff(self.exp, var)
        )


""" Misc Functions """


class Sin(UnaryFunction):
    SYMBOL = "sin"
    _evaluate_helper = math.sin

    def _diff(self, var):
        return Cos(self.arg) * diff(self.arg, var)


class Ln(UnaryFunction):
    SYMBOL = "ln"
    _evaluate_helper = math.log

    def _diff(self, var):
        return self.arg ** -1 * diff(self.arg, var)


class Cos(UnaryFunction):
    SYMBOL = "cos"
    _evaluate_helper = math.cos

    def _diff(self, var):
        return -1 * Sin(self.arg) * diff(self.arg, var)


cos = Cos
sin = Sin
ln = Ln
