
from __future__ import division
import numpy as np

DEFAULT = 0


def handle_exceptions(default):
    def wrap(f):
        def inner(*args):
            try:
                return f(*args)
            except Exception, e:
                return default
        return inner
    return wrap


@handle_exceptions(DEFAULT)
def _add(inp):
    return inp[0] + inp[1]


@handle_exceptions(DEFAULT)
def _mul(inp):
    return inp[0] * inp[1]


@handle_exceptions(DEFAULT)
def _div(inp):
    return inp[0] / inp[1]


@handle_exceptions(DEFAULT)
def _sub(inp):
    return inp[0] - inp[1]


@handle_exceptions(DEFAULT)
def _sqrt(inp):
    return np.sqrt(abs(inp[0]))


@handle_exceptions(DEFAULT)
def _or(inp):
    return int(inp[0] | inp[1])


@handle_exceptions(DEFAULT)
def _xor(inp):
    return int(inp[0] ^ inp[1])


@handle_exceptions(DEFAULT)
def _and(inp):
    return int(inp[0] & inp[1])


@handle_exceptions(DEFAULT)
def _lt(inp):
    return int(inp[0] < inp[1])


@handle_exceptions(DEFAULT)
def _gt(inp):
    return int(inp[0] > inp[1])


@handle_exceptions(DEFAULT)
def _not(inp):
    return int(~inp[0])


@handle_exceptions(DEFAULT)
def _max(inp):
    return max(inp[0], inp[1])


@handle_exceptions(DEFAULT)
def _min(inp):
    return min(inp[0], inp[1])


@handle_exceptions(DEFAULT)
def _if(inp):
    if inp[0]:
        return inp[1]
    return inp[2]


@handle_exceptions(DEFAULT)
def _cos(inp):
    return np.cos(inp[0])


@handle_exceptions(DEFAULT)
def _sin(inp):
    return np.sin(inp[0])


@handle_exceptions(DEFAULT)
def _cosh(inp):
    return np.cosh(inp[0])


@handle_exceptions(DEFAULT)
def _sinh(inp):
    return np.sinh(inp[0])


@handle_exceptions(DEFAULT)
def _tang(inp):
    return np.tan(inp[0])


@handle_exceptions(DEFAULT)
def _tanh(inp):
    return np.tanh(inp[0])


@handle_exceptions(DEFAULT)
def _log(inp):
    return np.log(abs(inp[0]))


@handle_exceptions(DEFAULT)
def _pow(inp):
    return np.power(abs(inp[0]), inp[1])


@handle_exceptions(DEFAULT)
def _exp(inp):
    return np.exp(inp[0])


@handle_exceptions(DEFAULT)
def _abs(inp):
    return abs(inp[0])


@handle_exceptions(DEFAULT)
def _inv(inp):
    return (1/inp[0])


@handle_exceptions(DEFAULT)
def _mod(inp):
    return (inp[0] % inp[1])
