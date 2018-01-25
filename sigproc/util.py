from __future__ import print_function
import math, random, numpy, itertools, operator, functools, fractions
from functional_hv     import *

# freqs = [f0, f1, f2, ...., fN]
#  assume: freqs are increasing such that f(N+1) >= f(N)
# return:
#   ([f0', f1', f2', ..., fN'], "UNIT")
# with unit "kHz", "MHz", "GHz", whatever
multipliers = ["", "k", "M", "G", "T", "P", "E", "Z", "Y", "y", "z", "a", "f", "p", "n", "u", "m"]
#              0    3    6    9    12   15   18   21   24  -24  -21  -18  -15  -12   -9   -6   -3 
def rescale(freqs, unit):
    decades = int(math.log10(abs(freqs[-1] - freqs[0])) // 3)
    return (math.pow(10, decades*3), "{0}{1}".format(multipliers[decades], unit))

# Print a summary of a numarray
def summary(x, descr):
    # define what and how to print
    fmt   = "{0}:{1:+12.10f}".format
    props = [("min", numpy.min), ("max", numpy.max), ("avg", numpy.mean), ("std", numpy.std)]
    print(descr,"/"," ".join(map(lambda pair: fmt(pair[0], pair[1](x)), props)))

# courtesy https://gist.github.com/endolith/114336
def gcd(*numbers):
    """Return the greatest common divisor of the given integers"""
    return functools.reduce(fractions.gcd, numbers)

# Least common multiple is not in standard libraries? It's in gmpy, but this is simple enough:
def lcm(*numbers):
    """Return lowest common multiple."""    
    def lcm(a, b):
        return (a * b) // gcd(a, b)
    return functools.reduce(lcm, numbers)

# lcm for Fractions
import fractions

lcm_f_impl = lambda a, b: (a * b) / fractions.gcd(a, b)

def lcm_f(*numbers):
    return functools.reduce(lcm_f_impl, map(fractions.Fraction, numbers))



