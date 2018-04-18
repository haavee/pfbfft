import functools, fractions

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

lcm_f_impl = lambda a, b: (a * b) / fractions.gcd(a, b)

def lcm_f(*numbers):
    return functools.reduce(lcm_f_impl, map(fractions.Fraction, numbers))
