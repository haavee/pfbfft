def Hilbert45(N,a,w1,w2,d):
    from math import sin, cos, pi, sqrt
    output=[]

    for i in range(N):
        t = 2 * pi * (i - (N-1)/2)
        if(t == 0):
            o = sqrt(2)*(w2-w1)
        elif(t == pi/2*a):
            o = a*(sin(pi/4*((a+2*w2)/a))-sin(pi/4*((a+2*w1)/a)))
        elif(t == -pi/2*a):
            o = a*(sin(pi/4*((a-2*w1)/a))-sin(pi/4*((a-2*w2)/a)))
        else:
            o = 2*(pi**2)*cos(a*t)/(t*(4*(a**2)*(t**2)-pi**2))*(sin(w1*t+pi/4)-sin(w2*t+pi/4))
        output.append(o)
    if(d==1):
        output.reverse()
    return output

# always return A(t) because B(t) is simply A(t)[::-1]
def hilbert45_hv(N, a, w1, w2):
    from numpy import arange, isclose, piecewise
    from math  import pi, sin, cos, sqrt

    timestamps = 2*pi*(arange(N) - (N-1)/2)
    return piecewise(timestamps,
                     # the conditions
                     [
                           isclose(timestamps, 0)
                         , isclose(timestamps, pi/2*a)
                         , isclose(timestamps, -pi/2*a)
                     ],
                     # what to evaluate to
                     [ 
                          lambda t: sqrt(2)*(w2-w1)
                        , lambda t: a * (sin(pi/4*((a+2*w2)/a)) - sin(pi/4*((a+2*w1)/a)))
                        , lambda t: a * (sin(pi/4*((a-2*w1)/a)) - sin(pi/4*((a-2*w2)/a)))
                        , lambda t: 2*pi**2*cos(a*t)/(t*(4*a**2*t**2 - pi**2) * (sin(w1*t + pi/4) - sin(w2*t + pi/4)))
                     ])
              
#
#    # generate the time stamps
#    ts = 2*pi*(arange(N) - (N-1)/2)
#    print("ts=",ts)
#    # generic formula
#    at = 2*pi**2*cos(a*ts)/(ts*(4*a**2*ts**2 - pi**2) * (sin(w1*ts + pi/4) - sin(w2*ts + pi/4)))
#    print("at[gen]=",at)
#    # when t==0
#    at[ where(isclose(ts, 0)) ] = sqrt(2)*(w1 - w2)
#    print("at[t==0]=",at)
#    # when t ~ pi/2a or t~-pi/2*a 
#    at[ where(isclose(ts,  pi/2*a)) ] = 
#    print("at[t==pi/2a]=",at)
#    at[ where(isclose(ts, -pi/2*a)) ] = 
#    print("at[t==-pi/2a]=",at)
#    return at
