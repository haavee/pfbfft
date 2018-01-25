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
