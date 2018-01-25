import math, random, numpy, itertools, rdvdif, operator
from matplotlib import pylab

random.seed()


def gaussian_noise(amplitude=1.0, mean=0.0):
    while True:
        yield amplitude * numpy.random.randn() + mean


def sinewave(period, phase = 0.0, amplitude = 1.0):
    idx = 0
    PI2 = 2*math.pi
    SIN = math.sin
    while True:
        yield amplitude * SIN( PI2*idx/period + phase )
        idx = idx + 1

def channel_from_vdif(fn, ch):
    filedecoder = rdvdif.decode_file(fn)
    while True:
        curFrame = filedecoder.next()
        for sample in curFrame[ch]:
            yield sample


# fixed delay, in samples. Handles both +-ive and --ive values
def delay(n, samples):
    if n>0:
        while n:
            yield float(0)
            n = n-1
    elif n<0:
        while n:
            samples.next()
            n = n+1
    while True:
        yield samples.next()

def add(*signals):
    return itertools.imap(sum, itertools.izip(*signals))

def take(n, generator):
    result = []
    while n>0:
        result.append( generator.next() )
        n = n-1
    return result


#noise = mk_noise(amplitude=0.1)
#sine  = mk_sinewave(16, amplitude=0.5)

#samples = numpy.add(getn(sine, 64), getn(noise, 64))

def mk_sig(n, amps, ampn, period=16, phase=0):
    return mk_sig2(n, gaussian_noise(amplitude=ampn), sinewave(period, amplitude=amps, startphase=phase))

# Extract n samples from each signal generator and return
# an array of the summed samples
def mk_sig2(n, *signals):
    return take(n, add(*signals))


def fir_filter(coeff, stream):
    buff = [0.0] * len(coeff)
    while True:
        sample = stream.next()
        yield sum(itertools.imap(operator.mul, buff, coeff))
        del buff[-1]
        buff = [sample]+buff 


class fft:
    def __init__(self, npts, window=None):
        self.nPoints = npts
        self.Coeffs  = numpy.ones(self.nPoints) if window is None else window(self.nPoints)

    def fftSize(self):
        return self.nPoints

    def reset(self):
        pass

    def __call__(self, samples, verbose=True):
        # chunk the samples up in chunks of nPoints
        r   = []
        idx = 0
        cnt = 0
        while (idx + self.nPoints)<=len(samples):
            r.append( numpy.fft.fft(numpy.multiply(self.Coeffs,samples[idx:idx+self.nPoints])) )
            idx = idx + self.nPoints
            cnt = cnt + 1
        if verbose:
            print "fft: {0} iterations @{1} points".format( cnt, self.nPoints )
        return r


class blackman_harris:
    def __init__(self):
        pass

    def __call__(self, nPts):
        # compute coefficients according to:
        # http://en.wikipedia.org/wiki/Window_function#Generalized_Hamming_windows
        # w(n) = a0 - a1 * cos (2pi*n/(N-1)) + a2 * cos (4pi n / (N-1)) - a3 * cos( 6pi*n/(N-1))
        # a0 = 0.35875, a1 = 0.48829, a2 = 0.14128, a3 = 0.01168
        a0    = 0.35875; a1 = 0.48829; a2 = 0.14128; a3 = 0.01168
        Nmin1 = float(nPts-1)
        PI2   = 2*math.pi
        PI4   = 4*math.pi
        PI6   = 6*math.pi
        COS   = math.cos
        f = lambda n : a0 - a1*COS(PI2 * (n/Nmin1)) + a2*COS(PI4*(n/Nmin1)) - a3*COS(PI6*(n/Nmin1))
        return [f(n) for n in xrange(nPts)]


class sinc:
    def __init__(self):
        pass

    def __call__(self, nPts):
        # overlay a sinc function from -2*pi => 2*pi over the range
        # of points
        PI2   = 2*math.pi
        NPTS2 = nPts/2
        DPHI  = (2 * PI2) / nPts
        SIN   = math.sin
        def f(n):
            if n==NPTS2:
                return 1.0
            x = (n - NPTS2) * DPHI
            return SIN(x)/x
        return [f(n) for n in xrange(nPts)]

class hanning:
    def __init__(self):
        pass

    def __call__(self, nPts):
        # Hanning window: 0.5*(1-cos((2*PI*n)/N-1))
        PI2   = 2*math.pi
        COS   = math.cos
        NPTS1 = nPts - 1
        f     = lambda n: 0.5 - COS(PI2*n/NPTS1)
        return [f(n) for n in xrange(nPts)]


class bandpass:
    # xs, ys = arrays of x,y coordinates of desired
    #          frequency response in fractional bw units [0, 1]
    # E.g. for lowpass filter which passes lower 1/4 of the band
    # with a transition region of 0.02*bandwidth:
    #   xs = [0, 0.25-0.02, 0.25, 1.0]
    #   ys = [ 1         1,    0,   0]
    def __init__(self, xs, ys, window=None):
        if len(xs)==0 or len(ys)==0 or len(xs)!=len(ys):
            raise RuntimeError, "x/y are not non-zero length arrays of the same length"
        self.xs     = xs
        self.ys     = ys
        self.window = window

    def __call__(self, nPts):
        # interpolate over the requested nr of points
        nextx = 0
        grid  = []
        for curX in numpy.linspace(0, 1, nPts):
            while nextx<(len(self.xs)-1) and self.xs[nextx]<=curX:
                nextx = nextx + 1
            pX = self.xs[nextx-1]
            nX = self.xs[nextx]
            pY = self.ys[nextx-1]
            nY = self.ys[nextx]
            grid.append( pY + (curX - pX)*((nY-pY)/(nX-pX)) )
        # we have translated from fractional bandwidth to frequency repsonse 
        # values for freqbins 0..N-1
        # Now perform ifft
        coeff_ift = numpy.abs( numpy.fft.ifft(grid) )
        # Do windowing, if requested
        if not self.window is None:
            coeff_ift = numpy.multiply(coeff_ift, self.window(nPts) )
        # normalize
        f = coeff_ift[nPts/2]
        coeff_ift = numpy.divide(coeff_ift, f)
        return coeff_ift



class highpass:
    def __init__(self, lowerLimit):
        self.lowerLimit = abs(lowerLimit)

    def __call__(self, nPts):
        # http://www.vyssotski.ch/BasicsOfInstrumentation/SpikeSorting/Design_of_FIR_Filters.pdf
        # p.116:
        #   h(n) = sin(pi*n)/pi*n - sin(omega_c*n)/pi*n
        #   and multiply by a hann window and delay by N/2 [?]
        PI        = math.pi
        #response  = lambda f: 1.0 if (f<-self.lowerLimit or f>self.lowerLimit) else 0.0
        response  = lambda f: -math.cos(f)
        coeff_ift = numpy.fft.ifft( map(response, numpy.linspace(-PI, PI, nPts)) )
        window    = hanning()(nPts)
        return numpy.multiply(window, numpy.abs(coeff_ift))

class pfb:
    def __init__(self, nPts, Depth, Window):
        if Depth<1:
            raise RuntimeError, "PFB depth cannot be <1!"
        self.nPoints = nPts
        self.Depth   = Depth
        self.coeff   = numpy.array( list(reversed(Window(self.nPoints * self.Depth))) ).reshape( self.Depth, self.nPoints )
        #self.coeff   = numpy.array( Window(self.nPoints * self.Depth) ).reshape( self.Depth, self.nPoints )
        self.reset()

    def fftSize(self):
        return self.nPoints

    def reset(self):
        self.buff    = [numpy.zeros(self.nPoints) for i in range(self.Depth)]

    def __call__(self, samples, verbose=True):
        res = []
        idx = 0
        cnt = 0
        while (idx + self.nPoints)<=len(samples):
            # remove last element of buffer
            del self.buff[0]
            # append data
            self.buff.append( samples[idx:idx+self.nPoints] )
            # this will hold the sum
            tmp = numpy.zeros( self.nPoints )
            for i in range(self.Depth):
                tmp = numpy.add(tmp, numpy.multiply(self.buff[i], self.coeff[i,:]) )
            # do FFT to sum
            res.append( numpy.fft.fft(tmp) )
            idx = idx + self.nPoints
            cnt = cnt + 1
        if verbose:
            print "pfb/fft: {0} iterations @{1} points".format( cnt, self.nPoints )
        return res


#c = lambda : pylab.clf()
#p = lambda ds, attr=None: pylab.plot(ds, attr)
def pm(a, ls=None):
    pylab.interactive(True)
    for i in range(len(a)):
        if ls:
            pylab.plot(range(len(a[i])), numpy.abs(a[i]), ls)
        else:
            pylab.plot(numpy.abs(a[i]))
    pylab.show()

mk_steps = lambda s, e, step=0.1: numpy.linspace(min(s, e), max(s, e), int(float(abs(e-s))/step))

def measure_binshape(engine):
    # the idea is to loop over a number of sine-wave frequencies
    # and measure the amplitude of a pre-selected bin
    shape = []
    # from the FFT size and chosen bin, we compute the frequencies
    # that we'll iterate over to measure the binshape
    fftSz = engine.fftSize()
    ckBin = 85
    # we have to make a frequency sweep; make it evenly spaced in
    # *frequency*!
    sFreq = float(ckBin)-15
    eFreq = float(ckBin)+15
    print "measuring binshape @bin {0} [{1} -> {2})".format(ckBin, sFreq, eFreq)
    for f in mk_steps(sFreq, eFreq):
        # generate samples. 
        ffts = engine( take(8*1024, mk_sinewave(float(fftSz)/f)), verbose=False )

        # select the appropriate bin from the last FFT
        shape.append( numpy.abs(ffts[-1][ckBin]) )
    return shape


def extract(n):
    return lambda ds: ds[0:n]

def fx_cor(n, s1, s2, engine, nint=None, debug=False):
    #extractor = extract(engine.fftSize()/2)
    extractor = lambda x: x
    print "Generating FFT segments"
    engine.reset()
    fftsegments1 = map(extractor, engine(take(n, s1), verbose=False))
    engine.reset()
    fftsegments2 = map(extractor, engine(take(n, s2), verbose=False))
    print "Correlating them"
    r = []
    for (i, (ft1, ft2)) in enumerate(zip(fftsegments1, fftsegments2)):
        # start a new integration if necessary
        if nint is None or (not (nint is None) and (i % nint)==0):
            r.append( numpy.zeros( engine.fftSize() ) )
            #r.append( numpy.zeros( engine.fftSize()/2 ) )
        r[-1] = numpy.add(r[-1], numpy.multiply(ft1, ft2) )
    if debug:
        return (r, fftsegments1, fftsegments2)
    else:
        if nint is None:
            return r[0]
        else:
            return r


