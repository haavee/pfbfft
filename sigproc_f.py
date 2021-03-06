#import math, random, numpy, itertools, rdvdif3, operator, functools, fractions, userinput, hilbert, peak
import math, random, numpy, itertools, rdvdif3, operator, functools, fractions, hilbert, peak
# magic!  (ylppa = apply spelled backwards :D)
ylppa = lambda *args, **kwargs: lambda f: f(*args, **kwargs)


#D = userinput.debug
D = lambda *args: None

class Spectrum(numpy.ndarray):
    def __new__(cls, input_array, **kwargs):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = numpy.asarray(input_array).view(cls)
        # add the new attributes to the created instance
        obj.__dict__ = kwargs
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is not None:
            self.__dict__ = getattr(obj, '__dict__', dict())

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

def idx_peakert(x, threshold=0.9, n=3):
    indices = sorted(peak.indexes(x, threshold), key=lambda p: x[p])
    return indices[:(n if len(indices)>n else None)]

#
random.seed()

def summary(x, descr):
    # define what and how to print
    fmt   = "{0}:{1:+12.10f}".format
    props = [("min", numpy.min), ("max", numpy.max), ("avg", numpy.mean), ("std", numpy.std)]
    print(descr,"/"," ".join(map(lambda pair: fmt(pair[0], pair[1](x)), props)))

def rdbeamform(fn):
    # code for reading n heaps:
    #a = numpy.fromfile(fn, dtype=numpy.int8, count=1*1024*128*2)
    #b = a.astype(numpy.float32)
    #c = b.view(numpy.complex64)
    #d = c.reshape(1, 1024, 128).transpose((0, 2, 1))
    with open(fn) as infile:
        # read one heap
        heapnum = 0
        while True:
            try:
                curHeap = numpy.fromfile(infile, dtype=numpy.int8, count=1024*128*2).astype(numpy.float32).view(numpy.complex64).reshape((1024,128)).T #transpose((1,0))
                for s in range(curHeap.shape[0]):
                    yield curHeap[s][::-1]
                heapnum = heapnum + 1
            except Exception as E:
                print("Caught exception reading heap #",heapnum," - ",E)
                raise StopIteration

def rdbeamform_raw(fn):
    # code for reading n heaps:
    #a = numpy.fromfile(fn, dtype=numpy.int8, count=1*1024*128*2)
    #b = a.astype(numpy.float32)
    #c = b.view(numpy.complex64)
    #d = c.reshape(1, 1024, 128).transpose((0, 2, 1))
    with open(fn) as infile:
        # read one heap
        heapnum = 0
        while True:
            try:
                curHeap = numpy.fromfile(infile, dtype=numpy.int8, count=1024*128*2).astype(numpy.float32).view(numpy.complex64).reshape((1024,128)).T #transpose((1,0))
                for s in range(curHeap.shape[0]):
                    yield curHeap[s]
                heapnum = heapnum + 1
            except Exception as E:
                print("Caught exception reading heap #",heapnum," - ",E)
                raise StopIteration

def rdbeamform_meerkat(fn, nspec=128, start_spectrum=0):
    # meerkat beamform is in HDF5
    import h5py, pickle
    with h5py.File(fn, 'r') as infile:
        data        = infile['Data']['bf_raw']
        tel_state   = infile['TelescopeState']
        bandwidth   = pickle.loads(tel_state.attrs['i0_bandwidth'])
        centre_freq = pickle.loads(tel_state.attrs['sdp_l0_center_freq'])
        dc_freq     = centre_freq - bandwidth/2.0
        print("rdbeamform_meerkat: file=",fn," start_spectrum=",start_spectrum,"  nspec=",nspec, " data.shape=",data.shape)
        while True:
            try:
                yield Spectrum( data[:,start_spectrum:start_spectrum+nspec,:]
                            .astype(numpy.float32)\
                            .view(numpy.complex64)\
                            .squeeze()\
                            .T,\
                            DCEdge=0, DCFrequency=dc_freq, BW=bandwidth)
                start_spectrum += nspec
            except Exception as E:
                print("Caught exception reading spectrum #",start_spectrum," - ",E)
                raise StopIteration

def rdbeamform_heap(fn, nheap=1, start_spectrum=0):
    # code for reading n heaps:
    #a = numpy.fromfile(fn, dtype=numpy.int8, count=1*1024*128*2)
    #b = a.astype(numpy.float32)
    #c = b.view(numpy.complex64)
    #d = c.reshape(1, 1024, 128).transpose((0, 2, 1))
    with open(fn) as infile:
        # compute start of heap
        heapnum        = start_spectrum // 128
        start_spectrum = start_spectrum - (heapnum * 128)
        infile.seek(heapnum * 128 * 2048)
        print("rdbeamform_heap: file=",fn," start_spectrum=",start_spectrum," heap=",heapnum," seek to ",heapnum * 128 * 2048, " nheap=",nheap)
        while True:
            try:
                yield Spectrum(numpy.fromfile(infile, dtype=numpy.int8, count=nheap*1024*128*2)\
                                .astype(numpy.float32)\
                                .view(numpy.complex64)\
                                .reshape((nheap, 1024,128))\
                                .transpose((0, 2, 1))\
                                .reshape((nheap * 128, 1024)),
                                DCEdge=(1700.49e6 + 200e6), BW=-400e6)[start_spectrum:]
                start_spectrum = 0
                heapnum        = heapnum + nheap
            except Exception as E:
                print("Caught exception reading heap #",heapnum," - ",E)
                raise StopIteration

def averager(quant, src):
    data_sum = quant_sum = None #numpy.ndarray(0)
    for s in src:
        data_sum  = numpy.vstack([data_sum, s]) if data_sum is not None else s
        quant_sum = quant_sum + quant(s) if quant_sum is not None else quant(s)
    yield quant_sum/data_sum.shape[0]
    yield quant(numpy.average(data_sum, axis=1))

def quantizer(quant, src):
    for s in src:
        yield quant(s)

# assume mods = [ (idx, value), .... ]
def replacer(mods, src):
    (indices, values) = zip(*mods)
    indices           = list(indices)
    values            = numpy.array( list(values) )
    for s in src:
        s[indices] = values
        yield s

def replacer_h(mods, src):
    (indices, values) = zip(*mods)
    indices           = list(indices)
    values            = numpy.array( list(values) )
    for s in src:
        s[:,indices] = values
        yield s

def auto_comb(n, src):
    for spectrum in src:
        # generate spikes
        spike_idx = numpy.linspace(len(spectrum)/n, len(spectrum), n, endpoint=False, dtype=numpy.uint32)
        m         = numpy.max(spectrum)
        spectrum[ spike_idx ] = m*(1.5 +  numpy.sin( numpy.linspace(0, math.pi, len(spike_idx)) ))
        yield spectrum

def take(n, src):
    while n>0:
        yield next(src)
        n = n-1

def testbf(src):
    import matplotlib.pyplot as plt

    f, plots = plt.subplots(nrows=2, ncols=1)
    for (i,s) in enumerate(src):
        plots[i % 2].plot( s )
    plt.show()

# freq is assumed to be in Hz
PI2 = 2*math.pi
SIN = math.sin
def sinewave(freq, phase=0.0, amplitude=1.0):
    period = 1.0/freq
    def gen(ts):
        # compute the value for the given timestamp (unit: seconds)
        return amplitude * SIN( PI2 * ts/period + phase )
    return gen


def gaussian_noise(amplitude=1.0, mean=0.0):
    def gen(ts):
        return amplitude * numpy.random.randn() + mean
    return gen

def sample(*signals):
    def gen(ts):
        return map(ylppa(ts), signals)
    return gen

def add(*signals):
    sampler = sample(*signals)
    def gen(ts):
        return sum(sampler(ts))
        #return sum(map(ylppa(ts), signals))
    return gen

def mix(*signals):
    sampler = sample(*signals)
    def gen(ts):
        return functools.reduce(operator.mul, sampler(ts))
        #return reduce(operator.mul, map(ylppa(ts), signals))
    return gen

def take(n, generator):
    result = []
    while n>0:
        result.append( next(generator) )
        n = n-1
    return result

# SNR is the ratio of the sine wave compared to the mean (guassian) noise
def sine_embedded_in_noise(freq, SNR=0.001):
    return add(gaussian_noise(amplitude=1.0), sinewave(freq, amplitude=SNR))

def sines_embedded_in_noise(freqs, SNR=0.001):
    return add(gaussian_noise(amplitude=1.0), *map(lambda f: sinewave(f, amplitude=SNR), freqs))

def mk(snr, freq=6, sr=100):
    s = list(map(sine_embedded_in_noise(freq, SNR=snr), numpy.linspace(0,1,sr)))
    return (s, numpy.fft.fft(s))

def mk2(line, lo, sr=100, snr=0.01, nseconds=1):
    s = list(map(mix(sine_embedded_in_noise(line, SNR=snr), sinewave(lo)), numpy.linspace(0,nseconds*sr,sr)))
    return (s, numpy.fft.fft(s))

def d(freq=6, snr=4):
    #import matplotlib
    #matplotlib.use('MacOSX')
    import matplotlib.pyplot as plt
    (sr_mi, sr_ma) = int(freq/2), int(freq*3)
    for i in range(sr_mi,sr_ma,max(int((sr_ma-sr_mi)/20), 1)):
        plt.figure().clear()
        (z, ftz) = mk(snr, freq=freq, sr=i)
        plt.plot(z); plt.plot( numpy.abs(ftz) ); plt.title("SR={0} F={1}".format(i, freq)); plt.show()


# take a time series and do digital downconversion
def ddc1(lo, bw, *signals):
    import scipy.signal as SIGNAL
    import matplotlib.pyplot as plt

    # mix with cosine [should be complex?]
    mixed = mix(sinewave(lo, phase=math.pi/2), *signals)

    # we need a lowpass filter
    sr    = 8000
    n_tap = 128
    h = SIGNAL.firwin(n_tap, bw / sr)

    series      = list(map(mixed, numpy.linspace(0, 1, sr)))
    ft_series   = numpy.fft.fft(series)
    series_f    = SIGNAL.lfilter(h, 1, series)
    ft_series_f = numpy.fft.fft(series_f)

    f, plots = plt.subplots(nrows=4, ncols=1)
    plots[0].plot( series )
    plots[1].plot( numpy.abs(ft_series[:sr//2]) )
    plots[2].plot( series_f )
    plots[3].plot( numpy.abs(ft_series_f[:int(2*bw)]) )
    plt.show()

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


# do samplerate conversion; sr_in, sr_out integer amount of samples / second
#  need to find L, M such that L * sr_in / M = sr_out
#  L = lcm(sr_in, sr_out) / sr_in
#  M = lcm(sr_in, sr_out) / sr_out
def ddc2(lo, bw, sr_in, sr_out, *signals):
    import scipy.signal as SIGNAL
    import matplotlib.pyplot as plt

    # Get L, M
    mult  = lcm(sr_in, sr_out)
    L     = mult // sr_in
    M     = mult // sr_out
    print("DDC/sr_in={0} sr_out={1}, using sr_in * {2} / {3} = sr_out".format(sr_in, sr_out, L, M))
    # mix with cosine [should be complex?]
    mixed = mix(sinewave(lo, phase=math.pi/2), *signals)

    # we need a lowpass filter
    n_tap = 128
    h = SIGNAL.firwin(n_tap, bw / sr_in)

    series      = list(map(mixed, numpy.linspace(0, 1, sr_in)))
    ft_series   = numpy.fft.fft(series)
    # upsample
    up          = numpy.zeros(len(series)*L)
    up[::L]     = series

    # 'kay. Now filter that upsampled series
    series_f    = SIGNAL.lfilter(h, 1, up)
    ft_series_f = numpy.fft.fft(series_f)

    # and downsample
    down        = series_f[::M]
    ft_down     = numpy.fft.fft(down)

    f, plots = plt.subplots(nrows=6, ncols=1)
    plots[0].plot( series )
    plots[1].plot( numpy.abs(ft_series[:sr_in//2]) )
    plots[2].plot( series_f )
    plots[3].plot( numpy.abs(ft_series_f[:int(2*bw)]) )
    plots[4].plot( down )
    plots[5].plot( numpy.abs(ft_down[:int(2*bw)]) )
    plt.show()

#peaks = lambda cutoff: lambda series: numpy.where(numpy.abs(series)>(cutoff*numpy.amax(numpy.abs(series))))[0]
#peaks = lambda nsd: lambda series: numpy.where(numpy.abs(series)>(numpy.mean(numpy.absseries)+nsd*numpy.std(numpy.abs(series))))[0]
def peaks_sd(nsd):
    def doit(series):
        a = numpy.abs(series)
        return numpy.where(a>(numpy.mean(a)+nsd*numpy.std(a)))[0]
    return doit
def peaks_peak(cutoff):
    def doit(series):
        a = numpy.abs(series)
        return numpy.where(a>=(cutoff*numpy.max(a)))[0]
    return doit

def ddc3(lo, sr_in, sr_out, *signals):
    import scipy.signal as SIGNAL
    import matplotlib.pyplot as plt

    # Get L, M
    mult  = lcm(sr_in, sr_out)
    L     = mult // sr_in
    M     = mult // sr_out
    print("DDC/sr_in={0} sr_out={1}, using sr_in * {2} / {3} = sr_out".format(sr_in, sr_out, L, M))
    # mix with cosine [should be complex?]
    mixed = mix(sinewave(lo, phase=math.pi/2), *signals)

    # we need a lowpass filter - take sr_out / 2 as Nyquist limited band
    n_tap = 128
    h = SIGNAL.firwin(n_tap, sr_out/2.0/sr_in) #bw / sr_in)

    series      = list(map(mixed, numpy.linspace(0, 1, sr_in)))
    ft_series   = numpy.fft.fft(series)
    # upsample
    up          = numpy.zeros(len(series)*L)
    up[::L]     = series

    # 'kay. Now filter that upsampled series
    series_f    = SIGNAL.lfilter(h, 1, up)
    ft_series_f = numpy.fft.fft(series_f)

    # and downsample
    down        = series_f[::M]
    ft_down     = numpy.fft.fft(down)

    all_peaks   = list(map(peaks_peak(0.8), [ft_series, ft_series_f[:sr_in//2], ft_down[:sr_out//2]]))
    print("peaks in FFT of mixed signal:", all_peaks[0])
    print("peaks in FFT of upsampled+filtered mixed signal:",all_peaks[1]," +LO=",all_peaks[1]+lo)
    print("peaks in FFT of downsampled signal:",all_peaks[2]," +LO=",all_peaks[2]+lo)

    f, plots = plt.subplots(nrows=6, ncols=1)
    plots[0].plot( series )
    plots[1].plot( numpy.abs(ft_series[:sr_in//2]) )
    # series_f is the upsampled-by-L-and-filtered signal
    plots[2].plot( series_f )
    plots[3].plot( numpy.abs(ft_series_f[:(sr_in * L)//2]) )
    # downsampled signal has sr_out sample rate so after fft only 1/2 of the points are useful
    plots[4].plot( down )
    plots[5].plot( numpy.abs(ft_down[:sr_out//2]) )
    plt.show()

################################################################################################################
#
# A working digital downconverter!
#
# lo = desired zero frequency
# sr_in, sr_out = sample rate of the input and output signals in samples/second
# dt = delta t = duration in seconds
#
#################################################################################################################
def ddc4(lo, sr_in, sr_out, dt, *signals):
    import scipy.signal as SIGNAL
    import matplotlib.pyplot as plt

    # Get L, M
    mult  = lcm(sr_in, sr_out)
    L     = mult // sr_in
    M     = mult // sr_out
    print("DDC/sr_in={0} sr_out={1}, using sr_in * {2} / {3} = sr_out".format(sr_in, sr_out, L, M))
    # mix with cosine [should be complex?]
    mixed = mix(sinewave(lo, phase=math.pi/2), *signals)

    # we need a lowpass filter - take sr_out / 2 as Nyquist limited band
    n_tap = 128
    h = SIGNAL.firwin(n_tap, sr_out/1.0/sr_in)
    # how many seconds?
    series      = list(map(mixed, numpy.arange(0, dt, 1.0/sr_in)))
    ft_series   = numpy.fft.fft(series)
    # upsample
    up          = numpy.zeros(len(series)*L)
    up[::L]     = series

    # 'kay. Now filter that upsampled series
    series_f    = SIGNAL.lfilter(h, 1, up)
    ft_series_f = numpy.fft.fft(series_f)

    # and downsample
    down        = series_f[::M]
    ft_down     = numpy.fft.fft(down)


    # Analysis + plot part
    # figure out which bits of the FFT to use.
    # if we have more samples than the actual sample rate of the signal no point in showing > n/2 points
    n_series    = min(sr_in//2, len(series)//2)
    n_series_f  = min(len(series_f)//2, (sr_in * L)//2)
    n_down      = min(sr_out//2, int(len(up)/M/2))
    all_peaks   = list(map(peaks_peak(0.8), [ft_series[:n_series], ft_series_f[:n_series_f], ft_down[:n_down]]))
    print("peaks in FFT of mixed signal:", all_peaks[0])
    print("peaks in FFT of upsampled+filtered mixed signal:",all_peaks[1]," +LO=",all_peaks[1]/dt+lo)
    print("peaks in FFT of downsampled signal:",all_peaks[2]," +LO=",all_peaks[2]/dt+lo)

    f, plots = plt.subplots(nrows=6, ncols=1)
    plots[0].plot( series )
    plots[1].plot( numpy.abs(ft_series[:n_series]) )
    # series_f is the upsampled-by-L-and-filtered signal
    plots[2].plot( series_f )
    plots[3].plot( numpy.abs(ft_series_f[:n_series_f]) )
    # downsampled signal has sr_out sample rate so after fft only 1/2 of the points are useful
    plots[4].plot( down )
    plots[5].plot( numpy.abs(ft_down[:n_down]) )
    plt.show()

# samples are assumed to be taken at sample rate sr_in
def ddc5(lo, sr_in, sr_out, samples):
    import scipy.signal as SIGNAL
    import matplotlib.pyplot as plt

    # Get L, M
    mult  = lcm(sr_in, sr_out)
    L     = mult // sr_in
    M     = mult // sr_out
    print("DDC/sr_in={0} sr_out={1}, using sr_in * {2} / {3} = sr_out".format(sr_in, sr_out, L, M))
    # generate cosine to mix with 
    mixed = samples * numpy.cos( 2 * math.pi * lo * numpy.arange(len(samples))*1.0/sr_in )

    # we need a lowpass filter - take sr_out / 2 as Nyquist limited band
    n_tap = 128
    h = SIGNAL.firwin(n_tap, sr_out/1.0/sr_in)
    # how many seconds?
    series      = mixed #list(map(mixed, numpy.arange(0, dt, 1.0/sr_in)))
    ft_series   = numpy.fft.fft(series)
    # upsample
    up          = numpy.zeros(len(series)*L)
    up[::L]     = series

    # 'kay. Now filter that upsampled series
    series_f    = SIGNAL.lfilter(h, 1, up)
    ft_series_f = numpy.fft.fft(series_f)

    # and downsample
    down        = series_f[::M]
    ft_down     = numpy.fft.fft(down)

    # Analysis + plot part
    # figure out which bits of the FFT to use.
    # if we have more samples than the actual sample rate of the signal no point in showing > n/2 points
    dt          = float(len(series))/sr_in
    n_series    = min(sr_in//2, len(series)//2)
    n_series_f  = min(len(series_f)//2, (sr_in * L)//2)
    n_down      = min(sr_out//2, int(len(up)/M/2))
    all_peaks   = list(map(peaks_peak(0.8), [ft_series[:n_series], ft_series_f[:n_series_f], ft_down[:n_down]]))
    print("peaks in FFT of mixed signal:", all_peaks[0])
    print("peaks in FFT of upsampled+filtered mixed signal:",all_peaks[1]," +LO=",all_peaks[1]/dt+lo)
    print("peaks in FFT of downsampled signal:",all_peaks[2]," +LO=",all_peaks[2]/dt+lo)

    f, plots = plt.subplots(nrows=6, ncols=1)
    plots[0].plot( series )
    plots[1].plot( numpy.abs(ft_series[:n_series]) )
    # series_f is the upsampled-by-L-and-filtered signal
    plots[2].plot( series_f )
    plots[3].plot( numpy.abs(ft_series_f[:n_series_f]) )
    # downsampled signal has sr_out sample rate so after fft only 1/2 of the points are useful
    plots[4].plot( down )
    plots[5].plot( numpy.abs(ft_down[:n_down]) )
    #plt.show()

# DBBC giving upper + lower sidebands around lo
#  sr_in/sr_out will be made rationals/fractions
def dbbc(lo, sr_in, sr_out, samples):
    import scipy.signal as SIGNAL
    import scipy.fftpack as FFT
    import matplotlib.pyplot as plt

    # make sure sr_in/sr_out are Fractions
    sr_in, sr_out = list(map(fractions.Fraction, [sr_in, sr_out]))
    # Get L, M
    mult  = lcm_f(sr_in, sr_out)
    L     = mult / sr_in
    M     = mult / sr_out
    # Assert that both L and M are integer, not fractions.
    # prevent ludicrous ratios - if L>>10 that would mean
    # that, realistically, L and M are not very compatible
    # [800 and 64 should give 25 and 2 which is 'acceptable']
    if not (L.denominator==1 and M.denominator==1 and L<=10 and M<=200):
        raise RuntimeError(("DBBC/ sample rates sr_in={0}, sr_out={1} cannot be transformed 'simply' "+
                           "through upsample by L, downsample by M; L={2} M={3}").format(sr_in, sr_out, L, M))
    print("DBBC/sr_in={0} sr_out={1}, using sr_in * {2} / {3} = sr_out".format(sr_in, sr_out, L, M))
    print("     LO=",lo)

    # Make sure that L, M are back to normal integers
    L = L.numerator
    M = M.numerator

    # make sure samples is the real part of our complex
    D("making timeseries real ... n=",len(samples))
    samples = numpy.array(numpy.real(samples), dtype=numpy.complex64)

    # multiply by complex cosine exp( -j * 2 * pi * f * t)
    D("mixing ... samples.dtype=",samples.dtype)
    mixed  = samples * numpy.exp( -2j * numpy.pi * lo * (numpy.arange(len(samples), dtype=numpy.float64)/numpy.float64(sr_in))  )
    D("   len=",len(mixed)," dtype=",mixed.dtype)

    #down    = SIGNAL.resample_poly(mixed, L, M, window=('kaiser', 6.76))
    #coeffs  = SIGNAL.firwin(129, float(sr_out/sr_in)/2, window=userinput.window)
    #D("resampling *",L,"/",M," [window=",userinput.window,"]")
    down    = SIGNAL.resample_poly(mixed, L, M, window=('kaiser', 14))
    #down    = SIGNAL.resample_poly(mixed, L, M, window=userinput.window)
    D("   len=",len(down)," dtype=",down.dtype)

    coeffs_r= hilbert.Hilbert45(501, 0.05, 0.05, 0.45, 0)
    coeffs_i= hilbert.Hilbert45(501, 0.05, 0.05, 0.45, 1)
    re      = SIGNAL.lfilter(coeffs_r,                 1, down.real)
    im      = SIGNAL.lfilter(coeffs_i,                 1, down.imag)

    summary(down.real, "Re(down)")
    summary(down.imag, "Im(down)")
    summary(re, "Re(hilb)")
    summary(im, "Im(hilb)")

    D("making USB ...")
    usb     = re - im #re + im 
    D("making LSB ...")
    lsb     = re + im #re - im

    # About time to plot some things
    f, plots = plt.subplots(nrows=4, ncols=1)
    #f, plots = plt.subplots(nrows=3, ncols=1)
    plt.title("dbbc/resample_poly")
    D("plot samples ...")
    plots[0].plot( samples[:min(len(samples), 8000)] )
    D("plot down ... len=",len(down))
    plots[1].plot( down[:min(len(down), 8000)] )

    # The plotting part
    ftsize = int(2*(userinput.ftsize//2)) 
    nfft   = int(len(lsb)//ftsize)
    D("ftsize=",ftsize," nfft=",nfft)

    # Do the USB
    D("FFT usb/ len=",len(usb), " using ",ftsize*nfft)
    spectra = numpy.fft.fft(usb[:(ftsize * nfft)].reshape((nfft, ftsize)), axis=1)
    D("    spectra.shape=",spectra.shape)
    usb_freq = numpy.sum(numpy.abs(spectra), axis=0)[:ftsize//2]
    usb_pha  = numpy.angle(numpy.mean(spectra, axis=0)[:ftsize//2], deg=True)

    if lo<0:
        x_axis_u          = numpy.linspace(lo+(sr_out/2), lo, ftsize//2)
    else:
        x_axis_u          = numpy.linspace(lo, (lo+sr_out/2), ftsize//2)
    (u_scale, u_unit) = rescale(x_axis_u, "Hz")

    #plots[2].set_xlim(x_axis_u[0]/u_scale, x_axis_u[-1]/u_scale)
    plots[2].set_xlim( (lo - sr_out/2 - sr_out/10)/u_scale, (lo + sr_out/2 + sr_out/10)/u_scale ) 
    plots[2].plot( x_axis_u/u_scale, usb_freq, 'r', alpha=0.5 )
    plots[3].plot( x_axis_u/u_scale, usb_pha , 'r.', alpha=0.5 )
    plots[2].set_xlabel(u_unit)
    #plots[2].set_title("FFT/USB")
    plots[2].set_title("FFT")

    # Repeat for LSB
    D("FFT lsb/ len=",len(lsb), " using ",ftsize*nfft)
    spectra  = numpy.fft.fft(lsb[:(ftsize * nfft)].reshape((nfft, ftsize)), axis=1)
    lsb_freq = numpy.sum(numpy.abs(spectra), axis=0)[:ftsize//2]
    lsb_pha  = numpy.angle(numpy.mean(spectra, axis=0)[:ftsize//2], deg=True)

    if lo<0:
        x_axis_l          = numpy.linspace(lo, lo-(sr_out/2), ftsize//2)
    else:
        x_axis_l          = numpy.linspace(lo-(sr_out/2), lo, ftsize//2)
    (l_scale, l_unit) = rescale(x_axis_l, "Hz")

    #plots[3].set_xlim(x_axis_l[0]/l_scale, x_axis_l[-1]/l_scale)
    #plots[3].plot( x_axis_l/l_scale, lsb_freq, 'g', alpha=0.5 )
    plots[2].plot( x_axis_l/u_scale, lsb_freq, 'g', alpha=0.5 )
    plots[3].plot( x_axis_l/u_scale, lsb_pha , 'g.', alpha=0.5 )
    #plots[3].set_xlabel(l_unit)
    #plots[3].set_title("FFT/LSB")

    # Do loox0ring for peaks in USB/LSB
    limit = userinput.peak_threshold
    plots[2].axhline( limit*numpy.max(usb_freq) )
    for p in idx_peakert(usb_freq, limit):
        print("Found peak in USB: f={0:.8f}Hz [bin={1}]".format( x_axis_u[p], p))
        plots[2].plot( x_axis_u[p]/u_scale, usb_freq[p], marker='v', color='r')
    #plots[3].axhline( limit*numpy.max(lsb_freq) )
    for p in idx_peakert(lsb_freq, limit):
        print("Found peak in LSB: f={0:.8f}Hz [bin={1}]".format( x_axis_l[p], p))
        #plots[3].plot( x_axis_l[p]/l_scale, lsb_freq[p], marker='v', color='g')
        plots[2].plot( x_axis_l[p]/l_scale, lsb_freq[p], marker='v', color='g')

    return (lsb, usb)

def dbbc_oldstyle(lo, sr_in, sr_out, samples):
    import scipy.signal as SIGNAL
    import scipy.fftpack as FFT
    import matplotlib.pyplot as plt

    # make sure sr_in/sr_out are Fractions
    sr_in, sr_out = list(map(fractions.Fraction, [sr_in, sr_out]))
    # Get L, M
    mult  = lcm_f(sr_in, sr_out)
    L     = mult / sr_in
    M     = mult / sr_out
    # Assert that both L and M are integer, not fractions.
    # prevent ludicrous ratios - if L>>10 that would mean
    # that, realistically, L and M are not very compatible
    # [800 and 64 should give 25 and 2 which is 'acceptable']
    if not (L.denominator==1 and M.denominator==1 and L<=10 and M<=200):
        raise RuntimeError(("DBBC/ sample rates sr_in={0}, sr_out={1} cannot be transformed 'simply' "+
                           "through upsample by L, downsample by M; L={2} M={3}").format(sr_in, sr_out, L, M))
    print("DBBC/sr_in={0} sr_out={1}, using sr_in * {2} / {3} = sr_out".format(sr_in, sr_out, L, M))
    print("     LO=",lo)

    # Make sure that L, M are back to normal integers
    L = L.numerator
    M = M.numerator

    # make sure samples is the real part of our complex
    D("making timeseries real ... n=",len(samples))
    samples = numpy.array(numpy.real(samples), dtype=numpy.complex64)
    mixed   = numpy.zeros(len(samples) * L, dtype=numpy.complex128)
    # multiply by complex cosine exp( -j * 2 * pi * f * t)
    D("mixing ... samples.dtype=",samples.dtype)
    mixed[::L]  = samples * numpy.exp( -2j * numpy.pi * lo * (numpy.arange(len(samples), dtype=numpy.float64)/numpy.float64(sr_in))  )
    #mixed  = samples * numpy.exp( -2j * numpy.pi * lo * (numpy.arange(len(samples), dtype=numpy.float64)/numpy.float64(sr_in))  )
    D("   len=",len(mixed)," dtype=",mixed.dtype)

    # windowing + FIR
    # make it symmetric + odd
    #win      = SIGNAL.kaiser(51, beta=14)
    win      = 'hamming' #('kaiser', 14)
    # the filter: cutoff at sr_out/sr_in (say 8MHz out of 400MHz -> cutoff = 8/400
    #fir      = SIGNAL.firwin(501, cutoff=sr_out*L, window=win, nyq=sr_in)
    #fir      = SIGNAL.firwin(501, cutoff=float(M*sr_out)/(L*sr_in), window=win, nyq=float(L*sr_in))
    #fir      = SIGNAL.firwin(501, cutoff=float(60*sr_out)/sr_in, nyq=float(sr_in))
    #fir      = SIGNAL.firwin(501, cutoff=0.00934, nyq=1.0)
    fir      = SIGNAL.firwin(501, cutoff=float(sr_out)/float(sr_in), window='hamming', nyq=1.0)

    # mixed was already upsampled, now filter it
    #zi            = SIGNAL.lfilter_zi(fir, [1.0])
    #(filtered,zf) = SIGNAL.lfilter(fir, [1.0], mixed, zi=zi)
    filtered      = SIGNAL.lfilter(fir, [1.0], mixed)


    D("resampling *",L,"/",M," [window=",win,"]")
    down          = filtered[::M]
    D("   len=",len(down)," dtype=",down.dtype)

    #down    = SIGNAL.resample_poly(mixed, L, M, window=('kaiser', 6.76))
    #coeffs  = SIGNAL.firwin(129, float(sr_out/sr_in)/2, window=userinput.window)
    #D("resampling *",L,"/",M," [window=",userinput.window,"]")
    #down    = SIGNAL.resample_poly(mixed, L, M, window=userinput.window)
    #D("   len=",len(down)," dtype=",down.dtype)

    coeffs_r= hilbert.Hilbert45(501, 0.05, 0.05, 0.45, 0)
    coeffs_i= hilbert.Hilbert45(501, 0.05, 0.05, 0.45, 1)
    re      = SIGNAL.lfilter(coeffs_r,                 1, down.real)
    im      = SIGNAL.lfilter(coeffs_i,                 1, down.imag)

    summary(down.real, "Re(down)")
    summary(down.imag, "Im(down)")
    summary(re, "Re(hilb)")
    summary(im, "Im(hilb)")

    D("making USB ...")
    usb     = re - im #re + im 
    D("making LSB ...")
    lsb     = re + im #re - im

    # About time to plot some things
    f, plots = plt.subplots(nrows=4, ncols=1)
    #f, plots = plt.subplots(nrows=3, ncols=1)
    plt.title("dbbc/manual (i.e. [::L]=mixed, down=[::M])")
    D("plot samples ...")
    plots[0].plot( samples[:min(len(samples), 8000)] )
    D("plot down ... len=",len(down))
    plots[1].plot( down[:min(len(down), 8000)] )

    # The plotting part
    ftsize = int(2*(userinput.ftsize//2)) 
    nfft   = int(len(lsb)//ftsize)
    D("ftsize=",ftsize," nfft=",nfft)

    # Do the USB
    D("FFT usb/ len=",len(usb), " using ",ftsize*nfft)
    spectra = numpy.fft.fft(usb[:(ftsize * nfft)].reshape((nfft, ftsize)), axis=1)
    usb_freq = numpy.sum(numpy.abs(spectra), axis=0)[:ftsize//2]
    usb_pha  = numpy.angle(numpy.mean(spectra, axis=0)[:ftsize//2], deg=True)

    if lo<0:
        x_axis_u          = numpy.linspace(lo+(sr_out/2), lo, ftsize//2)
    else:
        x_axis_u          = numpy.linspace(lo, (lo+sr_out/2), ftsize//2)
    (u_scale, u_unit) = rescale(x_axis_u, "Hz")

    #plots[2].set_xlim(x_axis_u[0]/u_scale, x_axis_u[-1]/u_scale)
    plots[2].set_xlim( (lo - sr_out/2 - sr_out/10)/u_scale, (lo + sr_out/2 + sr_out/10)/u_scale ) 
    plots[2].plot( x_axis_u/u_scale, usb_freq, 'r', alpha=0.5 )
    plots[3].plot( x_axis_u/u_scale, usb_pha , 'r.', alpha=0.5 )
    plots[2].set_xlabel(u_unit)
    #plots[2].set_title("FFT/USB")
    plots[2].set_title("FFT")

    # Repeat for LSB
    D("FFT lsb/ len=",len(lsb), " using ",ftsize*nfft)
    spectra  = numpy.fft.fft(lsb[:(ftsize * nfft)].reshape((nfft, ftsize)), axis=1)
    lsb_freq = numpy.sum(numpy.abs(spectra), axis=0)[:ftsize//2]
    lsb_pha  = numpy.angle(numpy.mean(spectra, axis=0)[:ftsize//2], deg=True)

    if lo<0:
        x_axis_l          = numpy.linspace(lo, lo-(sr_out/2), ftsize//2)
    else:
        x_axis_l          = numpy.linspace(lo-(sr_out/2), lo, ftsize//2)
    (l_scale, l_unit) = rescale(x_axis_l, "Hz")

    #plots[3].set_xlim(x_axis_l[0]/l_scale, x_axis_l[-1]/l_scale)
    #plots[3].plot( x_axis_l/l_scale, lsb_freq, 'g', alpha=0.5 )
    plots[2].plot( x_axis_l/u_scale, lsb_freq, 'g', alpha=0.5 )
    plots[3].plot( x_axis_l/u_scale, lsb_pha , 'g.', alpha=0.5 )
    #plots[3].set_xlabel(l_unit)
    #plots[3].set_title("FFT/LSB")

    # Do loox0ring for peaks in USB/LSB
    limit = userinput.peak_threshold
    plots[2].axhline( limit*numpy.max(usb_freq) )
    for p in idx_peakert(usb_freq, limit):
        print("Found peak in USB: f={0:.8f}Hz [bin={1}]".format( x_axis_u[p], p))
        plots[2].plot( x_axis_u[p]/u_scale, usb_freq[p], marker='v', color='r')
    #plots[3].axhline( limit*numpy.max(lsb_freq) )
    for p in idx_peakert(lsb_freq, limit):
        print("Found peak in LSB: f={0:.8f}Hz [bin={1}]".format( x_axis_l[p], p))
        #plots[3].plot( x_axis_l[p]/l_scale, lsb_freq[p], marker='v', color='g')
        plots[2].plot( x_axis_l[p]/l_scale, lsb_freq[p], marker='v', color='g')

    return (lsb, usb)

def requantize(ds):
    ds = ((ds // (1.05*numpy.std(ds))).astype(numpy.int8).clip(-2, 1) + 2).reshape( len(ds)//4, 4 )
    ds = numpy.sum((ds * [1, 4, 16, 64]).astype(numpy.int8), axis=1).astype(numpy.int8)
    return ds

from ctypes import LittleEndianStructure, c_uint32, c_uint8

class VDIF_Header(LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        # word 0
        ("seconds", c_uint32, 30),
        ("legacy", c_uint32, 1),
        ("invalid", c_uint32, 1),

        # word 1
        ("frame_number", c_uint32, 24),
        ("epoch", c_uint32, 6),
        ("unassigned", c_uint32, 2),

        # word 2
        ("frame_length_8bytes", c_uint32, 24),
        ("log2_channels", c_uint32, 5),
        ("version", c_uint32, 3),
        
        # word 3
        ("station_id", c_uint32, 16),
        ("thread_id", c_uint32, 10),
        ("bits_per_sample_minus_1", c_uint32, 5),
        ("complex", c_uint32, 1),

        # extended header
        ("word4", c_uint32),
        ("word5", c_uint32),
        ("word6", c_uint32),
        ("word7", c_uint32)
    ]
def wr_vdif_old(samples, fn):
    # write integral multiple of 8 # of samples 
    n   = (len(samples)//8) * 8
    # simple single header: one thread w/ 1 channel @ 2bits/sample
    hdr = VDIF_Header(frame_length_8bytes=(n + 32)//8,
                      bits_per_sample_minus1 = 1,
                      log2_channels = 0)
    with open(fn, "wb") as output:
        output.write( hdr )
        output.write( requantize(samples[:n]) )

def wr_vdif(samples, fn):
    # make sure we take a number of samples that is a multiple of 4 and 8
    # (4 samples per byte (2 bits / sample), VDIF frame must be integer
    #  number of 8-byte words)
    # let's requantize 
    quantized_packed  = requantize( samples[ : (len(samples)//32)*32 ] )
    # simple single header: one thread w/ 1 channel @ 2bits/sample
    hdr = VDIF_Header(frame_length_8bytes=(len(quantized_packed) + 32)//8,
                      bits_per_sample_minus_1 = 1,
                      log2_channels = 0)
    with open(fn, "wb") as output:
        output.write( hdr )
        output.write( quantized_packed )

def blah(*args):
    # generate cosine/sine to mix with
    twopift = 2 * numpy.pi * lo * numpy.array(numpy.arange(len(samples))/sr_in, dtype=numpy.double)
    mixed_i = samples * numpy.cos( twopift )#( 2 * math.pi * lo * numpy.arange(len(samples))*1.0/sr_in )
    mixed_q = samples * numpy.sin( twopift )#( 2 * math.pi * lo * numpy.arange(len(samples))*1.0/sr_in )

    # we need a lowpass filter - take sr_out / 2 as Nyquist limited band
    n_tap = 128
    h = SIGNAL.firwin(n_tap, sr_out/1.0/sr_in)
    # how many seconds?
    ft_series_i  = numpy.fft.fft(mixed_i)
    ft_series_q  = numpy.fft.fft(mixed_q)
    # upsample
    up_i         = numpy.zeros(len(samples)*L)
    up_i[::L]    = mixed_i
    up_q         = numpy.zeros(len(samples)*L)
    up_q[::L]    = mixed_q

    # 'kay. Now filter those upsampled series
    series_f_i  = SIGNAL.lfilter(h, 1, up_i)
    series_f_q  = SIGNAL.lfilter(h, 1, up_q)

    # and downsample
    down_i      = series_f_i[::M]
    down_q      = series_f_q[::M]

#    # Analysis + plot part
#    # figure out which bits of the FFT to use.
#    # if we have more samples than the actual sample rate of the signal no point in showing > n/2 points
#    dt          = float(len(series))/sr_in
#    n_series    = min(sr_in//2, len(series)//2)
#    n_series_f  = min(len(series_f)//2, (sr_in * L)//2)
#    n_down      = min(sr_out//2, int(len(up)/M/2))
#    all_peaks   = list(map(peaks_peak(0.8), [ft_series[:n_series], ft_series_f[:n_series_f], ft_down[:n_down]]))
#    print("peaks in FFT of mixed signal:", all_peaks[0])
#    print("peaks in FFT of upsampled+filtered mixed signal:",all_peaks[1]," +LO=",all_peaks[1]/dt+lo)
#    print("peaks in FFT of downsampled signal:",all_peaks[2]," +LO=",all_peaks[2]/dt+lo)
#
#    f, plots = plt.subplots(nrows=6, ncols=1)
#    plots[0].plot( series )
#    plots[1].plot( numpy.abs(ft_series[:n_series]) )
#    # series_f is the upsampled-by-L-and-filtered signal
#    plots[2].plot( series_f )
#    plots[3].plot( numpy.abs(ft_series_f[:n_series_f]) )
#    # downsampled signal has sr_out sample rate so after fft only 1/2 of the points are useful
#    plots[4].plot( down )
#    plots[5].plot( numpy.abs(ft_down[:n_down]) )
#    #plt.show()

def ddc6(lo, sr_in, sr_out, samples):
    import scipy.signal as SIGNAL
    import matplotlib.pyplot as plt

    # Get L, M
    mult  = lcm(sr_in, sr_out)
    L     = mult // sr_in
    M     = mult // sr_out
    L = M = 2
    print("DDC/sr_in={0} sr_out={1}, using sr_in * {2} / {3} = sr_out".format(sr_in, sr_out, L, M))
    # generate cosine to mix with 
    mixed = samples * numpy.cos( math.pi * 2 * lo * numpy.arange(len(samples))*1.0/sr_in )

    # we need a lowpass filter - take sr_out / 2 as Nyquist limited band
    n_tap = 128
    #h = SIGNAL.firwin(n_tap, sr_out/1.0/sr_in)
    h = SIGNAL.firwin(n_tap, 0.1)
    # how many seconds?
    series      = mixed #list(map(mixed, numpy.arange(0, dt, 1.0/sr_in)))
    fft_len     = 2048
    nfft        = len(series)//fft_len
    ft_series   = numpy.fft.fft(series[:nfft*fft_len].reshape((nfft, fft_len)))
    # upsample
    up          = numpy.zeros(len(series)*L)
    up[::L]     = series

    # 'kay. Now filter that upsampled series
    series_f    = SIGNAL.lfilter(h, 1, up)
    ft_series_f = numpy.fft.fft(series_f[:nfft*fft_len*L].reshape((nfft, fft_len*L)))

    # and downsample
    down        = series_f[::M]
    fft_len2    = max(fft_len//M, 1024)
    n_fft2      = max(len(series)//fft_len2, 1)
    n_pts       = min(n_fft2*fft_len2, len(series))
    #ft_down     = numpy.fft.fft(down[:n_pts].reshape((n_fft2, min(fft_len2, len(series)))), axis=0)
    ft_down     = numpy.fft.fft(down)

    print("ft_series.shape=",ft_series.shape)
    print("ft_series_f.shape=",ft_series_f.shape)
    print("ft_down.shape=",ft_down.shape)
    # Analysis + plot part
    # figure out which bits of the FFT to use.
    # if we have more samples than the actual sample rate of the signal no point in showing > n/2 points
    dt          = float(len(series))/sr_in
    n_series    = min(sr_in//2, len(series)//2)
    n_series_f  = min(len(series_f)//2, (sr_in * L)//2)
    n_down      = min(sr_out//2, int(len(up)/M/2))
    all_peaks   = list(map(peaks_peak(0.8), [ft_series[:n_series], ft_series_f[:n_series_f], ft_down[:n_down]]))
    print("peaks in FFT of mixed signal:", all_peaks[0])
    print("peaks in FFT of upsampled+filtered mixed signal:",all_peaks[1]," +LO=",all_peaks[1]/dt+lo)
    print("peaks in FFT of downsampled signal:",all_peaks[2]," +LO=",all_peaks[2]/dt+lo)

    f, plots = plt.subplots(nrows=6, ncols=1)
    plots[0].plot( series )
    #plots[1].plot( numpy.abs(ft_series[:n_series]) )
    for i in range(ft_series.shape[0]):
        plots[1].plot(ft_series[i])
    # series_f is the upsampled-by-L-and-filtered signal
    plots[2].plot( series_f )
    #plots[3].plot( numpy.abs(ft_series_f[:n_series_f]) )
    for i in range(ft_series_f.shape[0]):
        plots[3].plot(ft_series_f[i])
    # downsampled signal has sr_out sample rate so after fft only 1/2 of the points are useful
    plots[4].plot( down )
    plots[5].plot( numpy.abs(ft_down[:n_down]) )

def tst1():
    import matplotlib.pyplot as plt
    sr  = 4000#2048
    z   = list(map(sines_embedded_in_noise([768.0, 128.0, 110.0], SNR=0.2), numpy.linspace(0, 1, sr)))
    #z   = list(map(sines_embedded_in_noise([63.4], SNR=1), numpy.linspace(0, 1, 2048)))
    ftz = numpy.fft.fft(z)
    f, plots = plt.subplots(nrows=2, ncols=1)
    plots[0].plot(z); plots[1].plot( numpy.abs(ftz[0:sr//2]) ); plt.show()



########################################################################################################
##
##
##    Now we want to generate a time series from spectra; synthesizerts!
##
##
########################################################################################################


def gaussian_noise_s(npoint, amplitude=1.0, mean=0.0):
    def gen(ts):
        return amplitude * numpy.random.randn(npoint) + mean
    return gen

def tones_s(npoint, tones):
    try:
        for toneidx in tones:
            assert toneidx>=0 and toneidx<npoint
    except TypeError:
        assert tones>=0 and tones<npoint
    template = numpy.zeros(npoint)
    template[tones] = 1.0
    def gen(ts):
        return template
    return gen

def top_hat_s(npoint, lo, hi):
    return tones_s(npoint, range(lo, hi+1))

def add_s(*spectra):
    sampler = sample(*spectra)
    def gen(ts):
        return functools.reduce(operator.add, sampler(ts))
    return gen

# start with something that generates spectra
def noisy_spectrum_w_tone(npoint, tones, snr=10):
    return add_s(gaussian_noise_s(npoint, amplitude=1.0/snr), tones_s(npoint, tones))
    #def gen(idx):
    #    # at this point we don't care about the index
    #    noisy_spec[tones] = 1.0
    #    return noisy_spec
    #return gen

def noisy_gaussian(npoint, tones, snr=10):
    try:
        for toneidx in tones:
            assert toneidx>=0 and toneidx<npoint
    except TypeError:
        assert tones>=0 and tones<npoint
    def gen(idx):
        # at this point we don't care about the index
        noisy_spec = numpy.random.randn(npoint) / snr
        noisy_spec[tones] = 1.0
        return noisy_spec
    return gen


def synthesizert(bw, spectrum_src):
    import matplotlib.pyplot as plt

    f, plots = plt.subplots(nrows=7, ncols=1)
    spectra = list(map(spectrum_src, range(10)))
    timeseries  = numpy.ndarray(0)
    timeseries2 = numpy.ndarray(0)
    for s in spectra:
        plots[0].plot( numpy.abs(s)+numpy.max(numpy.abs(s)) )
        tmp = numpy.zeros( len(s)*2, dtype=numpy.complex64 )
        tmp2 = numpy.zeros( len(s)*2, dtype=numpy.complex64 )
        # try upper side band
        tmp2[len(s):] = s
        tmp[0:len(s)] = numpy.conj(s[::-1])
        timeseries = numpy.hstack([timeseries, numpy.real(numpy.fft.ifft(tmp))])
        timeseries2 = numpy.hstack([timeseries2, numpy.real(numpy.fft.ifft(tmp2))])
        #timeseries = numpy.hstack([timeseries, numpy.real(numpy.fft.ifft(s))])
        #timeseries = numpy.hstack([timeseries, numpy.abs(numpy.fft.ifft(s))])
    print("timeseries length = ",len(timeseries), " dtype = ",timeseries.dtype)
    print("timeseries2 length = ",len(timeseries2), " dtype = ",timeseries2.dtype)
    plots[1].plot(timeseries)
    plots[2].plot(timeseries2)
    plots[3].plot(timeseries-timeseries2)
    ft1 = numpy.abs(numpy.fft.fft(timeseries[:len(timeseries)//2]))
    ft2 = numpy.abs(numpy.fft.fft(timeseries2[:len(timeseries2)//2]))
    plots[4].plot(ft1)
    plots[5].plot(ft2)
    plots[6].plot(ft1-ft2)
    plt.show()


def sinc(n):
    tmp = numpy.linspace(-math.pi, math.pi, n) * math.pi # pi * x
    tmp = numpy.sin(tmp) / tmp 
    tmp[ numpy.isnan(tmp) ] = 1.0 #tmp[numpy.where(tmp == numpy.NaN)] = 1.0
    return tmp


# return list of index-lists
# the last element in the list is where to put the current spectrum,
# the whole list should be used to index into the weight array
def gen_idx_seq(N):
    states = [list(range(N))]
    while len(states)<N:
        rng = states[-1]
        states.append( rng[1:]+[rng[0]] )
    return states


# expect spectra with length npoint, use PFB depth N
# currently use sinc window
def pfb_synth(bw, npoint, N, spectrum_src):
    import matplotlib.pyplot as plt

    f, plots = plt.subplots(nrows=3, ncols=1)

    # the pfb is N stages deep
    pfb_weights = sinc(2 * npoint * N).astype(numpy.complex64).reshape((N, 2*npoint))
    # upsampled spectra
    spectra_buf = numpy.zeros(2 * npoint * N, dtype=numpy.complex64).reshape((N, 2*npoint))
    # we cycle through these states
    states      = itertools.cycle( gen_idx_seq(N) )
    spectra     = list(map(spectrum_src, range(10)))
    timeseries  = numpy.ndarray(0)
    for s in spectra:
        plots[0].plot( numpy.abs(s)+numpy.max(numpy.abs(s)) )
        curstate = next(states)
        print("Current state = ",curstate)
        # put next spectrum at last index
        # in correct half (upper or lower side band)
        tmp = numpy.zeros( 2*npoint, dtype=numpy.complex64)
        tmp[0] = s[0]
        tmp[npoint+1:] = s[1:]
        #tmp[len(s):] = numpy.conj(s[::-1])
        # perform ifft of new spectrum and put into spectrum buffer next position
        spectra_buf[ curstate[-1] ] = numpy.fft.ifft( tmp )
        # now multiply with weights and sum over axis 1
        timeseries = numpy.hstack( [timeseries, numpy.real((spectra_buf[ curstate ] * pfb_weights).sum( axis=0 ))] )
    print("timeseries length = ",len(timeseries), " dtype = ",timeseries.dtype)
    plots[1].plot(timeseries)
    ft1 = numpy.abs(numpy.fft.fft(timeseries[:len(timeseries)//2]))
    plots[2].plot(ft1)
    #plt.show()

def synthesizer2(bw, spectrum_src):
    import matplotlib.pyplot as plt

    f, plots = plt.subplots(nrows=3, ncols=1)
    spectra = list(map(spectrum_src, range(10)))
    timeseries  = numpy.ndarray(0)
    for s in spectra:
        plots[0].plot( numpy.abs(s)+numpy.max(numpy.abs(s)) )
        # assume this is lsb with freq increasing to the right
        tmp = numpy.zeros( len(s)*2, dtype=numpy.complex64 )
        tmp[0] = s[0]
        tmp[len(s)+1:] = s[1:]
        #tmp[0:len(s)] = numpy.conj(s[::-1])
        timeseries = numpy.hstack([timeseries, numpy.real(numpy.fft.ifft(tmp))])
        #timeseries = numpy.hstack([timeseries, numpy.real(numpy.fft.ifft(s))])
        #timeseries = numpy.hstack([timeseries, numpy.abs(numpy.fft.ifft(s))])
    print("timeseries length = ",len(timeseries), " dtype = ",timeseries.dtype)
    plots[1].plot(timeseries)
    ft1 = numpy.abs(numpy.fft.fft(timeseries[:len(timeseries)//2]))
    plots[2].plot(ft1)
    #plt.show()

###############################
#
# now with generators
#
###############################
def gaussian_noise_g(npoint, amplitude=1.0, mean=0.0):
    while True:
        yield amplitude * numpy.random.randn(npoint) + mean

def tones_g(npoint, tones, amplitude=1.0, tp=numpy.float64):
    try:
        for toneidx in tones:
            assert toneidx>=0 and toneidx<npoint
    except TypeError:
        assert tones>=0 and tones<npoint
    template = numpy.zeros(npoint, dtype=tp)
    template[tones] = tp(amplitude)
    while True:
        yield template

def noise_g_heap(npoint, nspec_p_heap):
    noise_gen = gaussian_noise_g(npoint * nspec_p_heap)
    while True:
        block = next(noise_gen)
        yield block.reshape((nspec_p_heap, npoint))

def tones_g_heap(npoint, tones, dc_freq=None, bandwidth=None, dc_idx=0, amplitude=1.0, tp=numpy.float64, nspec_p_heap=128):
    t_idx = []
    t_f   = []
    t_amp = []
    try:
        for tone in tones:
            if isinstance(tone, tuple):
                (tone_f, tone_a) = tone
            else:
                (tone_f, tone_a) = (tone, amplitude)
            toneidx = int((float(tone_f - dc_freq) / bandwidth) * npoint)
            assert toneidx>=0 and toneidx<npoint
            t_f.append( ((float(toneidx)/npoint) * bandwidth)  )
            t_idx.append( toneidx )
            t_amp.append( tp(tone_a)  )
    except TypeError:
        toneidx = int((float(tones - dc_freq) / bandwidth) * npoint)
        assert toneidx>=0 and toneidx<npoint
        t_f.append( ((float(toneidx)/npoint) * bandwidth)  )
        t_idx.append( toneidx )
        t_amp.append( tp(amplitude) )
    template = numpy.zeros((nspec_p_heap, npoint), dtype=tp)
    template[:,t_idx] = t_amp #tp(amplitude)
    print("Tones @",t_f, " AMP=",t_amp)
    while True:
        yield Spectrum(template, DCEdge=0, DCFrequency=dc_freq, BW=bandwidth)


def top_hat_g(npoint, lo, hi):
    return tones_g(npoint, range(lo, hi+1))

def add_g(*spectra):
    while True:
        yield functools.reduce(operator.add, map(next, spectra))

def synthesizer_g(bw, spectrum_src):
    import matplotlib.pyplot as plt

    f, plots = plt.subplots(nrows=3, ncols=1)
    #spectra = list(take(10, spectrum_src))
    timeseries  = numpy.ndarray(0)
    for s in take(10, spectrum_src): #spectra:
        plots[0].plot( numpy.abs(s)+numpy.max(numpy.abs(s)) )
        # assume this is lsb with freq increasing to the right
        tmp = numpy.zeros( len(s)*2, dtype=numpy.complex64 )
        tmp[0] = s[0]
        tmp[len(s)+1:] = (s[1:])[::-1]
        #tmp[0:len(s)] = numpy.conj(s[::-1])
        timeseries = numpy.hstack([timeseries, numpy.real(numpy.fft.ifft(tmp))])
        #timeseries = numpy.hstack([timeseries, numpy.real(numpy.fft.ifft(s))])
        #timeseries = numpy.hstack([timeseries, numpy.abs(numpy.fft.ifft(s))])
    print("timeseries length = ",len(timeseries), " dtype = ",timeseries.dtype)
    plots[1].plot(timeseries)
    ft1 = numpy.abs(numpy.fft.fft(timeseries[:len(timeseries)//2]))
    plots[2].plot(ft1)
    #plt.show()
    return timeseries

def synthesizer_g_cplx(bw, spectrum_src):
    import matplotlib.pyplot as plt
    
    # re, im, ifft_re, ifft_im, timeseries, ft_re, ft_im
    # 0   1   2        3        4           5      6
    f, plots    = plt.subplots(nrows=7, ncols=1)
    timeseries  = numpy.ndarray(0)
    ssize       = -1
    for s in take(10, spectrum_src): #spectra:
        if len(s)>ssize:
            ssize = len(s)
        #plots[0].plot( numpy.abs(s)+numpy.max(numpy.abs(s)) )
        plots[0].plot( numpy.real(s) )
        plots[1].plot( numpy.imag(s) )
        # complex source for ifft does not have to be symmetric 
        tmp = numpy.fft.ifft( s )
        plots[2].plot( numpy.real(tmp) )
        plots[3].plot( numpy.imag(tmp) )
        timeseries = numpy.hstack([timeseries, tmp])
    print("timeseries length = ",len(timeseries), " dtype = ",timeseries.dtype)
    plots[4].plot(numpy.real(timeseries) )
    for i in numpy.arange(0, len(timeseries), ssize):
        ft = numpy.fft.fft(timeseries[i:i+ssize])
        plots[5].plot( numpy.real(ft) )
        plots[6].plot( numpy.imag(ft) )
    return numpy.real(timeseries)

# transform any (SSB) spectrum into a real-valued time signal
# by making it symmetric
def synthesizer_g_real(bw, spectrum_src, **opts):
    import matplotlib.pyplot as plt

    # we must know if the spectrum is LSB or USB
    # assume that in both cases freq increases with index
    defaults = { 'sideband': None } # -1 == lower, +1 == upper
    defaults.update( opts ) 
    sb = defaults['sideband']
    if sb not in [1,-1]:
        raise RuntimeError("sideband parameter " + "not set" if sb is None else "can only be -1 or +1")
    # abs, ifft_re, ifft_im, timeseries, ft_re, ft_im
    # 0    1        2        3           4      5
    # 0                      1           2
    #f, plots    = plt.subplots(nrows=6, ncols=1)
    f, plots    = plt.subplots(nrows=3, ncols=1)
    x_freq      = numpy.linspace(0, bw, 1024)[::sb]
    x_scale,un  = rescale(x_freq, "Hz")
    timeseries  = numpy.ndarray(0)
    #for s in take(25, spectrum_src): #spectra:
    for s in take(userinput.nspec, spectrum_src): #spectra:
        #plots[0].plot( x_freq, numpy.real(s) )
        #plots[0].plot( x_freq, numpy.imag(s) )
        plots[0].plot( x_freq/x_scale, numpy.abs(s) )
        # source for ifft will be made symmetric
        tmp = numpy.zeros( 2*len(s), dtype=numpy.complex128 )
        # if LSB convert to USB
        if sb==-1:
            s = s * ([1,-1]*(len(s)//2))
        # USB: freq DC => DC+BW-Ch/2
        tmp[0:len(s)]  = s
        tmp[len(s)+1:] = numpy.conj( (s[1::])[::-1] )

        ftmp = numpy.fft.ifft( tmp )
        #plots[1].plot( numpy.real(ftmp) )
        #plots[2].plot( numpy.imag(ftmp) )
        timeseries = numpy.hstack([timeseries, ftmp])
    if userinput.lo<0:
        plots[0].axvspan(userinput.lo/x_scale, (userinput.lo-userinput.bw)/x_scale, alpha=0.5, facecolor='r')
        plots[0].axvspan((userinput.lo+userinput.bw)/x_scale, userinput.lo/x_scale, alpha=0.5, facecolor='g')
    else:
        plots[0].axvspan(userinput.lo/x_scale, (userinput.lo+userinput.bw)/x_scale, alpha=0.5, facecolor='r')
        plots[0].axvspan((userinput.lo-userinput.bw)/x_scale, userinput.lo/x_scale, alpha=0.5, facecolor='g')
    print("timeseries length = ",len(timeseries), " dtype = ",timeseries.dtype)
    plots[1].plot(numpy.real(timeseries) )
    summary(numpy.real(timeseries), "Re(timeseries)")
    summary(numpy.imag(timeseries), "Im(timeseries)")
    stack = None
    ssize = int(2*(userinput.ftsize//2))
    for i in numpy.arange(0, len(timeseries), ssize):
        ft = numpy.fft.fft( numpy.real(timeseries[i:i+ssize]) )
        stack = numpy.vstack([stack, ft]) if stack is not None else ft
        #plots[4].plot( numpy.real(ft) )
        #plots[5].plot( numpy.imag(ft) )
    x              = numpy.linspace(0, bw, ssize/2)[::sb]
    (x_scal, unit) = rescale(x, "Hz")
    plots[2].plot(x/x_scal , numpy.sum(numpy.abs(stack), axis=0)[:ssize//2] )
    plots[2].set_xlabel(unit)
    return numpy.real(timeseries)

def synthesizer_g_real_2(bw, spectrum_src, **opts):
    import matplotlib.pyplot as plt

    # we must know if the spectrum is LSB or USB
    # assume that in both cases freq increases with index
    defaults = { 'sideband': None } # -1 == lower, +1 == upper
    defaults.update( opts ) 
    sb = defaults['sideband']
    if sb not in [1,-1]:
        raise RuntimeError("sideband parameter " + "not set" if sb is None else "can only be -1 or +1")
    # abs, ifft_re, ifft_im, timeseries, ft_re, ft_im
    # 0    1        2        3           4      5
    # 0                      1           2
    #f, plots    = plt.subplots(nrows=6, ncols=1)
    f, plots    = plt.subplots(nrows=3, ncols=1)
    x_freq      = numpy.linspace(0, bw, 1024)[::sb]
    x_scale,un  = rescale(x_freq, "Hz")
    timeseries  = numpy.ndarray(0)
    #for s in take(25, spectrum_src): #spectra:
    for s in take(userinput.nspec, spectrum_src): #spectra:
        plots[0].plot( x_freq/x_scale, numpy.abs(s) )

        # spectrum is symmetric - that is, we NEED it to be in order to get a real timeseries
        ftmp = numpy.fft.irfft(s, 2*len(s))
        timeseries = numpy.hstack([timeseries, ftmp])
    plots[0].axvspan(userinput.lo/x_scale, (userinput.lo+userinput.bw)/x_scale, alpha=0.5, facecolor='r')
    plots[0].axvspan((userinput.lo-userinput.bw)/x_scale, userinput.lo/x_scale, alpha=0.5, facecolor='g')
    print("timeseries length = ",len(timeseries), " dtype = ",timeseries.dtype)
    plots[1].plot(numpy.real(timeseries) )
    summary(numpy.real(timeseries), "Re(timeseries)")
    summary(numpy.imag(timeseries), "Im(timeseries)")
    stack = None
    ssize = int(2*(userinput.ftsize//2))
    #ftsize= 
    #for i in numpy.arange(0, len(timeseries), ssize):
    for i in numpy.arange(0, ssize * int(len(timeseries)//ssize), ssize):
        ft = numpy.fft.fft( numpy.real(timeseries[i:i+ssize]) )
        stack = numpy.vstack([stack, ft]) if stack is not None else ft
    x              = numpy.linspace(0, bw, ssize/2)[::sb]
    (x_scal, unit) = rescale(x, "Hz")
    plots[2].plot(x/x_scal , numpy.sum(numpy.abs(stack), axis=0)[:ssize//2] )
    plots[2].set_xlabel(unit)
    return numpy.real(timeseries)

def synthesizer_g_real_2_heap(bw, spectrum_src, **opts):
    import matplotlib.pyplot as plt

    # abs, ifft_re, ifft_im, timeseries, ft_re, ft_im
    # 0    1        2        3           4      5
    # 0                      1           2
    f, plots    = plt.subplots(nrows=3, ncols=1)
    x_freq      = None         #numpy.linspace(0, bw, 4096)
    x_scale,un  = (None, None) #rescale(x_freq, "Hz")
    timeseries  = numpy.ndarray(0)
    for s in take(userinput.nspec, spectrum_src):
        print("synthesizer_g_real_2_heap/taking in spectra of shape ",s.shape)
        if x_freq is None:
            x_freq      = numpy.linspace(0, s.BW, s.shape[1])
            x_scale,un  = rescale(x_freq, "Hz")
            plots[0].set_xlim(x_freq[0]/x_scale, x_freq[-1]/x_scale)
        plots[0].plot( x_freq/x_scale, numpy.sum(numpy.abs(s), axis=0) )

        # spectrum is symmetric - that is, we NEED it to be in order to get a real timeseries
        ftmp = numpy.fft.irfft(s, 2*s.shape[1], axis=1)
        timeseries = numpy.hstack([timeseries, ftmp.flatten()])
    if userinput.lo<0:
        plots[0].axvspan(userinput.lo/x_scale, (userinput.lo-userinput.bw)/x_scale, alpha=0.5, facecolor='g')
        plots[0].axvspan((userinput.lo+userinput.bw)/x_scale, userinput.lo/x_scale, alpha=0.5, facecolor='r')
    else:
        plots[0].axvspan(userinput.lo/x_scale, (userinput.lo+userinput.bw)/x_scale, alpha=0.5, facecolor='r')
        plots[0].axvspan((userinput.lo-userinput.bw)/x_scale, userinput.lo/x_scale, alpha=0.5, facecolor='g')
    print("timeseries length = ",len(timeseries), " dtype = ",timeseries.dtype)
    plots[1].plot(numpy.real(timeseries)[:min(len(timeseries), 8000)] )
    summary(numpy.real(timeseries), "Re(timeseries)")
    summary(numpy.imag(timeseries), "Im(timeseries)")

    ssize   = int(2*(userinput.ftsize//2))
    nfft    = int(len(timeseries)//ssize)

    spectra = numpy.fft.fft(timeseries[:(ssize * nfft)].reshape((nfft, ssize)), axis=1)
    x       = numpy.linspace(0, bw, ssize/2)
    (x_scal, unit) = rescale(x, "Hz")
    plots[2].set_xlim(x[0]/x_scal, x[-1]/x_scal)
    plots[2].plot(x/x_scal, numpy.sum(numpy.abs(spectra), axis=0)[:ssize//2])
    plots[2].set_xlabel(unit)
    return numpy.real(timeseries)

# expect spectra with length npoint, use PFB depth N
# currently use sinc window
def pfb_synth_g(bw, npoint, N, spectrum_src):
    import matplotlib.pyplot as plt
    import scipy.signal as SIGNAL

    f, plots = plt.subplots(nrows=4, ncols=1)

    # the pfb is N stages deep
    #pfb_weights = numpy.sinc( numpy.linspace(-math.pi, math.pi, 2*npoint*N).astype(numpy.complex64).reshape((N, 2*npoint)) )
    pfb_weights = (SIGNAL.firwin(npoint * 2 * N, 1. / (npoint * 2)) * npoint ).astype(numpy.float32).reshape(N,2*npoint)
    #w = numpy.sinc( numpy.linspace(-math.pi, math.pi, 2*npoint*N).astype(numpy.complex64).reshape((N, 2*npoint)) )
    #plots[3].plot( pfb_weights.flatten()/w.flatten() )
    #sinc(2 * npoint * N).astype(numpy.complex64).reshape((N, 2*npoint))
    # upsampled spectra
    spectra_buf = numpy.zeros(2 * npoint * N, dtype=numpy.complex64).reshape((N, 2*npoint))
    # we cycle through these states
    states      = itertools.cycle( gen_idx_seq(N) )
    timeseries  = numpy.ndarray(0)
    for s in take(10, spectrum_src): #spectra:
        plots[0].plot( numpy.abs(s)+numpy.max(numpy.abs(s)) )
        plots[3].plot( numpy.real(s)+numpy.max(numpy.abs(s)) )
        curstate = next(states)
        print("Current state = ",curstate)
        # put next spectrum at last index
        # in correct half (upper or lower side band)
        tmp = numpy.zeros( 2*npoint, dtype=numpy.complex64)
        tmp[0] = s[0]
        tmp[npoint+1:] = (s[1:])[::-1]
        #tmp[len(s):] = numpy.conj(s[::-1])
        # perform ifft of new spectrum and put into spectrum buffer next position
        spectra_buf[ curstate[-1] ] = numpy.fft.ifft( tmp )
        # now multiply with weights and sum over axis 1
        timeseries = numpy.hstack( [timeseries, numpy.real((spectra_buf[ curstate ] * pfb_weights).sum( axis=0 ))] )
    print("timeseries length = ",len(timeseries), " dtype = ",timeseries.dtype)
    plots[1].plot(timeseries)
    ft1 = numpy.abs(numpy.fft.fft(timeseries[:len(timeseries)//2]))
    print("ft1 dtype=",ft1.dtype)
    plots[2].plot(ft1)
    #plt.show()
    return timeseries

