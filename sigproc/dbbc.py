from __future__ import print_function
import scipy.signal as SIGNAL
import scipy.fftpack as FFT
import numpy, fractions, sigproc.util, sigproc.hilbert
#from numba import jit

Hilbert45 = sigproc.hilbert.Hilbert45
#D         = print if __debug__ else lambda *args: None 
D         = lambda *args: None 
#D = print

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


# DBBC giving upper + lower sidebands around lo
#  sr_in/sr_out will be made rationals/fractions
def dbbc(lo, sr_in, sr_out, window=None):
    # make sure sr_in/sr_out are Fractions
    sr_in, sr_out = list(map(fractions.Fraction, [sr_in, sr_out]))
    # Get L, M
    mult  = sigproc.util.lcm_f(sr_in, sr_out)
    L     = mult / sr_in
    M     = mult / sr_out
    # Assert that both L and M are integer, not fractions.
    # prevent ludicrous ratios - if L>>10 that would mean
    # that, realistically, L and M are not very compatible
    # [800 and 64 should give 25 and 2 which is 'acceptable']
    if not (L.denominator==1 and M.denominator==1 and L<=10 and M<=300):
        raise RuntimeError(("DBBC/ sample rates sr_in={0}, sr_out={1} cannot be transformed 'simply' "+
                           "through upsample by L, downsample by M; L={2} M={3}").format(sr_in, sr_out, L, M))
    sf = sigproc.util.rescale([lo, 0] if lo < 0 else [0,lo] , "Hz")
    print("DBBC/lo={4}{5} sr_in={0} sr_out={1}, using sr_in * {2} / {3} = sr_out".format(sr_in, sr_out, L, M, lo/sf[0], sf[1]))

    # Make sure that L, M are back to normal integers
    L = L.numerator
    M = M.numerator

   
    # Done preprocessing - now set up state machine
    nCoeff       = 301
    coeffs_r     = Hilbert45(nCoeff, 0.05, 0.05, 0.45, 0)
    coeffs_i     = Hilbert45(nCoeff, 0.05, 0.05, 0.45, 1)
    window       = 'hamming' if window is None else window
    # after down sampling of the upsampled signal we want at least nCoeff samples
    limit        = nCoeff * M #max(L, M) // sigproc.util.gcd(L, M)
    print( "L={0} M={1} LIMIT={2}".format(L, M, limit))

    # modifyable state
    class State(object):
        nSamples     = 0   # keep track of phase within second; 0 <= nSamples < sr_in
        prev_samples = numpy.ndarray(0)
        wrVDIF       = None

    def do_it(samples):
        # make sure samples is the real part of our complex
        D("making timeseries real ... n=",len(samples))
        samples  = numpy.array(numpy.real(samples), dtype=numpy.complex64)
        nSamples = len(samples)

        # multiply by complex cosine exp( -j * 2 * pi * f * (t + t0))
        frac = float(State.nSamples)/float(sr_in)
        D("mixing ... phase = ", frac , " turns = ", frac * 2 * numpy.pi, " [State.nSamples=",State.nSamples,"]")
        mixed  = samples * numpy.exp( -2j * numpy.pi * lo * ((numpy.arange(nSamples, dtype=numpy.float64)+State.nSamples)/numpy.float64(sr_in))  )
        # keep remainder of samples withing sample rate [basically the phase offset for when the next chunk of time samples get in]
        State.nSamples = int( (State.nSamples + nSamples) % sr_in )
        D("   len=",len(mixed)," dtype=",mixed.dtype)

        # append to previous samples
        State.prev_samples = numpy.hstack([State.prev_samples, mixed])

        # only if we have enough time samples for filtering?
        nFilter,nRemain = divmod(len(State.prev_samples), limit)
        nFilter         = nFilter * limit
        D("   nFilter,nRemain=", nFilter, nRemain) 
        if not nFilter:
            return None 

        # split into two arrays: integer number of limit + remainder
        mixed, State.prev_samples  = numpy.split(State.prev_samples, [nFilter])

        #down    = SIGNAL.resample_poly(mixed, L, M, window=('kaiser', 6.76))
        #coeffs  = SIGNAL.firwin(129, float(sr_out/sr_in)/2, window=userinput.window)
        down    = SIGNAL.resample_poly(mixed, L, M, window=window) #window=userinput.window)
        re      = SIGNAL.lfilter(coeffs_r,                 1, down.real)
        im      = SIGNAL.lfilter(coeffs_i,                 1, down.imag)
        # Now we have USB = re + im, LSB = re - im [apparently other way around????]
        lsb     = re + im
        usb     = re - im
#        if State.wrVDIF is None:
#            print("Writing VDIF")
#            wr_vdif(lsb, "/tmp/lsb.vdif")
#            wr_vdif(usb, "/tmp/usb.vdif")
#            State.wrVDIF = True
        return (lsb, usb)
    return do_it

# DBBC giving upper + lower sidebands around lo
#  sr_in/sr_out will be made rationals/fractions
def dbbc_old(lo, sr_in, sr_out, samples):
    # make sure sr_in/sr_out are Fractions
    sr_in, sr_out = list(map(fractions.Fraction, [sr_in, sr_out]))
    # Get L, M
    mult  = sigproc.util.lcm_f(sr_in, sr_out)
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

    # Make sure that L, M are back to normal integers
    L = L.numerator
    M = M.numerator

    # make sure samples is the real part of our complex
    D("making timeseries real ... n=",len(samples))
    samples = numpy.array(numpy.real(samples), dtype=numpy.complex64)

    # multiply by complex cosine exp( -j * 2 * pi * f * t)
    D("mixing ...")
    mixed   = samples * numpy.exp( -2j * numpy.pi * lo * numpy.array(numpy.arange(len(samples))/sr_in, dtype=numpy.float) )
    D("   len=",len(mixed)," dtype=",mixed.dtype)

    #down    = SIGNAL.resample_poly(mixed, L, M, window=('kaiser', 6.76))
    #coeffs  = SIGNAL.firwin(129, float(sr_out/sr_in)/2, window=userinput.window)
    down    = SIGNAL.resample_poly(mixed, L, M, window='hamming') #window=userinput.window)
#    # upsample
#    D("upsampling ...")
#    up      = numpy.zeros( len(mixed) * L, dtype=numpy.complex128 )
#    up[::L] = mixed
#    D("   len=",len(up)," dtype=",up.dtype)
#
#    # Pass the upsampled signal through a LPF, then downsample
#    D("lfiltering ...")
#    #up      = SIGNAL.lfilter(SIGNAL.firwin(129, float(sr_out/2)/sr_in), 1, up)
#    #coeffs  = SIGNAL.firwin(128, float((sr_out*L)/(sr_in*M)), window=userinput.window)
#    coeffs  = SIGNAL.firwin(128, float((sr_out*L)/(sr_in*M)), window=userinput.window)
#    #coeffs  = SIGNAL.firwin(128, 0.5)
#    #D("coeffs: ",coeffs)
#    up      = SIGNAL.lfilter(coeffs, 1, up)
#    D("   len=",len(up)," dtype=",up.dtype)
    D("downsampling ...")
#    down    = up[::M]
    D("   len=",len(down)," dtype=",down.dtype)

    coeffs_r= hilbert.Hilbert45(301, 0.05, 0.05, 0.45, 0)
    coeffs_i= hilbert.Hilbert45(301, 0.05, 0.05, 0.45, 1)
    re      = SIGNAL.lfilter(coeffs_r,                 1, down.real)
    im      = SIGNAL.lfilter(coeffs_i,                 1, down.imag)
    #re      = SIGNAL.lfilter(coeffs,                 1, down.real)
    #im      = SIGNAL.lfilter(list(reversed(coeffs)), 1, down.imag)
    # Sweet. Extract re,im and do Hilbert on the im
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
#    re      = down.real
#    D("Hilberting ...")
#    coeffs  = numpy.fft.fftshift(numpy.fft.ifft([0]+[1]*100+[0]*100))
#    hilb    = SIGNAL.hilbert(down.imag)
    #hilb    = FFT.hilbert(down.imag)
#   #hilb    = SIGNAL.lfilter(coeffs, 1, down.imag)
#    im      = numpy.imag(hilb)
#    im_org  = numpy.real(hilb)
#    D("len(re) = ",len(re)," shape=",re.shape," dtype=",re.dtype)
#    D("len(im) = ",len(im)," shape=",im.shape," dtype=",im.dtype)
#    summary(down.imag, "Im(down)")
#    summary(im_org, "Re(hilb)")
#    summary(down.real, "Re(down)")
#    summary(down.imag, "Im(down)")
#    summary(re, "Re(hilb)")
#    summary(im, "Im(hilb)")

    # Now we have USB = re + im, LSB = re - im [apparently other way around????]
    D("making LSB ...")
    lsb     = re + im #down.real #re + im
    D("making USB ...")
    usb     = re - im #down.imag #re - im
    return (lst, usb)
