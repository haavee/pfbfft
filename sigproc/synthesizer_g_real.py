import numpy, functional_hv, math, sigproc.types

# assume mods = (idx, value), .... 
def replacer(*mods):
    (indices, values) = zip(*mods)
    indices           = list(indices)
    values            = numpy.array( list(values) )
    def do_it(spectrum):
        spectrum[ indices ] = values
        return spectrum
    return do_it

def auto_comb(n):
    def do_it(spectrum):
        # generate spikes
        spike_idx = numpy.linspace(len(spectrum)/n, len(spectrum), n, endpoint=False, dtype=numpy.uint32)
        m         = numpy.max(spectrum)
        spectrum[ spike_idx ] = m*(1.5 +  numpy.sin( numpy.linspace(0, math.pi, len(spike_idx)) ))
        return spectrum
    return do_it

def synthesizer_real(*ppfuncs):
    pp_func = functional_hv.compose(*ppfuncs)
    def do_it(spectrum):
        # make sure we have a normalized and pre-processed spectrum
        spectral_data = pp_func(spectrum.normalize())
        # reserve twice the number of points
        spec_len      = len(spectral_data)
        time_series   = numpy.zeros( 2*spec_len, dtype=numpy.complex128 )
        # normalized spectrum so f increases to the right; make symmetric
        # as per numpy.fft.fft specifications [how fft outputs pos/neg frequencies]
        time_series[ 0:spec_len  ] = spectral_data
        time_series[ spec_len+1: ] = numpy.conj( (spectral_data[1::])[::-1] )
        time_series               = numpy.fft.ifft( time_series )
        return sigproc.types.TimeSeries(numpy.real(time_series), TimeStamp=spectrum.TimeStamp, SampleRate=2*spectrum.BandWidth)
    return do_it

