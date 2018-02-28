import numpy, functional_hv

def replacer_heap(*mods):
    (indices, values) = zip(*mods)
    indices           = list(indices)
    values            = numpy.array( list(values) )
    def do_it(spectra): 
        spectra[:,indices] = values
        yield spectra
    return do_it

def synthesizer_g_real_irfft_heap(*ppfuncs):
    pp_func = functional_hv.compose(*ppfuncs)

    def do_it(spectra):
        # spectrum is symmetric - that is, we NEED it to be in order to get a real timeseries
        # hence the irfft
        return numpy.real( numpy.fft.irfft(spectra, 2*spectra.shape[1], axis=1).flatten() )
    return do_it
