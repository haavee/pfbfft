#!/usr/bin/env python
# this HAS to happen first
from __future__ import print_function
# and next we MUST do this before we include anything else to make sure all time
# handling is done in GMT
import os, time
os.environ['TZ'] = ''
time.tzset()
# Right. NOW we can move on with our lives
import argparse, math, calendar, fractions, collections, re, os, sys, numpy
from sigproc          import types
from functional_hv    import *
from beam_red_meerkat import Observation, ReadBeamformHeap


def integrate_spectra(src):
    total  = 0
    n_spec = 0
    for s in src:
        # s.shape == (nspec, nchan)
        n_spec += s.shape[0]
        total   = numpy.add(total, types.Spectrum(numpy.sum(numpy.abs(s), axis=0), DCEdge=0))
    total.n_spec = n_spec
    return total


def plot(tuplst):
    # [(filename, spectrum, Observation), ...]
    import matplotlib.pyplot as plt
    nPlot = len(tuplst)
    tInt  = None
    for (i, spec) in enumerate(tuplst):
        (fn, spectrum, obs) = spec
        xaxis = numpy.linspace(obs.dc_freq, obs.dc_freq+obs.bandwidth, len(spectrum))/1e6
        plt.subplot(nPlot, 1, i+1)
        plt.plot(xaxis[1:], spectrum[1:])
        plt.title(os.path.basename(fn))
        if tInt is None:
            tInt = float(spectrum.n_spec / obs.spectra_p_s)
        if tInt != float(spectrum.n_spec / obs.spectra_p_s):
            raise RuntimeError("Different amount of spectra averaged")
    if tInt is not None:
        plt.suptitle("Integration time = {0:4f}s".format(tInt))
    plt.show()
    return ()
    

#main = lambda reader: compose(consume, Map(compose(print, "{0[0]}: {0[1]}".format)),
main = lambda reader: compose(consume, plot, list,
                              Map(pam(identity, compose(numpy.abs, integrate_spectra, reader,Observation), Observation)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Integrate a number of spectra from the .h5 file(s)"
        )

    parser.add_argument("-v","--verbose",action="store_true",default=False,
                        help="This produces verbose output")

    parser.add_argument("-n", "--n-spectra", dest="n", default=10, type=int,
                              help="Number of spectra to integrate")
    parser.add_argument("-s", "--start-spectrum", dest="s", default=0, type=int,
                              help="Spectrum offset for start (0=first spectrum in file)")

    parser.add_argument("data_files", nargs='+',
                        help="*.h5 file names to process")
    parser.add_argument("-o", "--output", 
                        help="Save plots to this file")

    opts = parser.parse_args()

    # construct function which provides the spectra
    mk_reader = lambda obs: ReadBeamformHeap(obs, specnum=obs.begin_spectra+opts.s, 
                                                  end_specnum=obs.begin_spectra+opts.s+opts.n)
    main(mk_reader)( opts.data_files )
    #compose(consume, Map(proc_file))( opts.data_files )

    sys.exit( 0 )
