import sigproc, matplotlib.pyplot as plt, numpy

#SYNTH = sigproc.synthesizer_g

#SYNTH(400, sigproc.tones_g(1024, 12, tp=numpy.complex128, amplitude=.5+.2j))

#SYNTH = sigproc.synthesizer_g_cplx
#SYNTH(400, sigproc.tones_g(1024, [12, 70], tp=numpy.complex128, amplitude=1+1j))

SYNTH = sigproc.synthesizer_g_real
#SYNTH(400, sigproc.tones_g(1024, [12, 70], tp=numpy.float64, amplitude=1), sideband=+1)
#SYNTH(400
#      , sigproc.add_g(  sigproc.tones_g(1024, [12], tp=numpy.float64, amplitude=0.13)
#                      , sigproc.tones_g(1024, [70], tp=numpy.float64, amplitude=1.13))
#      , sideband=+1)
SYNTH(400
      , sigproc.add_g(  sigproc.tones_g(1024, [112], tp=numpy.complex128, amplitude=25+0j)
                      , sigproc.replacer(  [(-1, 0+0j)]
                                         , sigproc.rdbeamform("295661428.dat.snip")
                                        )
                      )
      , sideband=-1)
plt.show()
