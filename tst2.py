import sigproc_f as sigproc, matplotlib.pyplot as plt, numpy, userinput

#SYNTH = sigproc.synthesizer_g

#SYNTH(400, sigproc.tones_g(1024, 12, tp=numpy.complex128, amplitude=.5+.2j))

#SYNTH = sigproc.synthesizer_g_cplx
#SYNTH(400, sigproc.tones_g(1024, [12, 70], tp=numpy.complex128, amplitude=1+1j))

SYNTH = sigproc.synthesizer_g_real
#SYNTH(400, sigproc.tones_g(1024, [12, 70], tp=numpy.float64, amplitude=1), sideband=+1)
s = SYNTH(400e6
      , sigproc.add_g(
                      #  sigproc.tones_g(1024, [280], tp=numpy.float64, amplitude=6.1)
                      #, sigproc.tones_g(1024, [180], tp=numpy.float64, amplitude=3.4)
                      #, sigproc.tones_g(1024, [int((120.0/400)*1024)], tp=numpy.float64, amplitude=3.4)
                      #, sigproc.replacer([(0,0)], sigproc.gaussian_noise_g(1024))
                      sigproc.auto_comb(40,
                        sigproc.replacer([(0,0)], sigproc.gaussian_noise_g(1024))
                      )
                      #, sigproc.tones_g(1024, [100, 110, 140], tp=numpy.float64, amplitude=1.13)
                      )
      , sideband=+1)
#s = SYNTH(400e6
#      , sigproc.add_g(  sigproc.tones_g(1024, [112], tp=numpy.complex128, amplitude=25+0j)
#                      , sigproc.replacer(  [(-1, 0+0j)]
#                                         , sigproc.rdbeamform("295661428.dat.snip")
#                                        )
#                      )
#      , sideband=-1)

#sigproc.dbbc(200, 800e6, 64e6, s)
#sigproc.dbbc(23e6, 800e6, 16e6, s)
sigproc.dbbc(userinput.lo, 800e6, 2*userinput.bw, s)
plt.show()
