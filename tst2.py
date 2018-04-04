import sigproc_f as sigproc, matplotlib.pyplot as plt, numpy, userinput

#SYNTH = sigproc.synthesizer_g

#SYNTH(400, sigproc.tones_g(1024, 12, tp=numpy.complex128, amplitude=.5+.2j))

#SYNTH = sigproc.synthesizer_g_cplx
#SYNTH(400, sigproc.tones_g(1024, [12, 70], tp=numpy.complex128, amplitude=1+1j))

#SYNTH = sigproc.synthesizer_g_real
#SYNTH = sigproc.synthesizer_g_real_2
SYNTH = sigproc.synthesizer_g_real_2_heap
#rdbeamform_heap = lambda x: sigproc.rdbeamform_heap(x, nheap=50)
#SYNTH(400, sigproc.tones_g(1024, [12, 70], tp=numpy.float64, amplitude=1), sideband=+1)
s = SYNTH(-400e6
      #, sigproc.replacer_h([(0,0)], sigproc.rdbeamform_heap("/Users/verkouter/src/Data/scan4_1_20160922/295661428.dat"))
      , sigproc.replacer_h([(0,0)], sigproc.rdbeamform_heap("/Users/verkouter/src/Data/scan4_1_20160922/295661428.dat", nheap=50, start_spectrum=3054))
      #, sigproc.replacer_h([(0,0)], sigproc.rdbeamform_heap("/Users/verkouter/src/Data/scan6_1_20160922/317984512.dat"))
#      , sigproc.rdbeamform_heap("/Users/verkouter/src/Data/scan4_1_20160922/295661428.dat")
#      , sigproc.add_g(
#                       sigproc.tones_g(1024, [747], tp=numpy.float64, amplitude=6.1)
#                      , sigproc.tones_g(1024, [180], tp=numpy.float64, amplitude=3.4)
#                      , sigproc.tones_g(1024, [700, 710,720,730,750], tp=numpy.float64, amplitude=2.4)
                      #, sigproc.tones_g(1024, [int((120.0/400)*1024)], tp=numpy.float64, amplitude=3.4)
                      #, sigproc.replacer([(0,0)], sigproc.gaussian_noise_g(1024))
                      #sigproc.auto_comb(40
                        #,sigproc.replacer([(0,0)], sigproc.gaussian_noise_g(1024))
#                        , sigproc.replacer([(0,0)], sigproc.rdbeamform_raw("/Users/verkouter/src/Data/scan4_1_20160922/295661428.dat"))
                      #)
                      #, sigproc.tones_g(1024, [100, 110, 140], tp=numpy.float64, amplitude=1.13)
#                      )
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
(l, u) = sigproc.dbbc(userinput.lo, 800e6, 2*userinput.bw, s)
sigproc.wr_vdif(l, "/tmp/lsb_f.vdif")
sigproc.wr_vdif(u, "/tmp/usb_f.vdif")
plt.show()
