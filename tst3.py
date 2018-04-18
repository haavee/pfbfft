import sigproc_f as sigproc, matplotlib.pyplot as plt, numpy, userinput
#import sigproc_red
#SYNTH = sigproc.synthesizer_g

#SYNTH(400, sigproc.tones_g(1024, 12, tp=numpy.complex128, amplitude=.5+.2j))

#SYNTH = sigproc.synthesizer_g_cplx
#SYNTH(400, sigproc.tones_g(1024, [12, 70], tp=numpy.complex128, amplitude=1+1j))

#SYNTH = sigproc.synthesizer_g_real
#SYNTH = sigproc.synthesizer_g_real_2
SYNTH = sigproc.synthesizer_g_real_2_heap
rdbeamform_heap = lambda x: sigproc.rdbeamform_heap(x, nheap=50)
#SYNTH(400, sigproc.tones_g(1024, [12, 70], tp=numpy.float64, amplitude=1), sideband=+1)
import math

#  s = sigproc_red.freq_to_time(
#          numpy.vstack(sigproc.take(4, sigproc.replacer_h([(0, 0)], sigproc.rdbeamform_meerkat('/mnt/disk2/MeerKAT/vlbi_J0530+1331_array_scan08_bf_pol0_1519310448.h5', start_spectrum=1000000))))
#  )

s = SYNTH(856e6
         #, sigproc.tones_g_heap(1024, [(x*1e6, math.sin((x-1500) * (math.pi/100))+0.5 ) for x in range(1500, 1600, 8)], dc_freq=856e6, bandwidth=856e6, tp=numpy.float64, amplitude=1)
         ,sigproc.replacer_h([(0, 0)], sigproc.rdbeamform_meerkat('/mnt/disk2/MeerKAT/vlbi_J0530+1331_array_scan08_bf_pol0_1519310448.h5', start_spectrum=1000000)) #125428480-16384))
         #,sigproc.replacer_h([(0, 0)], sigproc.rdbeamform_meerkat('/mnt/disk0/MeerKAT/vlbi_J0237+2848_array_scan03_bf_pol0_1519306068.h5', start_spectrum=125428480-16384))
         , sideband=+1)

#s = SYNTH(-400e6
#      , sigproc.replacer_h([(0,0)], rdbeamform_heap("/mnt/disk0/ft010/scan4_1_20160922/295661428.dat"))
#      , sideband=+1)

#s = SYNTH(-400e6
#      #, sigproc.replacer_h([(0,0)], sigproc.rdbeamform_heap("/Users/verkouter/src/Data/scan4_1_20160922/295661428.dat"))
#      , sigproc.replacer_h([(0,0)], rdbeamform_heap("/Users/verkouter/src/Data/scan4_1_20160922/295661428.dat"))
#      #, sigproc.replacer_h([(0,0)], sigproc.rdbeamform_heap("/Users/verkouter/src/Data/scan6_1_20160922/317984512.dat"))
##      , sigproc.rdbeamform_heap("/Users/verkouter/src/Data/scan4_1_20160922/295661428.dat")
##      , sigproc.add_g(
##                       sigproc.tones_g(1024, [747], tp=numpy.float64, amplitude=6.1)
##                      , sigproc.tones_g(1024, [180], tp=numpy.float64, amplitude=3.4)
##                      , sigproc.tones_g(1024, [700, 710,720,730,750], tp=numpy.float64, amplitude=2.4)
#                      #, sigproc.tones_g(1024, [int((120.0/400)*1024)], tp=numpy.float64, amplitude=3.4)
#                      #, sigproc.replacer([(0,0)], sigproc.gaussian_noise_g(1024))
#                      #sigproc.auto_comb(40
#                        #,sigproc.replacer([(0,0)], sigproc.gaussian_noise_g(1024))
##                        , sigproc.replacer([(0,0)], sigproc.rdbeamform_raw("/Users/verkouter/src/Data/scan4_1_20160922/295661428.dat"))
#                      #)
#                      #, sigproc.tones_g(1024, [100, 110, 140], tp=numpy.float64, amplitude=1.13)
##                      )
#      , sideband=+1)
#s = SYNTH(400e6
#      , sigproc.add_g(  sigproc.tones_g(1024, [112], tp=numpy.complex128, amplitude=25+0j)
#                      , sigproc.replacer(  [(-1, 0+0j)]
#                                         , sigproc.rdbeamform("295661428.dat.snip")
#                                        )
#                      )
#      , sideband=-1)

#sigproc.dbbc(200, 800e6, 64e6, s)
#sigproc.dbbc(23e6, 800e6, 16e6, s)
(l1, u1) = sigproc.dbbc(userinput.lo, 1712e6, 2*userinput.bw, s)
#(l2, u2) = sigproc.dbbc_oldstyle(userinput.lo, 1712e6, 2*userinput.bw, s)
#sigproc_red.ddc(s, userinput.lo, userinput.bw)
sigproc.wr_vdif(l1, "/tmp/lsb1.vdif")
sigproc.wr_vdif(u1, "/tmp/usb1.vdif")
#sigproc.wr_vdif(l2, "/tmp/lsb2.vdif")
#sigproc.wr_vdif(u2, "/tmp/usb2.vdif")
plt.show()
