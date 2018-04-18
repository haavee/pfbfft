import os
import logging
import numpy as np
#import h5py
import optparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import scipy.signal

#import katpoint
def filetodict(path='',filename='obs_info.dat',delimiter=':'):
    d = {}
    d['directory'] = path
    with open("%s/%s"%(path,filename)) as f:
        for line in f:
            #print line
            if ':' in line:
                string = line.strip().split(delimiter)
                key = string[0]
                del(string[0])
                val = string 
                #print string
                val= ':'.join(string) # this will break on a 2 charecter line
                d[key.strip()] = val.strip()
    d['half_band'] = d['half_band']=='True'
    d['transpose'] = d['transpose']=='True'
    if d['half_band'] :
        d['n_chans'] = 512
    else:
        d['n_chans'] = 1024
    files = os.listdir(path)
    files.remove('obs_info.dat')
    dellist = []
    for name in files:
        if not name[-4:]=='.dat':dellist.append(name)
            
    for name in dellist:
        files.remove(name)
    files.sort()
    d['ants'] = d['ants'].replace("'","").replace("[","").replace("]","").split(",") # convert a list string into a list
    d['sync_time'] = np.double(d['sync_time'])
    d['scale_factor_timestamp'] = np.double(d['scale_factor_timestamp'])
    d['centre_freq']= np.double(d['centre_freq'])
    d['timestep'] = np.double(2048./800e6)
    #def tmp(packet_timestamp):
    #    return d['sync_time'] + packet_timestamp/d['scale_factor_timestamp']
    def tmp(spectra_timestamp):
        return d['sync_time'] + spectra_timestamp/(800e6/2048.)

    d['seconds_since_sync'] = tmp
    d['file_list'] = {}
    d['begin_spectra'] = np.zeros((len(files)))
    d['end_spectra'] = np.zeros((len(files)))
    d['filename'] = np.zeros((len(files)),dtype='S20') # hope it is long enough
    for i,name in enumerate(files):
        size = os.path.getsize(path+name)
        packet_time = np.int(name[:-4])
        #print "Number of spectrum in '%s' = %i "%(name,size/(d['n_chans']*2), )
        d['file_list'][i] = [name,packet_time,size/(d['n_chans']*2)]
        d['filename'][i] = name
        d['begin_spectra'][i] =  packet_time*32
        d['end_spectra'][i] =packet_time*32 + size/(d['n_chans']*2) # expected time stamp times
        #print "'%s' "%(name,),d['begin_timestamps'][i],d['end_timestamps'][i],size/(d['n_chans']*2)
    d['vis'] = None
    return d
def read_beamformdata(info,readheaps = 10,specnum=-1,verbose=False,raw=True):
    """directory # directory containing a beamformer obsevation string
    info = object detailing the beamformer obsevation   
    readheaps  = 10  # number of heaps to read in each chunk
    specnum = 0   # start spectrum number
    returns with shape  =  (readheaps*128,n_chans) unless end of file then less
    verbose=False,raw"""
    if specnum < info['begin_spectra'][0] : 
        specnum = info['begin_spectra'][0]
        if logging : logging.warning('Spectrum number before the start of the File, start of the file used')
    
    # find the valid files 
    validfiles =  ~(info['end_spectra'] <= specnum)# * (info['begin_spectra'] <= specnum) 
    if verbose : print  "There are %i valid files in the file list"%(validfiles.sum())
    
    leftover_read = 0
    a = None
    for i,filename in enumerate(info['filename']):
        # find the heap to read
        if validfiles[i] : 
            #print specnum > info['begin_spectra'][i]
            if verbose : print "Looking for the Heap number for spectrum %i in the file which starts at spectrum number %i "%(specnum ,info['begin_spectra'][i])
            #print "index %i , %s, %s "%(i,info['filename'][i],filename)
            heapnum  = np.floor(np.long(specnum - info['begin_spectra'][i])/128.)
            if heapnum < 0 :
                heapnum = 0
                if verbose : print  "Starting from the start of the file Heap offset number set to 0  "
            if verbose : print "Need to read Heap with offset number %i from '%s' which starts at heap number %i "%(heapnum,filename,info['begin_spectra'][i]/128) 
            with open(info['directory']+filename) as f:
                size = os.path.getsize(info['directory']+filename) # check
                heaps_in_file = size/(128*info['n_chans']*2)
                if verbose : print "Number of spectrum in '%s' = %i"%(filename,heaps_in_file*128)
                while heapnum < heaps_in_file:
                    if verbose : print "Going to heap number %i"%(heapnum)
                    f.seek(heapnum*info['n_chans']*128*2, os.SEEK_SET)  # seek to heap start
                    if leftover_read > 0:
                        readheaps_curr = leftover_read
                        leftover_read = 0
                    else:
                        readheaps_curr = readheaps
                    max_read = np.int(heaps_in_file-heapnum) # heaps left
                    if max_read < readheaps_curr:
                        if verbose : print "There are only %i heaps left in the file and we want to read %i heaps, Oh no! "%(max_read , readheaps_curr)
                        leftover_read = readheaps_curr - max_read
                        readheaps_curr = max_read

                        #readheaps = heaps_in_file-heapnum
                    if verbose : print "Going to Read %i heap(s) , %i heaps will be left in the file "%(readheaps_curr,heaps_in_file-heapnum-readheaps_curr)
                    d = np.fromfile(f,dtype=np.int8,count=readheaps_curr*info['n_chans']*128*2) 
                    if a is None:
                        a = d
                    else:
                        a = np.concatenate((a,d))

                    if leftover_read == 0:
                        #a = np.zeros((3*info['n_chans']*128*2),dtype=np.int8)
                        if not info['transpose'] :
                            if verbose : print "Transposing heaps"
                            a = a.reshape(readheaps,info['n_chans'],128,2).transpose((0,2,1,3)).reshape(readheaps*128,info['n_chans'],2)
                        if raw:
                            yield a 
                        else:
                            q = a.astype(np.float32)
                            q = q.view(np.complex64)
                            yield  q[...,0]
                        a = None
                    heapnum += readheaps_curr

 



import scipy.signal as signal

def freq_to_time(dataspec):
    N=8192 #N=2048
    P=4 
    def sinc(x):
        tmp = np.sin(np.pi*x)/(np.pi*x)
        tmp[np.isnan(tmp)] = 1.0
        return tmp
    #w = (  (sinc(np.arange(P * N)/N -P/2)   ) * np.hamming(P * N)  ).astype(np.float32).reshape(P,N)
   
    w = (signal.firwin(P * N, 1. / N) * N ).astype(np.float32).reshape(P,N)
    pfb_output = dataspec.flatten()

    no_samples = pfb_output.shape[0]
    #print no_samples
    #N/2 of non-redundant samples in a real fft by the Hermite-symmetric property (the last frequency bin is discarded due to KAT-7 infrastructure, should have ideally been N/2+1):
    fft_non_redundant = N / 2
    #print no_samples % fft_non_redundant
    assert(no_samples % fft_non_redundant == 0) #ensure we're only inverting an integral number of FFTs (with only non-redundant samples)
    ipfb_output_size = (no_samples/fft_non_redundant)*N
    
    '''
    INVERSE PFB
    '''
    pad = N*P
    #print ">>>Computing inverse pfb"
    '''
    Setup the inverse windowing function (note it should be the flipped impulse responses of the original subfilters,
    according to Daniel Zhou, A REVIEW OF POLYPHASE FILTER BANKS AND THEIR APPLICATION
    '''
    w_i = np.fliplr(w)
    #w_i = np.flipud(w)
    
    pfb_inverse_ifft_output = np.zeros(ipfb_output_size+pad).astype(np.float32)
    pfb_inverse_output = np.zeros(ipfb_output_size).astype(np.float32)

    '''
    for computational efficiency invert every FFT from the forward process only once... for xx large data
    we'll need a persistant buffer / ring buffer to store the previous IFFTs -- the buffering approach is explained and implemented in the CUDA version
    '''
    #for lB in xrange(0,no_samples,fft_non_redundant):
    #    #reverse to what we've done in the forward pfb: we jump in steps of N on the LHS and steps of N/2 on the RHS
    #    output_lB = (lB/fft_non_redundant)*N + pad
    #    output_uB = output_lB + N
    #    print lB,lB+fft_non_redundant
    #    #the inverse real FFT expects N/2 + 1 inputs by the Hermite property... We will have to pad each input chunk by 1. 
    #    padded_ifft_input = np.zeros(fft_non_redundant + 1,dtype=np.complex64)
    #    #padded_ifft_input[fft_non_redundant] = 0 #ensure the padded sample is set to 0
    #    #reverse the scaling factor (Parseval's Theorem) that we had to add in the forward python PFB to make it compatible with the CUDA real IFFT implementation
    #    padded_ifft_input[0:fft_non_redundant] = pfb_output[lB:lB+fft_non_redundant] * N
    #    pfb_inverse_ifft_output[output_lB:output_uB] = np.real(np.fft.irfftn(padded_ifft_input))
    pfb_inverse_ifft_output = np.fft.irfft(dataspec[:,:],axis=1,n=N).flatten()
    #print "Hello 1",(np.fft.irfft(dataspec[:,:],axis=1,n=N).flatten()==0.0)[-10000:].sum(),ipfb_output_size
    '''
    Inverse filterbank
    See discussion in ipfb GPU code
    '''
    for lB in range(0,ipfb_output_size-(P*N),N): 
        #print lB,lB+N ,lB,lB+(P*N)
        pfb_inverse_output[lB:lB+N] = np.flipud(pfb_inverse_ifft_output[lB:lB+(P*N)].reshape(P,N)*w_i).sum(axis=0)
    #for spec in xrange(0,pfb_inverse_ifft_output.shape[0]-P): 
    #    pfb_inverse_output[spec*N:(spec+1)*N] = (pfb_inverse_ifft_output[spec:spec+P,:]*w_i).sum(axis=0).flatten() * N
    #pfb_inverse_output = pfb_inverse_ifft_output.flatten() * N
    return pfb_inverse_output[:-(25*N)]


def ddc(input_data , lo_freq, bandwidth):
    #### PARAMETERS
    # input_data = np.array() np.int8 assumed
    # lo_freq: input_data frequency - target frequency (in Hz)
    # bandwith: target output bandwidth in MHz, allowed values: 32, 64, 128

    #### PARAMETERS

    # Input sample rate, fs
    sample_rate = 1712e6 #800e6
    no_samples = input_data.shape[0]#8192
    # Input tone frequency
    tone_freq = 711.49e6 #254e6
    # LO frequency used in mixing
    lo_freq = lo_freq#lo_freq
    # Interpolate by factor L and decimate by factor M
    # The settings below give 800e6 * L / M = 128e6

    #L = 4
    #M = 25
    # Integer factors for the z^(-1) trick, determined via Euclid's theorem by solving -n0 * L + n1 * M = 1
    #n0 = 31
    #n1 = 5
    L, M, n0, n1 = {32: (2, 25, 12, 1),
                    64: (4, 25, 31, 5),
                    128: (8, 25, 28, 9)}[bandwidth]
    # MeerKAT 1712Msamp/s * 8 / 107 == 128Msamp/s == 64MHz
    L, M, n0, n1  = 8, 107, 40, 3
    # Filter parameters (operating at a sample rate of L * fs)
    taps = 128
    cutoff = 64e6

    # Number of FFT samples in plots
    no_freqs = 8 * no_samples

    #### DESIGN FILTER

    # Filter operates at highest sample rate (refer to brute-force block diagram)
    nyq = L * sample_rate / 2.
    # Transition bandwidth for Hamming-windowed design
    trans_bw = nyq * 8 / taps
    # Maximum allowed ripple (in linear terms)
    pass_ripple = stop_ripple = 1 / 256.
    # More refined filter specification of passband and stopband edge frequencies
    # The Remez design reduces the transition BW to 0.64 of that of the Hamming design
    # This keeps the maximum passband and stopband ripple below 1/256, which is good for 8-bit data
    pass_edge = cutoff - .64 * trans_bw / 2.
    stop_edge = cutoff + .64 * trans_bw / 2.
    # Relative weight given to matching the stopband
    stop_weight = 0.5
    # Normalise all frequencies to Nyquist (i.e. f = 1 means pi rads / sample)
    wc, wp, ws = cutoff / nyq, pass_edge / nyq, stop_edge / nyq


    ## Use windowing method
    h_window = scipy.signal.firwin(taps, wc)

    ## Use Parks-Mclellan aka Remez exchange method
    # Increased grid density a bit, as filter bandwidth is quite low (1 / M)
    #h_remez = scipy.signal.remez(taps, [0, wp, ws, 1], [1, 0], weight=[1 - stop_weight, stop_weight],
    #                             Hz=2, grid_density=64)

    ## Use eigenfilter method
    # This method designs a FIR filter h[n] of even order N, having N+1 tap weights h[0] to h[N].
    # Since the filter has even symmetry to ensure linear phase, it can be represented by
    # the (N // 2 + 1) Fourier cosine coefficients b[m] of its real and even amplitude response.
    # In this case, N is taken to be taps - 2, to ensure that this filter is not longer than one above.
    #coefs = (taps - 2) // 2 + 1
    #mm, nn = np.meshgrid(np.arange(coefs), np.arange(coefs))
    #Pstop = 0.5 * (np.sinc(nn + mm) - ws*np.sinc((nn + mm)*ws) + np.sinc(nn - mm) - ws*np.sinc((nn - mm)*ws))
    #Ppass = wp * (1 - np.sinc(nn*wp) - np.sinc(mm*wp) + 0.5*np.sinc((nn + mm)*wp) + 0.5*np.sinc((nn - mm)*wp))
    #Poverall = stop_weight * Pstop + (1 - stop_weight) * Ppass
    # Pick eigenvector associated with smallest eigenvalue (minimising Rayleigh quotient)
    #u, s, vh = np.linalg.svd(Poverall)
    #b = u[:, -1]
    # Recreate filter tap weights h[n] from cosine coefficients b[m]
    #h_eigen = np.hstack((0.5 * np.flipud(b[1:]), (b[0],), 0.5 * b[1:]))
    #h_eigen /= h_eigen.sum()
    #h = np.hstack((h_eigen, (0,)))

    #print "Effective transition regions:"
    #for name, filt in zip(['window', 'remez', 'eigen'], [h_window, h_remez, h_eigen]):
        # Amplitude response with MHz bins
    #    nfreqs = int(2 * nyq / 1e6)
    #    amp_resp = np.abs(np.fft.fft(filt, nfreqs))
    #    eff_pass_edge = (amp_resp[:nfreqs // 2] > 1 - pass_ripple).tolist().index(False)
    #    eff_stop_edge = (amp_resp[:nfreqs // 2] < stop_ripple).tolist().index(True)
    #    print "%s = %d to %d MHz" % (name, eff_pass_edge, eff_stop_edge)

    # Choose filter
    h = h_window

    #### CREATE SIGNALS

    # Discrete time indices
    n = np.arange(no_samples)
    # LO signal
    cos_lo_n = 2 * np.cos(2 * np.pi * lo_freq / sample_rate * n)
    # Input signal: single sinusoid at given frequency
    #x = np.cos(2 * np.pi * tone_freq / sample_rate * n)
    # Input signal: bandlimited white noise
    #x = scipy.signal.lfilter(h, 1, np.random.randn(no_samples)) * \
    #    np.cos(2 * np.pi * (lo_freq + cutoff / 4) / sample_rate * n)


    ## Use windowing method
    #print "  scipy.signal.firwin(%i,%f) "%(taps,wc)
    
    x =  input_data #np.fromfile(input_file,dtype=np.int8, count=no_samples)
    #### PERFORM MORE EFFICIENT DDC

    # Mix signal
    i_mix = x * cos_lo_n
    # Polyphase components (type 1: E_0 to E_(L-1) or type 2: R_(L-1) to R_0)
    poly_h = h.reshape(-1, L)
    poly_taps = poly_h.shape[0]
    # The delays might not be correct for all L and M...
    poly_delay = np.arange(0, -L * n0, -n0) + M * (np.arange(0, L * n1, n1) // L)
    #print poly_delay
    poly_delay = np.flipud(poly_delay) - poly_delay.min()
    # Effective filter memory
    memory = poly_delay.max() + poly_taps
    # Prepend zeroes to signal to serve as initial filter memory
    # (this makes the comparison with the brute-force approach exact)
    #print "Padding needed", poly_taps - 1,np.zeros(poly_taps - 1).shape
    i_mix_plus_memory = i_mix #padding done out side loop#np.hstack((np.zeros(poly_taps - 1), i_mix))
    # Number of output samples when first interpolating and then decimating
    no_outputs_up_down = (L * no_samples) // M #+ 1
    y = np.zeros(no_outputs_up_down)
    for l in range(L):
        tap_weights = np.flipud(poly_h[:, l])
        #print range(poly_delay[l], len(i_mix_plus_memory) - poly_taps, M)
        for k, start in enumerate(range(poly_delay[l], len(i_mix_plus_memory) - poly_taps, M)):
            y[l + k * L] = np.dot(i_mix_plus_memory[start:start + poly_taps], tap_weights)
    #assert np.all(y == i_out), 'Efficient version not identical to brute-force one'

    #print "Dumping output to disk\n"
    return y#.astype(np.int8).tofile("py_ddc_out.dump") #dump the output as 8bit signed ints

if __name__ == "__main__":
    parser = optparse.OptionParser(usage='%prog [options] <data dir1>  <optional additional data dir>...',
                                   description='This script transposes beamformer obs into a compressed h5 file')

    parser.add_option("-v","--verbose",action="store_true",default=False,
                      help="This produces verbose output")
    parser.add_option("-f","--DDC-freq",default=1822.0,
                      help="This is the top end of the ddc channel in MHz")

    (opts, args) = parser.parse_args()
    DDC_topend  = opts.DDC_freq
    import time
    starttime = time.time()
    if len(args) >= 1 :
        for path in args:
            filename = path.split('/')[-1]
            if filename == '' : filename = path.split('/')[-2]
            #FILE = filename
            directory=path
            info1 = filetodict(directory)
            verbose=False
            lo_freq = info1['centre_freq'] +200.0 - float(DDC_topend)
            beam1 = read_beamformdata(info1,readheaps = 500,specnum=-1,verbose=verbose,raw=False)
            datapad=None
            ddcpad = None
            EDIM_current =0
            i = 0
            print "First sample at %25f25"%( info1['seconds_since_sync'](info1['begin_spectra'][0]))
            for data in beam1: # 
                #print "Beam Data",data.shape ,data.shape[0]*data.shape[1]*2
                #print "time taken to read data = %f"% (time.time()- starttime)
                if not datapad is None : 
                    data = np.r_[datapad,data]
                outputdata = freq_to_time(data) #output will be short 8192 samples 
                #print data.shape[0]*data.shape[1]*2,outputdata.shape
                datapad = data[-25:,:] #4*6.25  samples  for next sample
                #print "time taken for ipfb = %f"% (time.time()- starttime)
                #print "Time data",outputdata.shape
                #x = np.fft.fft(outputdata.reshape(-1,2048),axis=1)
                #plt.plot(((np.abs(x[:,1:1024] ).mean(axis=0)))) 
                if not ddcpad is None : 
                    outputdata = np.hstack((ddcpad,outputdata))
                ddcpad = outputdata[-50:]
                filtered_data =  ddc(outputdata,lo_freq=lo_freq*1e6)[:-8] # 50/6.26 = 8  # has to be bigger than (taps/L)/6.25
                #x = np.fft.fft(filtered_data[0:(filtered_data.shape[0]//2048)*2048].reshape(-1,2048),axis=1)
                #plt.plot(((np.abs(x[:,1:1024] ).mean(axis=0)))) 
                print "filtered data" ,filtered_data.shape

                if i > 0 :
                    break
                i = i + 1


    else:
        raise IOError('Script neads >=1 path(s) given to it')

    print "First sample at %25f25"%( info1['seconds_since_sync'](info1['begin_spectra'][0]))
    print "time taken = %f"% (time.time()- starttime)
    #rand = ddc(np.random.normal(scale=16,size=21474755).astype(np.int8))
    #x = np.fft.fft(rand[0:(rand.shape[0]//2048)*2048].reshape(-1,2048),axis=1)
    #plot(((np.abs(x[:,1:1024] ).mean(axis=0)))) 

    #rand = ddc(np.random.normal(scale=0.067072593,size=21474755))
    #x = np.fft.fft(rand[0:(rand.shape[0]//2048)*2048].reshape(-1,2048),axis=1)
    #plt.plot(((np.abs(x[:,1:1024] ).mean(xaxis=0)))) 


    #plt.plot(np.linspace(+200,-200,1024)+info1['centre_freq'],np.abs(data[:,:]).mean(axis=0)  ,label='Data from Beamformer') 
    #x = np.fft.fft(outputdata.reshape(-1,2048),axis=1)
    #plt.plot(np.linspace(+200,-200,1024)+info1['centre_freq'],((np.abs(x[:,0:1024] ).mean(axis=0))),label='Data after IPFB , 1024 channels')
    #x = np.fft.fft(filtered_data[0:(filtered_data.shape[0]//256)*256].reshape(-1,256),axis=1)
    ##plt.plot(float(DDC_topend)-np.linspace(0,64,128),((np.abs(x[:,0:128]).mean(axis=0)*23)),label='Data after DDC 128 Channels (scaled)') 
    #plt.legend()

    #plt.plot(np.linspace(0,400,1024),np.abs(data[:,:]).mean(axis=0)  ,label='Data from Beamformer') 
    #x = np.fft.fft(outputdata.reshape(-1,32*2048),axis=1)
    #plt.plot(np.linspace(0,400,32*1024),((np.abs(x[:,0:32*1024] ).mean(axis=0)/4.)),label='Data after IPFB , 32768 channels')
    #x = np.fft.fft(filtered_data[0:(filtered_data.shape[0]//(32*256))*(32*256)].reshape(-1,32*256),axis=1)
    #plt.plot(np.linspace(0,64,32*128),((np.abs(x[:,0:32*128]).mean(axis=0)*23))/4.,label='Data after DDC 8196 Channels (scaled)') 
    #plt.legend()




    #poly_delay = np.array([ 0,  6, 12, 18])
    #L,M = 4 , 25
    #poly_taps = 32 
    #131071488 = len(i_mix_plus_memory)

    #for l in range(L):
    #    tap_weights = np.flipud(poly_h[:, l])
    #    print range(poly_delay[l], len(i_mix_plus_memory) - poly_taps, M)
    #    for k, start in enumerate(range(poly_delay[l], len(i_mix_plus_memory) - poly_taps, M)):
    #        y[l + k * L] = np.dot(i_mix_plus_memory[start:start + poly_taps], tap_weights)



    #    for lB in range(0,ipfb_output_size-(P*N),N): 
            #print lB,lB+N ,lB,lB+(P*N)
    #        pfb_inverse_output[lB:lB+N] = np.flipud(pfb_inverse_ifft_output[lB:lB+(P*N)].reshape(P,N)*w_i).sum(axis=0)
    #L,M = 4 , 25
    #(L * no_samples) // M + 1
