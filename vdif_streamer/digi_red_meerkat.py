from   __future__ import print_function
import re, os, operator, glob, numpy, sigproc, functools, fractions, h5py, pickle, sigproc_f
from   functional_hv import *
from   itertools     import takewhile, dropwhile


########  cf mail from Thomas Abbott (28 Mar 2018 13:22)
#
# Timestamps are ADC sample counts, at 1712 Msamples per second.
# - They start at zero when the digitiser is synchronised, which for this
#   stream is within about 20 microseconds of the UTC second boundary.
# - Which UTC second is defined by the Sync Epoch, the UNIX second of the sync
#   time. Sync Epoch should be included somewhere in the files you have, the h5
#   files are a good bet. It is common across the all parts of an observation.
# - So the UTC time of the first ADC sample in a .npy file is  SyncEpoch +
#   SampleCounter/1712.0e6    +- 20e-6
# - The same time stamps should be used to tag the beam spectra in the h5
#   files.


##  Maciej Serylak (28 Mar 2018 13:49)

#  In case of digitisers, there are also .npy files which are parts of the pcap
#  data sets converted to npy arrays. The timestamp files keep the first
#  timestamp of the recorded sample.

###### Maciej Serylak (28 Mar 2018 13:49)
# if you want to get sync time (which is done once every 40 hours or so) from correlator data file:
#  
#  import pickle
#  import h5py
#  h5 = h5py.File('/var/kat/archive3/comm-data/vlbi/vlbi_J0237+2848_array_scan03_corr_1519306068.h5')
#  pickle.loads(h5['TelescopeState'].attrs['i0_sync_time'])
#  
#  Procedure for beamformer file is identical:
#  
#      import pickle
#      import h5py
#      h5 = h5py.File('/var/kat/archive3/comm-data/vlbi/vlbi_J0237+2848_array_scan03_bf_pol0_1519306068.h5')
#      pickle.loads(h5['TelescopeState'].attrs['i0_sync_time'])
#  
#      For our experiment the sync time was: 1519299765
#  

# interactive Python looking for attributes having a particular string in their name
# A    = lambda obj: list(obj.attrs.keys())
# DOT  = operator.attrgetter
# SRCH = lambda s: re.compile(s, re.I).search
# list(map(DOT('string'), filter(operator.truth, map(SRCH('sample'), A(bf['TelescopeState']))))
## list(map(DOT('string'), filter(operator.truth, map(re.compile(r'sample', re.I).search, A(bf['TelescopeState'])))))
import datetime, math
class Observation(object):

    def __init__(self, fname, syncname, **kwargs):
        """fname: filename for digitizer data e.g.
            vlbi_J0237+2848_m011v_scan11_digi_1519313390.npy
           syncname: The observation metadata will be derived from this HDF5 file's properties/attributes;
                     may be correlator or beamformer data set
        
        Assumes <fname>.timestamp also exists - it reads the time stamp of the first sample from there
        """
        self.digi        = numpy.load(fname, mmap_mode='r')
        assert len(self.digi.shape) == 1 and self.digi.shape[0]>0, "Expect non-empty 1-dimensional time series"
        self.hdf5        = h5py.File(syncname, 'r')
        tel_state        = self.hdf5['TelescopeState']
        # get sync time & other meta stuff
        self.sync_time   = pickle.loads(tel_state.attrs['i0_sync_time'])
        assert round(self.sync_time) == int(self.sync_time), "Time was not synchronized on a UTC second"
        self.sync_time   = int(self.sync_time)
        self.bandwidth   = pickle.loads(tel_state.attrs['i0_bandwidth'])
        self.adc_sr      = pickle.loads(tel_state.attrs['i0_adc_sample_rate'])
        assert round(self.adc_sr) == int(self.adc_sr), "Non-integer amount of samples per second"
        self.adc_sr      = int(self.adc_sr)
        self.centre_freq = pickle.loads(tel_state.attrs['sdp_l0_center_freq'])
        self.dc_freq     = self.centre_freq - self.bandwidth/2.0
        # begin/end time sample counts
        self.begin_s     = int(open(fname+'.timestamp').read())
        self.end_s       = self.begin_s + self.digi.shape[0]
        sss = float(self.seconds_since_sync(self.begin_s)) + self.sync_time
        ts  = datetime.datetime.fromtimestamp(int(sss)).strftime('%Y-%m-%d %H:%M:%S')
        print(os.path.basename(fname),": first sample @",ts," +",math.modf(sss)[0])
        
    def seconds_since_sync(self, specnum):
        return fractions.Fraction(specnum, self.adc_sr)

    def specnum_from_time(self, ts):
        # convert time stamp to sample count since sync
        # then truncate to integer specnum + number of samples into timeseries
        sample_clock = ts * self.adc_sr
        # assert that the requested time stamp is, in fact, representable
        # as an ADC sample counter time stamp!
        assert sample_clock == int(sample_clock), "Requested time stamp {0} was not representable as ADC time stamp [{1}]".format(ts, sample_clock)
        return (int(sample_clock), 0)

    @property
    def begin_spectra(self):
        return self.begin_s #int(self.timestamps[0] // (2*self.n_chans))

    @property
    def end_spectra(self):
        return self.end_s #int(self.timestamps[-1] // (2*self.n_chans))

# bf_raw = 3D array: (n_chan, n_spectra, 2) where the 3rd dim is "complex" (real+imaginary)
#          dtype = int8, so we transform it into complex64 (the smallest complex)
#      using some 'astype' and 'view' majik
#un_raw = lambda bf_raw: bf_raw.astype(dtype=numpy.float32).view(dtype=numpy.complex64).squeeze()

class ReadDigi(object):
    def __init__(self, obs, specnum=None, end_specnum=None, verbose=False, raw=True):
        # give this object extra attributes 
        self.bandwidth          = obs.bandwidth
        self.dc_frequency       = obs.dc_freq 
        self.obs                = obs
        self.data               = obs.digi
        # requested start spectrum will be "absolute spectrum number" on entry or if None => start from the beginning
        self.requested_start_spectrum = self.spectrum_number = obs.begin_spectra if specnum is None else specnum
        self.end_spectrum                                    = obs.end_spectra if end_specnum is None else end_specnum
        assert self.spectrum_number >= obs.begin_spectra and self.spectrum_number<obs.end_spectra
        assert self.end_spectrum    >= obs.begin_spectra and self.end_spectrum<=obs.end_spectra
        self.requested_start_spectrum -= obs.begin_spectra
        self.spectrum_number          -= obs.begin_spectra
        self.end_spectrum             -= obs.begin_spectra

    def __iter__(self):
        return self

    def __next__(self):
        # the next batch of spectra
        # Attempt to read as many spectra from the current file as we can
        if self.spectrum_number>=self.end_spectrum:
            raise StopIteration
        t_slice = slice(self.spectrum_number, self.spectrum_number + min(self.end_spectrum - self.spectrum_number, 8*65536))
        tmp     = sigproc.TimeSeries(self.data[t_slice], SampleRate=self.obs.adc_sr, TimeStamp=self.obs.seconds_since_sync(self.spectrum_number))
        self.spectrum_number += (t_slice.stop - t_slice.start)
        return tmp


read_beamformdata = ReadDigi
freq_to_time      = lambda x: x
ddc               = sigproc.dbbc
