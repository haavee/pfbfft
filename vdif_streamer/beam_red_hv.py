from   __future__ import print_function
import re, os, operator, glob, numpy, sigproc, functools, fractions
from   functional_hv import *
from   itertools     import takewhile, dropwhile

isKeyValue = lambda delimiter: re.compile(r"^(?P<key>[^{0}]+){0}(?P<value>[^{0}]+)$".format(delimiter)).match
isDataFile = re.compile(r"(^|/)(?P<packet_time>[0-9]+).dat$").search

class DataFile(object):

    def __init__(self, fn, n_chans=None):
        # accept regex match object or string (turn into match object)
        validFile = isDataFile(fn) if isinstance(fn, str) else fn
        assert validFile is not None
        # now we can extract the file's properties!
        self.file_name     = validFile.string
        self.file_size     = os.path.getsize( self.file_name )
        nSpec              = self.file_size / (2*n_chans)
        if int(nSpec)!=nSpec:
            raise RuntimeError("File {0} does not contain an integer amount of spectra?! file_size={1}, spectrum_size={2}".format(self.file_name, self.file_size, 2*n_chans))
        nSpec              = int(nSpec)
        self.packet_time   = int(validFile.group('packet_time'))
        self.begin_spectra = 32 * self.packet_time
        self.end_spectra   = self.begin_spectra + nSpec


class Observation(object):

    def __init__(self, path, **kwargs):
        """path: directory containing KAT7 beamformed output + obs_info.dat file describing it
        **kwargs: filename=<...> to override default 'obs_info.dat'
                  delimiter=X    to override default ':'"""
        settings       = do_update(dict(filename="obs_info.dat", delimiter=':'), kwargs)
        self.directory = path
        prefix         = partial(os.path.join, self.directory)
        keywords       = reduce(do_update, 
                                map(lambda mo : { mo.group('key').strip() : mo.group('value').strip() },
                                    filter(truth,
                                        map(compose(isKeyValue(settings['delimiter']), str.strip),
                                            open(prefix(settings['filename']))))), dict())
        self.half_band   = (keywords['half_band']=='True')
        self.transpose   = (keywords['transpose']=='True')
        self.n_chans     = 512 if self.half_band else 1024
        self.sync_time   = numpy.double(keywords['sync_time'])
        assert int(self.sync_time) == round(self.sync_time)
        self.sync_time   = int(self.sync_time)
        self.scale_factor_timestamp = numpy.double(keywords['scale_factor_timestamp'])
        self.centre_freq = numpy.double(keywords['centre_freq']) * 1E6  # implicit MHz in file => to Hz here
        self.bandwidth   = -400e6    # KAT7 fixed bandwidth of 400MHz
        self.dc_freq     = self.centre_freq + (abs(self.bandwidth) /2) # DC edge is the highest freq here because it's LSB
        #self.timestep    = numpy.double(self.n_chans / self.bandwidth) #(2048./800e6) <-- 2 * n_chan / 2 * bw?
        self.spectra_p_s = fractions.Fraction(int(abs(self.bandwidth)), self.n_chans)
        self.ants        = re.sub(r"[][']", "", keywords['ants']).split(',')
        # Compile the list of datafiles and sort them by start spectrum number
        self.file_list   = sorted(map(partial(DataFile, n_chans=self.n_chans), filter(truth, map(isDataFile, glob.glob(prefix("*.dat"))))), 
                                  key=operator.attrgetter('begin_spectra'))
        if not self.file_list:
            raise RuntimeError("There are no observation files found in "+self.directory)

    def seconds_since_sync(self, specnum):
        return self.sync_time + fractions.Fraction(specnum, self.spectra_p_s)
        #return self.sync_time + specnum * self.timestep

    @property
    def begin_spectra(self):
        return self.file_list[0].begin_spectra
    @property
    def end_spectra(self):
        return self.file_list[-1].end_spectra

# Ignore readheaps
nSpecPerHeap = 128

class ReadBeamformData(object):
    def __init__(self, obs, readheaps = 10, specnum=None, end_specnum=None, verbose=False, raw=True):
        # give this object extra attributes 
        self.bandwidth                = obs.bandwidth
        # if we know the observation metadata we can precompute a number of things
        #  'heap size' = number of spectra per heap * (size of spectrum = n_channels * 2 (for Re + Im)) * 1 byte per number
        self.n_chan                   = obs.n_chans
        self.heap_size                = nSpecPerHeap * self.n_chan * 2
        self.obs                      = obs
        #  DC frequency and bandwidth:
        #  For KAT7 frequency axis in the data goes down towards higher channel number (indicated here by neg. bandwidth)
        # DC edge is at index 0 but only the center frequency is in the obs. info
        self.bandwidth                = obs.bandwidth
        self.dc_frequency             = obs.dc_freq 
        # requested start spectrum will be "absolute spectrum number" on entry or if None => start from the beginning
        self.requested_start_spectrum = self.start_spectrum = self.spectrum_number = obs.begin_spectra if specnum is None else specnum
        self.end_spectrum             = obs.end_spectra if end_specnum is None else end_specnum
        # Skip to the data file containing the requested spectrum
        def reductor(acc, nextfile):
            if acc and (nextfile.begin_spectra != acc[-1].end_spectra ):
                raise AssertionError("Missing spectra in selected time range: {0}.end_spectra[{2}] != {1}.begin_spectra[{3}]".format(acc[-1].file_name, nextfile.file_name, acc[-1].end_spectra, nextfile.begin_spectra))
            return acc.append( nextfile ) or acc

        files  = reduce(reductor, takewhile(lambda df: df.begin_spectra < self.end_spectrum,
                                            dropwhile(lambda df: df.end_spectra < self.start_spectrum, obs.file_list)), [])
        if not files:
            raise RuntimeError("No data found for selected spectrum range {0} -> {1}".format(self.start_spectrum, self.end_spectrum))
        self.file_list                = iter(files)
        self.current_fd               = None
        self.current_heap             = self.read_heap()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            nextSpectrum = sigproc.Spectrum(self.current_heap[self.start_spectrum], DCEdge=0,
                                            BandWidth=self.bandwidth, DCFrequency=self.dc_frequency, TimeStamp=self.obs.seconds_since_sync(self.spectrum_number))
            self.start_spectrum  = self.start_spectrum + 1
            self.spectrum_number = self.spectrum_number + 1
            return nextSpectrum
        except:
            # exhausted current heap, reset spectrum-within-heap-counter and get next heap from file 
            self.start_spectrum = 0
            self.current_heap   = self.read_heap()
            return next(self)

    def read_heap(self):
        try:
            # read one heap from file and transform into array of complex spectra
            return numpy.fromfile(self.current_fd, dtype=numpy.int8, count=self.heap_size).astype(numpy.float32).view(numpy.complex64).reshape((self.n_chan, nSpecPerHeap)).T
        except Exception as E:
            # move to next file?
            if self.current_fd is not None:
                self.current_fd.close()
            self.current_file = next(self.file_list)
            self.current_fd   = open(self.current_file.file_name)
            # If the requested start spectrum > 0 we're supposed to seek to a specific spectrum
            # but the files are blocked in "heaps" of 128 spectra so we must seek 
            # to the actual heap that contains the requested spectrum
            if self.start_spectrum>0:
                # compute heap number - we need it twice
                heap_number = int((self.start_spectrum - self.current_file.begin_spectra)/nSpecPerHeap)
                # seek to 'heap number * heap size' 
                self.current_fd.seek(  heap_number * self.heap_size, os.SEEK_SET )
                # change absolute requested spectrum number into relative, as in: relative to start of current heap!
                self.start_spectrum -= self.current_file.begin_spectra + heap_number * nSpecPerHeap
            return self.read_heap()

class ReadBeamformHeap(object):
    def __init__(self, obs, readheaps = 20, specnum=None, end_specnum=None, verbose=False, raw=True):
        # give this object extra attributes 
        self.bandwidth                = obs.bandwidth
        self.n_heap                   = readheaps
        # if we know the observation metadata we can precompute a number of things
        #  'heap size' = number of spectra per heap * (size of spectrum = n_channels * 2 (for Re + Im)) * 1 byte per number
        self.n_chan                   = obs.n_chans
        self.heap_size                = nSpecPerHeap * self.n_chan * 2
        self.obs                      = obs
        #  DC frequency and bandwidth:
        #  For KAT7 frequency axis in the data goes down towards higher channel number (indicated here by neg. bandwidth)
        # DC edge is at index 0 but only the center frequency is in the obs. info
        self.bandwidth                = obs.bandwidth
        self.dc_frequency             = obs.dc_freq 
        # requested start spectrum will be "absolute spectrum number" on entry or if None => start from the beginning
        self.requested_start_spectrum = self.start_spectrum = self.spectrum_number = obs.begin_spectra if specnum is None else specnum
        self.end_spectrum             = obs.end_spectra if end_specnum is None else end_specnum

        def reductor(acc, nextfile):
            if acc and (nextfile.begin_spectra != acc[-1].end_spectra ):
                raise AssertionError("Missing spectra in selected time range: {0}.end_spectra[{2}] != {1}.begin_spectra[{3}]".format(acc[-1].file_name, nextfile.file_name, acc[-1].end_spectra, nextfile.begin_spectra))
            return acc.append( nextfile ) or acc

        files  = reduce(reductor, takewhile(lambda df: df.begin_spectra < self.end_spectrum,
                                            dropwhile(lambda df: df.end_spectra < self.start_spectrum, obs.file_list)), [])
        if not files:
            raise RuntimeError("No data found for selected spectrum range {0} -> {1}".format(self.start_spectrum, self.end_spectrum))
        self.file_list                = iter(files)
        self.current_fd               = None
        self.current_nheap            = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            # Attempt to read as many heaps from the current file as we can
            act_nheap = min(self.n_heap, self.current_nheap)
            tmp = sigproc.Spectrum(numpy.fromfile(self.current_fd, dtype=numpy.int8, count=act_nheap*self.heap_size)\
                                .astype(numpy.float32)\
                                .view(numpy.complex64)\
                                .reshape((act_nheap, nSpecPerHeap, self.n_chan))\
                                .transpose((0, 2, 1))\
                                .reshape((act_nheap * nSpecPerHeap, self.n_chan)),
                                DCEdge=0, DCFrequency=self.dc_frequency, BW=self.bandwidth)
            # ok that worked w/o throwing an exception so now we can update our internals
            self.current_nheap -= act_nheap
            # skip to start spectrum only once
            if self.start_spectrum>0:
                tmp = tmp[self.start_spectrum:]
                self.start_spectrum = 0
            return tmp
        except Exception as E:
            # move to next file?
            if self.current_fd is not None:
                self.current_fd.close()
            self.current_file  = next(self.file_list)
            self.current_fd    = open(self.current_file.file_name)
            self.current_nheap = self.current_file.file_size // self.heap_size
            # If the requested start spectrum > 0 we're supposed to seek to a specific spectrum
            # but the files are blocked in "heaps" of 128 spectra so we must seek 
            # to the actual heap that contains the requested spectrum
            if self.start_spectrum>0:
                # compute heap number - we need it twice
                heap_number = int((self.start_spectrum - self.current_file.begin_spectra)/nSpecPerHeap)
                # seek to 'heap number * heap size' 
                self.current_fd.seek(  heap_number * self.heap_size, os.SEEK_SET )
                # change absolute requested spectrum number into relative, as in: relative to start of current heap!
                self.start_spectrum -= self.current_file.begin_spectra + heap_number * nSpecPerHeap
            return next(self)


def read_beamformdata_impl(obs, readheaps = 10, specnum=None, verbose=False, raw=True):
    print("read_beamformdata/ before: dir(...)=",dir(read_beamformdata))
    # give this object extra attributes 
    read_beamformdata.bandwidth = obs.bandwidth
    print("read_beamformdata/ after:  dir(...)=",dir(read_beamformdata))
    # if we know the observation metadata we can precompute a number of things
    #  'heap size' = number of spectra per heap * (size of spectrum = n_channels * 2 (for Re + Im)) * 1 byte per number
    heap_size                = nSpecPerHeap * obs.n_chan * 2
    #  DC frequency and bandwidth:
    #  For KAT7 frequency axis in the data goes down towards higher channel number (indicated here by neg. bandwidth)
    # DC edge is at index 0 but only the center frequency is in the obs. info
    bandwidth                = obs.bandwidth
    dc_frequency             = obs.dc_freq
    # requested start spectrum will be "absolute spectrum number" on entry or if None => start from the beginning
    requested_start_spectrum = spectrum_number = obs.begin_spectra if specnum is None else specnum
    # Skip to the data file containing the requested spectrum
    for df in itertools.drop_while(lambda df: df.end_spectra<requested_start_spectrum, obs.file_list):
        print("Reading file: ",df.file_name)
        with open(df.file_name) as infile:
            # If the requested start spectrum > 0 we're supposed to seek to a specific spectrum
            # but the files are blocked in "heaps" of 128 spectra so we must seek 
            # to the actual heap that contains the requested spectrum
            if requested_start_spectrum>0:
                # compute heap number - we need it twice
                heap_number = (requested_start_spectrum - df.begin_spectra)/nSpecPerHeap
                # seek to 'heap number * heap size' 
                infile.seek(  heap_number * heap_size, os.SEEK_SET )
                # change absolute requested spectrum number into relative, as in: relotive to start of current heap!
                requested_start_spectrum -= heap_number * nSpecPerHeap
            # file is open + positioned at start of heap so now read it
            curHeap = numpy.fromfile(infile, dtype=numpy.int8, count=heap_size).astype(numpy.float32).view(numpy.complex64).reshape((obs.n_chan, nSpecPerHeap)).T
            for s in range(requested_start_spectrum, curHeap.shape[0]):
                yield sigproc.Spectrum(curHeap[s], DCEdge=0, BandWidth=bandwidth, DCFrequency=dc_frequency, TimeStamp=obs.seconds_since_sync(spectrum_number))
                spectrum_number = spectrum_number + 1
            # after having sought (seeked) to requested start spectrum/heap first, we don't have to do that again
            requested_start_spectrum = 0

#def read_beamformdata(obs, readheaps = 10, specnum=None, verbose=False, raw=True):
def read_beamformdata_old(obs, *args, **kwargs):
    @functools.wraps(read_beamformdata)
    def do_it():
        print("read beamformdata/do_it")
        for x in read_beamformdata_impl(obs, *args, **kwargs):
            yield x
    rv = do_it()
    rv.bandwidth = obs.bandwidth
    return rv

read_beamformdata = ReadBeamformHeap #ReadBeamformData
#freq_to_time      = sigproc.synthesizer_real( sigproc.replacer((0, 0+0j)), sigproc.auto_comb(40) )
#freq_to_time      = sigproc.synthesizer_real( sigproc.replacer((0, 0+0j)) )
freq_to_time      = sigproc.synthesizer_g_real_irfft_heap( sigproc.replacer_heap((0, 0+0j)) )
ddc               = sigproc.dbbc
