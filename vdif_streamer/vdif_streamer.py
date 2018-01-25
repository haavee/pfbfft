#!/usr/bin/env python
from __future__ import print_function

from vdif_header import VDIF_Header
import beam_red_hv as beam_red

import numpy 
import numpy.ctypeslib as npct

import argparse
from datetime import datetime
from ctypes import c_ubyte, c_int, c_double
import os
import time
import calendar
import fractions
import collections
#from numba import jit

#@jit
def requantize(ds):
    ds = ((ds // (1.05*numpy.std(ds))).astype(numpy.int8).clip(-2, 1) + 2).reshape( len(ds)//4, 4 )
    ds = numpy.sum((ds * [1, 4, 16, 64]).astype(numpy.int8), axis=1).astype(numpy.int8)
    return ds

run_start = time.time()
class Data_Reader(object):
    def __init__(self, beamformed, lo_freq, bandwidth):
        self.beamformed = beamformed
        self.lo_freq    = lo_freq
        self.bandwidth  = bandwidth
        self.ddc        = beam_red.ddc(lo_freq, abs(self.beamformed.bandwidth*2), abs(self.bandwidth*2))
        # start with empty 2 by 0 ndarray; we always get lst, usb from 1 lo
        self.data       = numpy.ndarray((2,0)) #self.read()

    @property
    def size(self):
        # data is always of shape [num_sidebands, num_timesamples]
        # with num_sidebands == 2 gives: [2, num_timesamples]
        # the .size attribute is used to figure out how many samples ther are
        return self.data.shape[1]

    def read(self):
        # stay here until the DDC has released at least one chunk of time series
        while True:
            spectral_data = next(self.beamformed)
            time_data     = beam_red.freq_to_time(spectral_data)
            filtered_data = self.ddc(time_data)
            if filtered_data is not None:
                return filtered_data

    def extend(self):
        #self.data = np.r_[self.data, self.read()]
        self.data = numpy.hstack([self.data, self.read()])
        return self.data

class QuantizeC(object):
    """
    uses precomiled C code to do the downsampling
    """
    def __init__(self, data_frame_size):
        # memory buffer for quantized data
        input_type = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
        output_type = c_ubyte * data_frame_size
        self.lib = npct.load_library("libquantize", ".")
        self.lib.quantize_samples.restype = None
        self.lib.quantize_samples.argtypes = [input_type, output_type, c_int, c_double]
        self.output_data = output_type()
        
    def do(self, input_data, threshold):
        assert input_data.dtype == np.double
        assert len(input_data) == len(self.output_data)*4
        self.lib.quantize_samples(np.ascontiguousarray(input_data),
                                  self.output_data,
                                  c_int(len(input_data)),
                                  c_double(threshold))
        
# VEX functions
def parse_value(text, unit):
    value_unit = text.split()
    assert len(value_unit) == 2
    assert value_unit[1] == unit
    return float(value_unit[0])

def parse_vex_time(text):
    vex_format = "%Yy%jd%Hh%Mm%Ss"
    return calendar.timegm(
            datetime.strptime(text, vex_format).timetuple())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script converts beamformer obs into "
        "one VDIF data stream.\n"
        "The parameters for conversion can be parsed from a VEX file, or "
        "overridden manually")

    parser.add_argument("-v","--verbose",action="store_true",default=False,
                        help="This produces verbose output")

    vex_group = parser.add_argument_group(title="VEX")
    vex_group.add_argument("-x", "--vex",
                        help="VEX file describing the recorded experiment")
    vex_group.add_argument("-n", "--scan",
                        help="Scan name to use, required if using a VEX file")
    vex_group.add_argument("-t", "--station", default="K7",
                        help="Station abbreviation to use")
    
    manual_group = parser.add_argument_group("manual")
    manual_group.add_argument("-s", "--start",
                              help="Start time of data conversion "
                              "in VEX Format (eg. 2016y063d12h51m00s)")
    manual_group.add_argument("-e", "--end",
                              help="End time of data conversion "
                              "in VEX Format (eg. 2016y063d12h51m00s)")
    manual_group.add_argument("-f", "--DDC-freq",
                              help="This is the top end of the ddc channel "
                              "in MHz")
    manual_group.add_argument("-r", "--sample_rate", default=None, type=float,
                              help="The output sample rate in MS/s")


    parser.add_argument("data_dir1",
                        help="Data directory of the first polarization")
    parser.add_argument("data_dir2",
                        help="Data directory of the second polarization")
    parser.add_argument("output",
                        help="Output file to dump VDIF frame, "
                        "has to be a new file")

    opts = parser.parse_args()

    start = None
    end = None
    
    if opts.vex is None:
        if (opts.DDC_freq is None) or (opts.sample_rate is None):
            parser.error("If VEX file isn't given, DDC freq and sample rate "
                         "are mandatory")
    else:
        if opts.scan is None:
            parser.error("If VEX is given, scan name is mandatory")
        # get output sample rate, DDC LO freq and start/end time from VEX
        from vex.vex import Vex
        vex = Vex(opts.vex)
        scan = vex["SCHED"][opts.scan]
        def find_section(sec):
            for line in vex["MODE"][scan["mode"]].getall(sec):
                if opts.station in line[1:]:
                    return vex[sec][line[0]]
            raise RuntimeError("Failed to find ${sec} section for {sta}".format(
                sec=sec, sta=opts.station))
        freq = find_section("FREQ")
        bbc = find_section("BBC")
        if_ = find_section("IF")

        # get sample rate form VEX file, first add up the channel sample rates
        # per channel per polarization
        
        # polarization -> total sample rate
        polarization_sample_rate = collections.defaultdict(float)
        sample_rate_per_channel = parse_value(freq["sample_rate"], "Ms/sec")
        if_polarization = {if_line[0]: if_line[2] for if_line in if_.getall(
            "if_def")}
        bbc_polarization = {bbc_line[0]: if_polarization[bbc_line[2]] \
                            for bbc_line in bbc.getall("BBC_assign")}
        for chan_def in freq.getall("chan_def"):
            polarization_sample_rate[bbc_polarization[chan_def[5]]] += \
                sample_rate_per_channel
        sample_rates = set(polarization_sample_rate.values())
        assert len(sample_rates) == 1, "Didn't find an unique sample rate"
        output_sample_rate = sample_rates.pop() * 1e6

        # ddc top end freq
        ddc_freq = 0
        for chan_def in freq.getall("chan_def"):
            sky_freq = parse_value(chan_def[1], "MHz")
            if chan_def[2] == "U":
                # add bandwidth if upper sideband
                sky_freq += parse_value(chan_def[3], "MHz")
            ddc_freq = max(ddc_freq, sky_freq)

        # scan start/end times
        start = parse_vex_time(scan["start"])
        for station_line in scan.getall("station"):
            if station_line[0] == opts.station:
                end = start + int(parse_value(station_line[2], "sec"))
        if end is None:
            raise RuntimeError("Failed to find station in scan")

    if opts.sample_rate is not None:
        output_sample_rate = opts.sample_rate * 1e6
    
    info1 = beam_red.Observation(opts.data_dir1)
    info2 = beam_red.Observation(opts.data_dir2)
    assert info1.centre_freq == info2.centre_freq
    if opts.DDC_freq is not None:
        ddc_freq = opts.DDC_freq
    #lo_freq = (info1.centre_freq + info1.bandwidth/2 - (float(ddc_freq) * 1e6)) # MHz -> Hz
    lo_freq = float(ddc_freq)*1e6 - (info1.centre_freq - info1.bandwidth/2)

    # compute the number of spectra per second and bail if it's not an integer number
    #spectra_per_second, remainder = divmod(info1.bandwidth, info1.n_chans) #390625
    #assert remainder==0
    spectra_per_second = fractions.Fraction( info1.bandwidth/info1.n_chans )
    assert spectra_per_second.denominator == 1, "Non-integer number of spectra per second"
    spectra_per_second = spectra_per_second.numerator
    assert info1.sync_time == info2.sync_time
    if opts.start is not None:
        start = parse_vex_time(opts.start)
    if opts.end is not None:
        end = parse_vex_time(opts.end)

    data_frame_size = 8000
    bits_per_sample = 2
    samples_per_frame = data_frame_size * 8 // bits_per_sample  # Python3 makes this Float 
    #assert int(samples_per_frame) == samples_per_frame
    #samples_per_frame = int(samples_per_frame)
    assert int(output_sample_rate)== output_sample_rate
    vdif_frames_per_second = fractions.Fraction(int(output_sample_rate), samples_per_frame)
    assert vdif_frames_per_second.denominator == 1, "Not an integer number of VDIF frames per second"
    vdif_frames_per_second  = vdif_frames_per_second.numerator

    #vdif_compatible_specnum = int(spectra_per_second / fractions.gcd(vdif_frames_per_second, spectra_per_second))
    vdif_compatible_specnum = spectra_per_second // fractions.gcd(vdif_frames_per_second, spectra_per_second)
        
    if start is None:
        # we must round to a spectra time stamp that is representable as a VDIF frame number
        #round_to = spectra_per_second 
        #round_to      = fractions.gcd(vdif_frames_per_second, spectra_per_second)
        start_specnum = max(info1.begin_spectra, info2.begin_spectra)
        # play it safe, round to a whole second
        #start_specnum = (start_specnum + spectra_per_second - 1) / spectra_per_second * spectra_per_second
        #start_specnum = ((start_specnum + round_to - 1) / round_to) * round_to
        #start_specnum = int(((start_specnum + vdif_compatible_specnum - 1) / vdif_compatible_specnum) * vdif_compatible_specnum)
        start_specnum = ((start_specnum + vdif_compatible_specnum - 1) // vdif_compatible_specnum) * vdif_compatible_specnum
    else:
        start_specnum = (start - info1.sync_time) * spectra_per_second
        assert (start_specnum >= info1.begin_spectra and start_specnum >= info2.begin_spectra), "start specnum: {ss}, begin spectra1: {bs1}, begin_spectra2: {bs2}".format(ss=start_specnum, bs1=info1.begin_spectra, bs2=info2.begin_spectra)

    if end is None:
        # we must round to a spectra time stamp that is representable as a VDIF frame number
        #round_to = spectra_per_second 
        end_specnum   = min(info1.end_spectra, info2.end_spectra)
        # play it safe, round to a whole second
        #start_specnum = (start_specnum + spectra_per_second - 1) / spectra_per_second * spectra_per_second
        #end_specnum   = ((end_specnum - round_to + 1) / round_to) * round_to
        #end_specnum = int(end_specnum / vdif_compatible_specnum) * vdif_compatible_specnum
        end_specnum = (end_specnum // vdif_compatible_specnum) * vdif_compatible_specnum
    else:
        end_specnum = (end - info1.sync_time) * spectra_per_second
        assert (end_specnum <= info1.end_spectra and end_specnum <= info2.end_spectra), "end specnum: {ss}, end spectra1: {bs1}, end_spectra2: {bs2}".format(ss=end_specnum, bs1=info1.end_spectra, bs2=info2.end_spectra)

    sss             = info1.seconds_since_sync(start_specnum)
    start_timestamp = int(sss)
    # split into integer time and fractional second
    start_framenr   = (sss - start_timestamp) * vdif_frames_per_second
    assert start_framenr.denominator == 1, "spectrum number {0} does not yield integer VDIF frame number".format(start_specnum)
    start_framenr = start_framenr.numerator
    #assert start_timestamp == int(start_timestamp), "not an integer second start time"

    beam1 = beam_red.read_beamformdata(
        info1, specnum=start_specnum, end_specnum=end_specnum,
        verbose=opts.verbose, raw=False)
    beam2 = beam_red.read_beamformdata(
        info2, specnum=start_specnum, end_specnum=end_specnum,
        verbose=opts.verbose,raw=False)

    data1 = None
    data2 = None
    def inc_header(header):
        header.frame_number += 1
        if header.frame_number >= vdif_frames_per_second:
            header.seconds += 1
            header.frame_number = 0

    #quantize = QuantizeC(data_frame_size)
    
    start_datetime = datetime.utcfromtimestamp(int(start_timestamp))
    epoch = datetime(year=start_datetime.year,
                     month=(1 if start_datetime.month<7 else 7),
                     day=1,
                     tzinfo=start_datetime.tzinfo)
    # epoch number is 1 per half year, counting from 2000
    epoch_number = (epoch.year - 2000) * 2 + (epoch.month // 6)
    seconds_since_epoch = (start_datetime - epoch).total_seconds()
    end_seconds_in_epoch = None
    if end is not None:
        end_seconds_in_epoch = (datetime.utcfromtimestamp(end) - epoch).total_seconds()

    def mk_header(thread):
        return VDIF_Header(seconds=int(seconds_since_epoch),
                           epoch=epoch_number,
                           thread_id=thread,
                           frame_length_8bytes=(data_frame_size+32)//8,
                           bits_per_sample_minus_1=bits_per_sample-1,
                           frame_number=start_framenr,
                           log2_channels=0)
    
    if opts.verbose:
        print("Run parameters: freq:", lo_freq, "samples/s", output_sample_rate, "start", start_timestamp, "end", end)

    def pct(rdr):
        return "{0:6.2f}".format(100.0*((rdr.spectrum_number - rdr.requested_start_spectrum)/(rdr.end_spectrum - rdr.requested_start_spectrum)))
    # open output file if it doesn't exist
    with os.fdopen(os.open(opts.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY),
                   "wb") as output:
        try:
            #threshold = 0.00095
            # multiplier values to "bit shift" 4 sample values to their position within byte
            #sample_scales = numpy.array([1, 4, 16, 64])
            # assume Nyquist sampling
            #bandwidth   = int(round(output_sample_rate / 2e6))
            bandwidth   = output_sample_rate // 2#e6
            reader1     = Data_Reader(beam1, lo_freq, bandwidth)
            reader2     = Data_Reader(beam2, lo_freq, bandwidth)
            threadState = dict()
            endSecond   = -1
            while True:
                if True: #opts.verbose:
                    print("appending data", time.time() - run_start,"s elapsed", pct(beam1),"% done")
                #while reader1.data.size < samples_per_frame:
                while reader1.size < samples_per_frame:
                    reader1.extend()
                #while reader2.data.size < samples_per_frame:
                while reader2.size < samples_per_frame:
                    reader2.extend()

                # We can compute how many frames' worth of data we have available
                nFrame          = (min(reader1.size, reader2.size) // samples_per_frame)
                # from there we can work out how many samples that are
                nSample         = nFrame * samples_per_frame
                # and also how many bytes that is
                nBytes          = nSample // 4
                if opts.verbose:
                    print("writing", nFrame, "frames", " [nSample:", nSample,"used",min(reader1.size, reader2.size),"avail] [nBytes=",nBytes,"] dT=",time.time() - run_start)
                # loop over lsb, usb per polarization, write out all frames
                thread = 0
                for reader in [reader1, reader2]:
                    for sb in [0,1]:
                        # compute quantization level for this dataset
#                        ds = reader.data[sb][:nSample]
                        # - divide by 1.05 stddev of the the samples
                        # - convert to 8 bit integers
                        # - anything < -1  changes to -1
                        #      id    >  2  changes to  2
                        # - add 1 to everything so now values are 0 .. 3
                        # - reshape to (n/4, 4)
#                        ds = ((ds // (1.05*numpy.std(ds))).astype(numpy.int8).clip(-2, 1) + 2).reshape( len(ds)//4, 4 )
#                        if opts.verbose:
#                            # print sampler stats
#                            hist = numpy.histogram(ds, [0, 1, 2, 3, 4])
#                            print("SamplerStats[Reader",1 if reader is reader1 else 2," SB",sb,"]: ",hist[0])
                        # the reshape is "handy" because now the 2nd axis has 4 successive samples in the range 0 .. 3
                        # - multiply by 2 ** 0, 2**2, 2**4, 2**6 to "bit shift" the samples to their location within the byte
                        # - then sum across that axis to find the byte value! Wheehee!
#                        ds = numpy.sum((ds * [1, 4, 16, 64]), axis=1).astype(numpy.int8)
                        ds = requantize( reader.data[sb][:nSample] )
                        # break up in chunks of data_frame_size (which is already in units of bytes!)
                        hdr = threadState.get(thread, None)
                        if hdr is None:
                            hdr = threadState.setdefault(thread, mk_header(thread))
                        for bIndex in range(0, nBytes, data_frame_size):
                            output.write(hdr)
                            output.write(ds[bIndex : bIndex+data_frame_size].tobytes())
                            #ds[bIndex:bIndex+data_frame_size].tofile(output)
                            inc_header( hdr )
                            endSecond = max(hdr.seconds, endSecond)
                        # wrote all frames for this sideband, move to next ie also next thread id
                        thread = thread + 1
                    # processed data from readerX for both sidebands
                    reader.data = reader.data[:,nSample:]
                # processed all data from all readers, check if we need to stop?
                if end_seconds_in_epoch is not None and endSecond >= end_seconds_in_epoch:
                    if opts.verbose:
                        print("End time reached")
                    break
        except StopIteration:
            if opts.verbose:
                print("End of data reached")

