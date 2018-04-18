#!/usr/bin/env python
# this HAS to happen first
from __future__ import print_function
# and next we MUST do this before we include anything else to make sure all time
# handling is done in GMT
import os, time
os.environ['TZ'] = ''
time.tzset()
# Right. NOW we can move on with our lives
from vdif_header import VDIF_Header
import beam_red_meerkat as beam_red

import numpy 
import numpy.ctypeslib as npct

import argparse, math, calendar, fractions, collections, re
from datetime import datetime
from ctypes import c_ubyte, c_int, c_double
#from numba import jit

stripper = re.compile(r'^[0-9]*').sub
asvex    = lambda ts, precision=5: "{tm:%Yy%jd%Hh%Mm%S}{ss}s".format(tm=datetime.fromtimestamp(float(ts)), ss=stripper("", "{0:0.{precision}f}".format(float(ts), precision=precision)))

#@jit
def requantize(ds):
    ds = ((ds // (1.05*numpy.std(ds))).astype(numpy.int8).clip(-2, 1) + 2).reshape( len(ds)//4, 4 )
    ds = numpy.sum((ds * [1, 4, 16, 64]).astype(numpy.int8), axis=1).astype(numpy.int8)
    return ds

run_start = time.time()
class Data_Reader(object):
    def __init__(self, beamformed, lo_freq, bandwidth, sample_offset=0):
        self.beamformed = beamformed
        self.lo_freq    = lo_freq
        self.bandwidth  = bandwidth
        self.skip_samps = sample_offset
        self.ddc        = beam_red.ddc(lo_freq, abs(self.beamformed.bandwidth*2), abs(self.bandwidth*2))
        # start with empty 2 by 0 ndarray; we always get lsb, usb from 1 lo
        self.data       = numpy.ndarray((2,0)) #self.read()

    @property
    def size(self):
        # data is always of shape [num_sidebands, num_timesamples]
        # with num_sidebands == 2 gives: [2, num_timesamples]
        # the .size attribute is used to figure out how many samples ther are
        return self.data.shape[1]

    def read(self):
        # stay here until the DDC has released at least one chunk of time series
        time_data = numpy.ndarray((0))
        while True:
            spectral_data = next(self.beamformed)
            time_data     = numpy.hstack([time_data, beam_red.freq_to_time(spectral_data)])
            # make sure we have more samples than we need to skip before we skip
            if len(time_data)<self.skip_samps:
                continue
            # skip samples, but only do that once
            if self.skip_samps:
                print("DataReader/seek to start sample ",self.skip_samps)
                time_data       = time_data[self.skip_samps:]
                self.skip_samps = 0
            # Now we can send it onwards to the DDC and wait for
            # it to release a chunk
            filtered_data = self.ddc(time_data)
            if filtered_data is not None:
                return filtered_data

    def extend(self):
        #self.data = np.r_[self.data, self.read()]
        self.data = numpy.hstack([self.data, self.read()])
        return self.data

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
    assert int(output_sample_rate) == output_sample_rate, "Requested output sample rate is not integer"
    
    info1 = beam_red.Observation(opts.data_dir1)
    info2 = beam_red.Observation(opts.data_dir2)
    assert info1.centre_freq == info2.centre_freq, "The observations' centre frequency does not match"
    if opts.DDC_freq is not None:
        ddc_freq = opts.DDC_freq

    # The mixing freq is the requested freq minus the dc freq
    lo_freq = float(ddc_freq)*1e6 - info1.dc_freq 

    # compute the number of spectra per second and bail if it's not an integer number
    assert info1.spectra_p_s == info2.spectra_p_s, "The number of spectra per second is not equal for the observations"
    spectra_per_second = info1.spectra_p_s
    assert info1.sync_time == info2.sync_time
    if opts.start is not None:
        start = parse_vex_time(opts.start)
    if opts.end is not None:
        end = parse_vex_time(opts.end)

    data_frame_size        = 4000
    bits_per_sample        = 2
    samples_per_frame      = data_frame_size * 8 // bits_per_sample  # Python3 makes this Float 
    vdif_frames_per_second = fractions.Fraction(int(output_sample_rate), samples_per_frame)
    frameperiod            = 1/vdif_frames_per_second
    assert vdif_frames_per_second.denominator == 1, "Not an integer number of VDIF frames per second"
    print(" VDIF frame period: ", frameperiod)

    # if no start time given, take it from the observation that started last
    # note that from here on we convert the start time to seconds since syncing
    if start is None:
        start = info1.seconds_since_sync( max(info1.begin_spectra, info2.begin_spectra) )
    else:
        # info1.sync_time == info2.sync_time (asserted above) so we can Just Do This(tm)
        start -= info1.sync_time

    # we must round the start time to a spectra time stamp that is representable as a VDIF frame number
    # Note: we have asserted before that both sync_time + spectra_per_second are
    #       identical between info1 and info2. Thus we only need to convert the
    #       requested time stamp to spectrum number once.
    vdif_compat_start = frameperiod * ((start // frameperiod) + 1)
    # specnum_from_time() will return tuple( integer-spectrum-number, sample-offset-within-timeseries-from-spectrum )
    # so basically indicating at which sample to start exactly
    start_specnum     = info1.specnum_from_time(vdif_compat_start)
    # Because we cannot guarantee that both observations actually start/end with the same spectra we must validate
    assert info1.begin_spectra <= start_specnum[0] < info1.end_spectra, \
            "Requested start time out of bounds of datafile 1: {0} <= {1} < {2}".format(info1.begin_spectra, start_specnum[0], info1.end_spectra)
    assert info2.begin_spectra <= start_specnum[0] < info2.end_spectra, \
            "Requested start time out of bounds of datafile 2: {0} <= {1} < {2}".format(info2.begin_spectra, start_specnum[0], info2.end_spectra)

    # Repeat for the end time (only use the observation that finished first ...)
    if end is None:
        end = info1.seconds_since_sync( min(info1.end_spectra, info2.end_spectra) )
    else:
        end -= info1.sync_time

    vdif_compat_end = frameperiod * (end // frameperiod)
    end_specnum     = info1.specnum_from_time(vdif_compat_end)
    assert info1.begin_spectra < end_specnum[0] <= info1.end_spectra, \
            "Requested end time out of bounds of datafile 1: {0} < {1} <= {2}".format(info1.begin_spectra, end_specnum[0], info1.end_spectra)
    assert info2.begin_spectra < end_specnum[0] <= info2.end_spectra, \
            "Requested end time out of bounds of datafile 2: {0} < {1} <= {2}".format(info2.begin_spectra, end_specnum[0], info2.end_spectra)

    print(" decided on start_specnum=", start_specnum)
    print("       => ", asvex(vdif_compat_start+info1.sync_time) )
    print(" decided on end_specnum=", end_specnum)
    print("       => ", asvex(vdif_compat_end+info1.sync_time) )

    # Need to get VDIF integer second + vdif start frame number
    # vdif_compat_start still with respect to sync time, does not matter for framenr within second
    start_timestamp = int(vdif_compat_start)
    start_framenr   = (vdif_compat_start - start_timestamp)/frameperiod
    assert start_framenr.denominator == 1, \
            "VDIF compat start {0} does not yield integer VDIF frame number?!".format(asvex(vdif_compat_start + info1.sync_time))
    start_framenr   = start_framenr.numerator
    # But now we must really convert back to full time stamp
    start_timestamp += info1.sync_time
    end             += info1.sync_time
    print(" Start UNIX time stamp   = ", asvex(start_timestamp))
    print(" Start VDIF frame number = ", start_framenr)

    # The start/end spectra numbers can be propagated to the beamform readers
    beam1 = beam_red.read_beamformdata(
        info1, specnum=start_specnum[0], end_specnum=end_specnum[0],
        verbose=opts.verbose, raw=False, readheaps=1024)
    beam2 = beam_red.read_beamformdata(
        info2, specnum=start_specnum[0], end_specnum=end_specnum[0],
        verbose=opts.verbose,raw=False, readheaps=1024)

    data1 = None
    data2 = None
    def inc_header(header):
        header.frame_number += 1
        if header.frame_number >= vdif_frames_per_second:
            header.seconds += 1
            header.frame_number = 0

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

    print("EPOCH: ",epoch.isoformat(),"  => sse=", seconds_since_epoch, " end_seconds_in_epoch=", end_seconds_in_epoch)
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
            bandwidth   = output_sample_rate // 2#e6
            # The data readers must be informed about how many samples to initially skip 
            # to end up at the sample to actually start working with!
            reader1     = Data_Reader(beam1, lo_freq, bandwidth, sample_offset=start_specnum[1])
            reader2     = Data_Reader(beam2, lo_freq, bandwidth, sample_offset=start_specnum[1])
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
                # tid      pol     sb   sb_vex
                # 0        H       LSB   USB   [KAT7 data is LSB so this chunk is 'high end' of the spectrum, so USB wrt LO]
                # 1        H       USB   LSB   [ ..          ..                   'low  end'    ...              LSB wrt LO]
                # 2        V       LSB   USB
                # 3        V       USB   LSB
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

