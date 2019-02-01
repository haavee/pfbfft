#!/usr/bin/env python
# print the time range inferred from a beamformer .h5 file
# this HAS to happen first
from __future__ import print_function
# and next we MUST do this before we include anything else to make sure all time
# handling is done in GMT
import os, time
os.environ['TZ'] = ''
time.tzset()
# Right. NOW we can move on with our lives
import beam_red_meerkat as beam_red
import argparse, math, calendar, fractions, collections, re
from   datetime import datetime

stripper = re.compile(r'^[0-9]*').sub
asvex    = lambda ts, precision=5: "{tm:%Yy%jd%Hh%Mm%S}{ss}s".format(tm=datetime.fromtimestamp(float(ts)), ss=stripper("", "{0:0.{precision}f}".format(float(ts), precision=precision)))


def print_time_range(h5name):
    info = beam_red.Observation(h5name)
    print(h5name,":",asvex(info.sync_time+info.seconds_since_sync(info.begin_spectra)," -> ",info.sync_time+info.seconds_since_sync(info.end_spectra)))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script prints the time range of MeerKAT beamformer .h5 observation(s)")

    parser.add_argument("files", nargs='+', help="MeerKAT beamformer .h5 file name(s)")

    opts = parser.parse_args()
    _    = map(print_time_range, opts.files)
