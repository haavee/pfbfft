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
from   datetime  import datetime
from   functools import partial, reduce

def compose(*fns)         : return lambda x: reduce(lambda acc, f: f(acc), reversed(fns), x)
def Map(fn)               : return partial(map, fn)
def asvex(ts, precision=5): return "{tm:%Yy%jd%Hh%Mm%S}{ss}s".format(tm=datetime.fromtimestamp(float(ts)), ss=stripper("", "{0:0.{precision}f}".format(float(ts), precision=precision)))
drain    = partial(collections.deque, maxlen=0)
stripper = re.compile(r'^[0-9]*').sub
#asvex    = lambda ts, precision=5: "{tm:%Yy%jd%Hh%Mm%S}{ss}s".format(tm=datetime.fromtimestamp(float(ts)), ss=stripper("", "{0:0.{precision}f}".format(float(ts), precision=precision)))

def print_time_range(h5name):
    info = beam_red.Observation(h5name)
    print(h5name,":",asvex(info.sync_time+info.seconds_since_sync(info.begin_spectra))," -> ",asvex(info.sync_time+info.seconds_since_sync(info.end_spectra)))

main     = compose(drain, Map(print_time_range))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script prints the time range of MeerKAT beamformer .h5 observation(s)")

    parser.add_argument("files", nargs='+', help="MeerKAT beamformer .h5 file name(s)")
    main( parser.parse_args().files )
