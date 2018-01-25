import tsig, calendar, datetime

class legacy_vdif_header(object):
    # assume data is list of 32-bit words
    def __init__(self, epoch, sec, frm, data):
        self.words = [0, 0, 0, 0]

        # legacy vdif + #-of-seconds since epoch
        self.word[0] |= (0x1<<30) # legacy bit
        self.word[0] |= sec & 0x3fffffff

        # framenumber within second and epoch
        self.word[1] |= frm & 0x00ffffff   
        self.word[1] |= ((epoch & 0x3f) << 24)

        # number of bytes in frame (including vdif header!) in units of 8-byte words
        self.word[2] |= ((len(data)*4+16)/8) & 0x00ffffff

        # 2log(nchan) is stored in 5 bits; bits 24:28 in word 2
        # we have 1 channel, thus 0 is just fine

        # "bits per sample - 1" is recorded in bits 26:30 (5 bits) of word 3
        # we do 2bps to we need to write '1'
        self.word[3] |= (0x1 << 26)
        # only one thread, thread 0. 10 bits in word 3; 16:25
        #self.word[3] |= (thrd & 0x3ff) << 16)


def quantizer(cutoff):
    cutoff = abs(cutoff)
    def do_it(sample):
        # <= -cutoff          => 00  (0)
        # -cutoff < samp <= 0 => 01  (1)
        # 0 < cutoff <= hi  => 10  (2)
        # > cutoff            => 11  (3)
        encoding = 0
        level    = -cutoff
        while sample>level:
            level    += cutoff
            encoding += 1
        return encoding
    return do_it


# generate a VDIF file
def wrvdif(fn, sig, sr, frmsz, tm):
    # Get start time as unix seconds
    t0    = calendar.timegm( tm.timetuple() )
    # make the time stamp of the epoch
    epoch = calendar.timegm( datetime.datetime(tm.year, (tm.month/6)*6, tm.day, 0, 0, 0, 0).timetuple() )
    # framesize in bytes * 8 = #bits, @2bits/sample = framesize*4 samples
    nsamp_frm   = frmsz*4         # number of samples per frame
    frm_second  = sr / nsamp_frm  # frames per second
    samplecount = 0
    frm         = 0
    sse         = t0 - epoch      # seconds since epoch
    with open(fn, 'w') as f:
        # get the samples
        data = map(quantize, sig.take(nsamp, sig))
       
        if samplecount and (samplecount % sr)==0:
            sse = sse + 1
            frm = 0
        else:
            frm = frm + 1

