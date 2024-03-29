import struct, os, collections, datetime, functools

def mapfold(fn, lst, init):
    # fn(acc, res) => (acc, res)
    # init = initial accumulator
    def reductor( acc_results, element):
        (acc, results) = acc_results
        (newacc, res) = fn(acc, element)
        return (newacc, results+[res])
    return reduce(reductor, lst, (init, []))


class vdif_frame(object):
    def __init__(self, vdif_file, rdData=True):
        byt = vdif_file.read(16)
        l = struct.unpack("<4I", byt)
        self.hdr  = l[0:4]
        # skip 16 bytes if not legacy
        if not self.legacy():
            vdif_file.seek(16, os.SEEK_CUR)
        # read data array
        daSz = self.dataArraySize()
        if rdData:
            self.data = struct.unpack("<{0}I".format(daSz/4), vdif_file.read(daSz) )
        else:
            vdif_file.seek(daSz, os.SEEK_CUR)
            self.data = None

    def time(self, year=None):
        # vdif time stamp is 'seconds since the epoch'
        return self.hdr[0] & 0x3fffffff

    def legacy(self):
        # bit 30 in word 0 of the header
        return bool(self.hdr[0] & 0x40000000)

    def complex(self):
        # complex flex is bit 31 in word 3
        return bool(self.hdr[3] & 0x80000000)

    def framenr(self):
        # bits 0:23 in word 1 of the header
        return self.hdr[1] & 0x00ffffff

    def VDIFEpoch(self):
        # epoch is 6 bits in word 1 of the header, after
        # (or before) the 24 bit frame number within seconds
        return (self.hdr[1] >> 24) & 0x3f

    def frameSize(self):
        # data array size is encoded in the header as 
        # 'number of 8-byte words' in bits 0:23 of word 2
        return ((self.hdr[2] & 0x00ffffff) * 8)

    def dataArraySize(self):
        return self.frameSize() - (16 if self.legacy() else 32)

    def nChannels(self):
        # 2log(nchan) is stored in 5 bits; bits 24:28 in word 2
        return 2**((self.hdr[2]>>24) & 0x1f)

    def bitsPerSample(self):
        # "bits per sample - 1" is recorded in bits 26:30 (5 bits) of word 3
        # Fix broken "bps-1" value in some (old) VDIF files produced by an
        # erroneous version of jive5ab ... [if only we know who the author of
        # that PoS was!!!]
        bps = ((self.hdr[3]>>26) & 0x1f) + 1
        return bps
        #return (bps-1 if (bps % 2) else bps)

    def threadID(self):
        # 10 bits in word 3; 16:25
        return ((self.hdr[3]>>16) & 0x3ff)

    def stationID(self):
        # 16 bits that are either two characters or one 16-bit number
        # station ID is lower 16 bits in word 3 
        # return as string?
        return "{0}{1} [0x{2:X}]".format( chr(self.hdr[3]&0xff00), chr(self.hdr[3]&0xff), int(self.hdr[3]&0xffff) )
        
    def timeVex(self):
        # epoch counts number of half years since 2000
        (ny, nhalf) = divmod(self.VDIFEpoch(), 2)
        (y, m, d)   = (2000 + ny, (6*nhalf)+1, 1)
        (y, m, d, h, m, s, dow, doy, tz)  \
                    = (datetime.datetime(y, m, d) + datetime.timedelta( float(self.time())/86400.0 )).timetuple() if \
                      (y>0 and m>0 and d>0) else [0]*9
        return "{0:04d}y{1:03d}d{2:02d}h{3:02d}m{4:02d}s".format( y, doy, h, m, s )

    def deChannelize(self):
        rv         = collections.defaultdict(list)
        numChannel = self.nChannels()
        chOffset   = 0
        nSampByte  = 8 / self.bitsPerSample()
        LUT        = luts[ self.bitsPerSample() ]

        if self.data is None:
            return rv
        # decode in chunks of numChannel
        # i.e. decode this thread into numChannel channels,
        #      but there may be other threadIDs too so we assume
        #      same #-channels-per-thread for all threads in this
        #      VDIF stream
        curWord    = 0
        curChannel = 0
        chanOffset = self.threadID()*numChannel
        while curWord<len(self.data):
            woord = self.data[curWord]
            for curByte in range(4):
                decodedChannels = LUT[ woord & 0xff ]
                # Loop over the samples in this byte
                for i in range(nSampByte):
                    rv[ chanOffset + curChannel ].append( decodedChannels[i] )
                    curChannel = (curChannel+1) % numChannel
                woord >>= 8
            curWord = curWord+1
        print( "Decoded frame:" )
        for (k, v) in rv.iteritems():
            print( "  Ch[{0}]: {1} samples".format(k, len(v)) )
        return rv


    def __str__(self):
        return self.timeVex()+"/{0:05d}".format(self.framenr())
        #return str(self.time()) + "/" + ("%05d" % self.framenr())

def frames(filenm, n, rddata):
    if n is None:
        n = -1
    with open(filenm, "rb") as f:
        cnt = 0
        while (n<0 or cnt<n):
            vf = vdif_frame( f, rddata )
            yield vf
            cnt += 1

############################
##
##  Print frame information
##
############################
def rdfile(fn, n=None):
    for frm in frames(fn, n, False):
        flags = []
        if frm.legacy():
            flags.append("Legacy")
        if frm.complex():
            flags.append("Complex")
        print( "{0} THRD:{1} NCH:{2}@{3}bps DATA:{4}".format(str(frm), frm.threadID(), frm.nChannels(), frm.bitsPerSample(), frm.dataArraySize()),
               "(", ",".join(flags), ")" )


if __name__ == "__main__":
    import sys
    functools.reduce(lambda acc, x: acc, map(rdfile, sys.argv[1:]), None)
