import numpy, ppgplot, sys, time

icomplex16 = numpy.dtype({'names': ['real', 'imag'], 'formats': [numpy.int8, numpy.int8]})


def draw_one(x):
    ppgplot.pgbbuf()
    ppgplot.pgenv(0, len(x), numpy.min(x)*0.95, numpy.max(x)*1.05)
    ppgplot.pgpt(numpy.arange(len(x)), x)
    ppgplot.pgebuf()
    time.sleep(4)

def rdspec(fn, quant):
    #z = numpy.fromfile(fn, dtype=numpy.cfloat)
    z = numpy.fromfile(fn, dtype=numpy.complex64)
    z = quant(z.reshape((1024, 128)))
    for tm in range(128): #range(127, 0, -1):
        print "Spectrum #",tm
        yield z[:,tm]

def rdbeamform(fn, quant):
    #a = np.fromfile(fn, dtype=np.int8, count=readheaps_curr*info['n_chans']*128*2) 
    #a = a.reshape(readheaps_curr,info['n_chans'],128,2).transpose((0,2,1,3)).reshape(readheaps_curr*128,info['n_chans'],2)
    a = numpy.fromfile(fn, dtype=numpy.int8, count=1*1024*128*2)
    a = a.reshape(1,1024,128,2).transpose((0,2,1,3)).reshape(1*128,1024,2)
    q = a.astype(numpy.float32)
    q = quant(q.view(numpy.complex64))
    for tm in range(q.shape[1]):
        print "Beamformed spec #",tm
        yield  q[...,tm]

def rdbeamform_nt(fn, quant):
    #a = np.fromfile(fn, dtype=np.int8, count=readheaps_curr*info['n_chans']*128*2) 
    #a = a.reshape(readheaps_curr,info['n_chans'],128,2).transpose((0,2,1,3)).reshape(readheaps_curr*128,info['n_chans'],2)
    a = numpy.fromfile(fn, dtype=numpy.int8, count=1*1024*128*2)
    a = a.reshape(1*128,1024,2)
    q = a.astype(numpy.float32)
    q = quant(q.view(numpy.complex64).reshape(128, 1024))
    print q.shape
    for tm in range(q.shape[0]):
        print "Beamformed spec(nt) #",tm
        yield  q[tm,...]

def rdbeamform_hv(fn, quant):
    #a = np.fromfile(fn, dtype=np.int8, count=readheaps_curr*info['n_chans']*128*2) 
    #a = a.reshape(readheaps_curr,info['n_chans'],128,2).transpose((0,2,1,3)).reshape(readheaps_curr*128,info['n_chans'],2)
    a = numpy.fromfile(fn, dtype=numpy.int8, count=1*1024*128*2)
    b = a.astype(numpy.float32)
    c = b.view(numpy.complex64)
    d = c.reshape(1, 1024, 128).transpose((0, 2, 1))
    e = quant(d)
    print e.shape
    for tm in range(e.shape[1]):
        print "beamformed spec(hv) #",tm
        yield  e[0,tm,...]
        #raise StopIteration

def rdbeamform_hv_avg(fn, quant):
    #a = np.fromfile(fn, dtype=np.int8, count=readheaps_curr*info['n_chans']*128*2) 
    #a = a.reshape(readheaps_curr,info['n_chans'],128,2).transpose((0,2,1,3)).reshape(readheaps_curr*128,info['n_chans'],2)
    a = numpy.fromfile(fn, dtype=numpy.int8, count=1*1024*128*2)
    b = a.astype(numpy.float32)
    c = b.view(numpy.complex64)
    d = c.reshape(1, 1024, 128).transpose((0, 2, 1))
    #e = quant(d)
    print "raw data shape=",d.shape
    e = quant(numpy.average(d, axis=1))
    f = numpy.average(quant(d), axis=1)
    print "vector avg (shape=",e.shape,")"
    yield e[0,...]
    print "scalar avg (shape=",f.shape,")"
    yield f[0,...]


def sum_it_impl(ar, quant):
    # shape = (nheap, nspec, nchan)
    # we sum over nspec
    a2 = numpy.sum(quant(ar), axis=1)
    # now shape = (nheap, nchan)
    # we reorganize as:
    # (nheap, 1, nchan)
    a2.shape = ( (a2.shape[0], 1, a2.shape[1]) )
    print "a2's shape = ",a2.shape
    return a2

sum_it = lambda quant: (lambda ar: sum_it_impl(ar, quant))

#proc_file = lambda quant: lambda fn: map(draw_one, rdspec(fn, quant))
#proc_file = lambda quant: lambda fn: map(draw_one, rdbeamform(fn, quant))
#proc_file = lambda quant: lambda fn: map(draw_one, rdbeamform_nt(fn, quant))
proc_file = lambda quant: lambda fn: map(draw_one, rdbeamform_hv(fn, quant))
#proc_file = lambda quant: lambda fn: map(draw_one, rdbeamform_hv_avg(fn, quant))
#ppgplot.pgopen("42/xw")
#ppgplot.pgopen("/tmp/amplitude-1spectrum.ps/cps")
#ppgplot.pgopen("/tmp/amplitude-1spectrum.png/png")
ppgplot.pgopen("?")
ppgplot.pgask(False)
#map(proc_file(numpy.abs), sys.argv[1:])
map(proc_file(sum_it(numpy.abs)), sys.argv[1:])
#map(proc_file(lambda x: numpy.angle(x, True)), sys.argv[1:])
