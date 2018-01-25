from   __future__ import print_function
import numpy

def rdbeamform(fn):
    # code for reading n heaps:
    #a = numpy.fromfile(fn, dtype=numpy.int8, count=1*1024*128*2)
    #b = a.astype(numpy.float32)
    #c = b.view(numpy.complex64)
    #d = c.reshape(1, 1024, 128).transpose((0, 2, 1))
    with open(fn) as infile:
        # read one heap
        heapnum = 0
        while True:
            try:
                curHeap = numpy.fromfile(infile, dtype=numpy.int8, count=1024*128*2).astype(numpy.float32).view(numpy.complex64).reshape((1024,128)).T #transpose((1,0))
                for s in range(curHeap.shape[0]):
                    yield curHeap[s][::-1]
                heapnum = heapnum + 1
            except Exception as E:
                print("Caught exception reading heap #",heapnum," - ",E)
                raise StopIteration
