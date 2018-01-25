import numpy, fractions

class Spectrum(numpy.ndarray):
    def __new__(cls, input_array, **kwargs):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = numpy.asarray(input_array).view(cls)
        # add the new attributes to the created instance
        obj.__dict__ = kwargs
        # Finally, we must return the newly created object:
        if not (obj.DCEdge == 0 or obj.DCEdge == (obj.n_chan - 1)):
            raise AssertionError("The DCEdge {0} is not 0 or {1}".format(obj.DCEdge, obj.n_chan-1))
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is not None:
            self.__dict__ = getattr(obj, '__dict__', dict())

    @property
    def n_chan(self):
        return self.shape[0]

    @property
    def sideband(self):
        #  USB:   DC .... n
        #         lo     hi   [BW > 0, DCEdge == 0]
        #
        #         0 .... DC
        #         hi     lo   [BW < 0, DCEdge == n_chan]
        #
        #  LSB:   0      DC    
        #         lo ... hi   [BW > 0, DCEdge == n_chan]
        #
        #         DC ...  n
        #         hi     lo   [BW < 0, DCEdge == 0]
        #
        # In short: we need to figure out if the DCEdge is the highest or the lowest frequency
        #           if that is true, then we have USB else LSB
        return "USB" if self.DCEdge == (0 if self.BandWidth > 0 else self.n_chan-1) else "LSB"

    #  ch_freq = DCFrequency + (index - DCEdge) * BandWidth/n_chan
    def ch_freq(self, *args):
        return ((numpy.arange(self.n_chan) - self.DCEdge) * self.BandWidth/self.n_chan) + self.DCFrequency

    def ch_freq_org(self, *args):
        return ((numpy.arange(self.n_chan)[slice(*args)] - self.DCEdge) * self.BandWidth/self.n_chan) + self.DCFrequency

    # make sure BW>0 such that freq increases to the right
    def normalize(self):
        if self.BandWidth<0:
            self.BandWidth = -self.BandWidth
            self.DCEdge    = (self.n_chan - 1) - self.DCEdge
            self           = numpy.conj( self[::-1] )
        return self


# where there's a spectrum there's also a timeseries
class TimeSeries(numpy.ndarray):
    def __new__(cls, input_array, **kwargs):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = numpy.asarray(input_array).view(cls)
        # add the new attributes to the created instance
        obj.__dict__   = kwargs
        # convert samplerate to fraction
        obj.SampleRate = fractions.Fraction(obj.SampleRate)
        if not (obj.SampleRate>0):
            raise AssertionError("Invalid samplerate given to time series")
        if not hasattr(obj, 'TimeStamp'):
            raise AssertionError("Time series does not have a TimeStamp")
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is not None:
            self.__dict__ = getattr(obj, '__dict__', dict())

