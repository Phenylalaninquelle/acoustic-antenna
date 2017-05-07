import numpy as np

class SignalProcessor:
    """
    Helper class that offers signal processing related
    methods
    """

    def __init__(self):
        pass


    def get_rms(self, sig):
        """
        Compute root mean square value for a given signal

        sig: numpy array with the signal values

        Returns: the rms value of the signal
        """
        return np.sqrt( 1./len(sig) * np.square(sig).sum())


    def to_db (self, val):
        """
        Computes decibels from linear value
        """
        return 20 * np.log10(val)


    def delay_signal(self, signal, delay):
        """
        Delay the given signal IN PLACE (!).

        signal: Numpy-Array (1-dim) with the signal values
        delay: delay in samples
        """
        if delay != 0:
            if np.trunc(delay) != delay:
                raise NotImplementedError("Only whole-sample delay for now!")
            if delay < 0 or delay > len(signal):
                raise ValueError("Delay must be 0 <= delay <= signalLength!")

            signal[delay:] = signal[:-delay]
            signal[:delay] = 0
