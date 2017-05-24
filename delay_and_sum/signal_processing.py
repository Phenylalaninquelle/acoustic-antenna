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

        returns: the rms value of the signal
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

            delay = int(delay)
            signal[delay:] = signal[:-delay]
            signal[:delay] = 0

    def delay_signals_with_baseDelay(self, signals, base_delay):
        """
        Take an (N,m) array with signals (m being the number of signals
        and N being the length of the signals) and create a constant
        delay (of 'base_delay') between the signals.

        signals: numpy array with the signals stacked horizontally
        base_delay: delay to create between the microphones in samples
                    (if not an integer, value will be rounded),
                    if this is positive the last sigal in the array will
                    have a delay of zero and the signal at position zero
                    will have a delay of (N-1)*base_delay (other way round
                    for negative base_delay)
        """
        num_mics = signals.shape[1]
        if base_delay > 0:
            delay_for_mic = lambda n: num_mics - n
        elif base_delay < 0:
            delay_for_mic = lambda n: n - 1
        else:
            # for angle of zero no delay is needed
            return

        for n in range(1, num_mics + 1):
            # compute delay in samples
            delay = np.round(delay_for_mic(n) * base_delay)
            self.delay_signal(signals[:,n - 1], np.abs(delay))
