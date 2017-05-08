import numpy as np

from .signal_processing import SignalProcessor

SPEED_OF_SOUND = 340.
TO_RAD = np.pi / 180
TO_DEG = 180 / np.pi

class TestsignalGenerator:
    """
    Helper class that generates test signals
    """

    def __init__(self, signal_processor=None):
        if signal_processor is None:
            self._sp = SignalProcessor()
        else:
            self._sp = signal_processor


    def create_sine(self, freq, length, fs=48000):
        """
        Create sine function array.

        freq: (Analog) Frequency in Hertz
        length: Length of the array in samples
        fs: sampling frequency

        Returns: (length, 1) - Numpy Array with sine signal
        """
        # compute discrete frequency for used samplerate
        if freq < 0 or freq >= fs/2.:
            raise ValueError("Frequency must be 0 <= freq < fs/2!")
        f = freq / fs
        n = np.arange(0, length, 1)
        return np.sin(2 * np.pi * f * n).reshape(length, 1)


    def plane_wave_testsignals(self, num_mics, freq, length, delta_t):
        """
        Generate test signals for the case of a plane wave

        num_mics: Number of microphones in the array
        freq: (Analog) frequency of the test sine
        length: length of the signals
        delta_t: time difference between the signals in two neighboured mics

        Returns: (length, num_mics) - Numpy Array with the test signals
                 The nth column contains the signal with a delay of n*delta_t
        """
        signals = np.concatenate([self.create_sine(freq, length) \
                                  for i in range(num_mics)], 1)
        self._sp.delay_signals_with_baseDelay(signals, delta_t)
        return signals[..., ::-1]
