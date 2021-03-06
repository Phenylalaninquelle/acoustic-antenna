import numpy as np
from copy import deepcopy

from .signal_processing import SignalProcessor

SPEED_OF_SOUND = 340.
TO_RAD = np.pi / 180
TO_DEG = 180 / np.pi


class PointSourceHelper:
    """
    Helper class that encapsulates some computations for the point source
    generation/computation
    """

    def __new__(cls, *args, **kwargs):
        raise NotImplementedError("This class is not meant to be instantiated!\n" \
                                  "Just use its static methods.")

    def mic_positions(length, delta_x):
        """
        Generate array with microphone positions

        length: length of mic array in meters
        delta_x: distance between mics in meters

        returns: (N, 2) - array with the microphone position vectors
        """
        start_x = length / 2
        stop_x = -length / 2 - delta_x / 2
        mic_x = np.arange(start_x, stop_x, -delta_x)
        mic_y = np.array([0] * len(mic_x))
        mic_positions = np.vstack([mic_x, mic_y]).T
        return mic_positions

    def src_position(angle, dist):
        """
        Compute source position vector

        angle: angle of the source to the normal vector of the arrayin degrees
        dist: distance to source plane in meters

        returns: (1, 2) - array with the source position
        """
        return np.array([np.tan(angle * TO_RAD) * dist, dist])

    def mic_delays(mic_positions, src_position, fs, invert=True):
        """
        Compute inverse delay values with given microphone distances

        mic_positions: (N, 2) - array as given by mic_positions function
        src_position: (1, 2) - array as given by src_position function
        fs: sampling rate

        returns: (N) - array of inverse delays for delay and sum of
                 point sources in samples (!)
        """
        mic_dists = np.linalg.norm(mic_positions - src_position, axis=1)
        # normalise to only consider differences to the smallest distance
        mic_min = np.amin(mic_dists)
        mic_dists -= mic_min
        mic_delays = mic_dists / SPEED_OF_SOUND * fs
        # invert delays if requested
        if invert:
            max_delay = np.amax(mic_delays)
            mic_delays = np.abs(mic_delays - max_delay)
        return mic_delays

    def max_angle(length, distance):
        """
        Compute the maximum absolute-wise angle that can be detected for
        sound sources at the given distance

        length: length of mic array
        distance: distance to source plane in meters
        """
        return np.arctan(length / distance) * TO_DEG


class TestsignalGenerator:
    """
    Helper class that generates test signals
    """

    def __init__(self, signal_processor=None):
        """
        Initialise new TestsignalGenerator object

        signal_processor: SignalProcessor object,
                          if not given, create new one
        """
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

        returns: (length, 1) - Numpy Array with sine signal
        """
        if freq < 0 or freq >= fs/2.:
            raise ValueError("Frequency must be 0 <= freq < fs/2!")

        omega = 2 * np.pi * freq / fs
        n = np.arange(0, length, 1)
        return np.sin(omega * n).reshape(length, 1)

    def plane_wave_testsignals(self, num_mics, delta_t, signal):
        """
        Generate test signals for the case of a plane wave

        num_mics: Number of microphones in the array
        delta_t: time difference between the signals in two neighboured mics
        signal: mono (one channel) source signal as (length, 1) - array

        returns: (length, num_mics) - Numpy Array with the test signals
                 The nth column contains the signal with a delay of n*delta_t
        """
        signals = np.concatenate([deepcopy(signal) \
                                  for i in range(num_mics)], 1)
        self._sp.delay_signals_with_baseDelay(signals, delta_t)
        return signals[..., ::-1]

    def point_source_testsignal(self, angle, distance, das, signal):
        """
        Generate a test signal using a point source model

        angle: angle at which the point source resides
        distance: distance from source plane to array
        das: DelayAndSumPointSources object with the array data to use
        signal: mono (one channel) source signal as (length, 1) - array

        returns:
        """
        max_angle = PointSourceHelper.max_angle(das.length, distance)
        if angle > max_angle or angle < -max_angle:
            raise ValueError("The given angle is too big for the array.")

        signals = np.concatenate([deepcopy(signal) \
                                  for i in range(das.num_mics)], 1)

        src_pos = PointSourceHelper.src_position(angle, distance)
        mic_pos = PointSourceHelper.mic_positions(das.length, das.delta_x)
        mic_delays = PointSourceHelper.mic_delays(mic_pos, src_pos, das.fs, invert=False)

        for s, d in zip(signals.T, mic_delays):
            self._sp.delay_signal(s, np.round(d))

        return signals
