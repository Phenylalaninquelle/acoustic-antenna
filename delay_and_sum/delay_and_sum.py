import numpy as np
from copy import deepcopy

from .signal_processing import SignalProcessor
from ._helper import SPEED_OF_SOUND, TO_RAD, TO_DEG


class DelayAndSum:
    """
    Base class for the delay and sum algorithm implementations.

    This is not intended to be instanciated just use the child classes.
    """

    def __init__(self, delta_x, num_mics, fs, signal_processor=None):
        """
        Initialise new DelayAndSum object.

        delta_x: distance between the microphones of the array izn meters
        num_mics: number of microphones in the array
        fs: sampling frequency in Hertz
        signal_processor: SignalProcessor object,
                          if not given, create new one
        """
        self.delta_x = delta_x
        self.num_mics = num_mics
        self.fs = fs
        if signal_processor is None:
            self._sp = SignalProcessor()
        else:
            self._sp = signal_processor

    def __repr__(self):
        desc = "<{cls} Object with {nm} mics with distance of {dx}, fs: {fs}>"
        return desc


class DelayAndSumPlane(DelayAndSum):
    """
    Offers methods for the delay and sum algorithm to realise
    an acoustical antenna. This implementation assumes the
    incoming sound wave to be a plane wave.
    """

    def __init__(self, delta_x, num_mics, fs, signal_processor=None):
        super(DelayAndSumPlane, self).__init__(delta_x,
                                               num_mics,
                                               fs,
                                               signal_processor)

    def __repr__(self):
        desc = super().__repr__()
        classname = self.__class__.__name__
        return desc.format(cls=classname,
                           nm=self.num_mics,
                           dx=self.delta_x,
                           fs=self.fs)


    def delta_t_for_angle(self, angle, in_samples=False):
        """
        Compute time delay between two adjacent microphones.

        angle: angle of incoming wave in degrees
        in_samples: if True, return delay in samples

        returns: delta_t value as float
        """
        if angle < -90 or angle > 90:
            raise ValueError("Angle must be in [-90, 90]!")

        delta_t = self.delta_x * np.sin(TO_RAD * angle) / SPEED_OF_SOUND
        if in_samples:
            return delta_t * self.fs
        else:
            return delta_t


    def make_rms_list(self, signals, start_angle=-90, stop_angle=90, angle_steps=1,
                      window = False):
        """
        Perform delay & sum algorithm for a given set of microphone signals
        to compute an array of rms values for given angles (default: -90 to 90)

        signals: numpy array containing the microphone signals
                 this has to be (L x N) array, with L being the
                 length of the signals and N being the number of signals
        start_angle: start at this angle
        stop_angle: compute up to this angle
        angle_steps: steps between angles

        returns: list of rms values for the angles from <start_angle> to
                 <stop_angle> in <angle_steps>
        """
        N = signals.shape[1]

        if N != self.num_mics:
            msg = "Number of given signals must equal the specified number of" \
                  "microphones({}, given: {})"
            raise ValueError(msg.format(self.num_mics, N))

        if start_angle > stop_angle or stop_angle - start_angle < angle_steps:
            raise ValueError("Given angle range not valid")


        rms_values = []
        delay_if_pos = lambda n: self.num_mics - n
        delay_if_neg = lambda n: n - 1
        w = self._sp.hann_window(signals.shape[1])

        # go through all angles and delay and sum
        for angle in range(start_angle, stop_angle + 1, angle_steps):
            signals_tmp = deepcopy(signals)
            if window:
                signals_tmp *= w
            delta_t = self.delta_t_for_angle(angle)
            delay = delta_t * self.fs

            self._sp.delay_signals_with_baseDelay(signals_tmp, delay)

            # sum up everything and add rms value of sum to the list
            rms_values.append(self._sp.get_rms(signals_tmp.sum(1)))

        return self._sp.to_db(rms_values)

class DelayAndSumPointSources(DelayAndSum):
    """
    Offers methods for the delay and sum algorithm to realise
    an acoustical antenna. This class handles point sources
    arranged on a plane in front of the mic array.
    """

    def __init__(self, delta_x, num_mics, fs, signal_processor=None):
        super(DelayAndSumPointSources, self).__init__(delta_x,
                                                      num_mics,
                                                      fs,
                                                      signal_processor)
        self._length = self.delta_x * (self.num_mics - 1)


    def __repr__(self):
        desc = super().__repr__()
        classname = self.__class__.__name__
        return desc.format(cls=classname,
                           nm=self.num_mics,
                           dx=self.delta_x,
                           fs=self.fs)


    def max_angle(self, distance):
        """
        Compute the maximum absolute-wise angle that can be detected for
        sound sources at the given distance

        distance: distance to source plane in meters
        """
        return np.arctan(self._length / distance) * TO_DEG


    def make_rms_list(self, signals, distance):
        """
        Compute RMS values for all valid positions on the sources
        positions plane.
        """
        N = signals.shape[1]

        if N != self.num_mics:
            msg = "Number of given signals must equal the specified number of" \
            "microphones({}, given: {})"
            raise ValueError(msg.format(self.num_mics, N))

        if distance <= 0:
            msg = "Distance to source plane must be bigger than zero!"
            raise ValueError(msg)

        max_angle = int(np.round(self.max_angle(distance)))
        angles = np.arange(-max_angle, max_angle + 1)
        mic_x = np.arange(self._length / 2,
                          -self._length / 2 - self.delta_x / 2,
                          -self.delta_x)
        mic_positions = np.vstack([mic_x,
                                  np.array([distance] * len(mic_x))]).T

        rms_values = []
        for ang in angles:
            # compute source position from angle
            src_pos = np.array([np.tan(ang * TO_RAD) * distance, 0])
            # compute mic distances as length of vector difference to source position
            mic_distances = np.linalg.norm(mic_positions - src_pos, axis=1)
            # subtract the global minimum distance from all distances
            mic_min = np.amin(mic_distances)
            mic_distances -= mic_min
            # get delay (in samples!!)
            mic_delays = mic_distances / SPEED_OF_SOUND * self.fs
            # calculate the inverse delays for this constellation
            max_delay = np.amax(mic_delays)
            mic_delays = np.abs(mic_delays - max_delay)
            # delay the signals accordingly
            sigs_tmp = deepcopy(signals)
            for d, s in zip(mic_delays, sigs_tmp):
                self._sp.delay_signal(s, d)
            # sum up everything and add rms value of sum to the list
            rms_values.append(self._sp.get_rms(signals_tmp.sum(1)))

        return self._sp.to_db(rms_values)
