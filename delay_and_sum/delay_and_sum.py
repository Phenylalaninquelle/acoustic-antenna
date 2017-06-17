import numpy as np
from copy import deepcopy

from .signal_processing import SignalProcessor
from ._helper import SPEED_OF_SOUND
from ._helper import TO_RAD
from ._helper import TO_DEG
from ._helper import PointSourceHelper


class DelayAndSum:
    """
    Base class for the delay and sum algorithm implementations.

    This is not intended to be instanciated just use the child classes.
    """

    def __init__(self, delta_x, num_mics, fs, sig_proc=None):
        """
        Initialise new DelayAndSum object.

        delta_x: distance between the microphones of the array izn meters
        num_mics: number of microphones in the array
        fs: sampling frequency in Hertz
        sp SignalProcessor object,
                          if not given, create new one
        """
        self.delta_x = delta_x
        self.num_mics = num_mics
        self.fs = fs
        self._sp = SignalProcessor() if sig_proc is None else sig_proc

    def __repr__(self):
        desc = "<{cls} Object with {nm} mics with distance of {dx}, fs: {fs}>"
        return desc


class DelayAndSumPlane(DelayAndSum):
    """
    Offers methods for the delay and sum algorithm to realise
    an acoustical antenna. This implementation assumes the
    incoming sound wave to be a plane wave.
    """

    def __init__(self, delta_x, num_mics, fs, sig_proc=None):
        super(DelayAndSumPlane, self).__init__(delta_x, num_mics, fs, sig_proc)

    def __repr__(self):
        desc = super().__repr__()
        classname = self.__class__.__name__
        return desc.format(cls=classname, nm=self.num_mics, dx=self.delta_x, fs=self.fs)

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
            delta_t *= self.fs
        return delta_t

    def make_rms_list(self, signals, start_angle=-90, stop_angle=90, angle_steps=1,
                      window=False):
        """
        Perform delay & sum algorithm for a given set of microphone signals
        to compute an array of rms values for given angles (default: -90 to 90)

        signals: numpy array containing the microphone signals
                 this has to be (L x N) array, with L being the
                 length of the signals and N being the number of signals
        start_angle: start at this angle
        stop_angle: compute up to this angle
        angle_steps: steps between angles
        window: boolean flag that indicates to use a window function

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

        w = self._sp.hann_window(signals.shape[1])
        rms_values = []

        for angle in range(start_angle, stop_angle + 1, angle_steps):
            signals_tmp = deepcopy(signals)
            if window:
                signals_tmp *= w
            delay = self.delta_t_for_angle(angle, in_samples=True)
            self._sp.delay_signals_with_baseDelay(signals_tmp, delay)
            rms_values.append(self._sp.get_rms(signals_tmp.sum(1)))

        return self._sp.to_db(rms_values)


class DelayAndSumPointSources(DelayAndSum):
    """
    Offers methods for the delay and sum algorithm to realise
    an acoustical antenna. This class handles point sources
    arranged on a plane in front of the mic array.
    """

    def __init__(self, delta_x, num_mics, fs, sig_proc=None):
        super(DelayAndSumPointSources, self).__init__(delta_x, num_mics, fs, sig_proc)
        self.length = self.delta_x * (self.num_mics - 1)

    def __repr__(self):
        desc = super().__repr__()
        classname = self.__class__.__name__
        return desc.format(cls=classname, nm=self.num_mics, dx=self.delta_x, fs=self.fs)

    def max_angle(self, distance):
        """
        Return the maximum angle to try for this array geometry.
        This is needed to avoid the x-coordinate of a sources position approaching
        to infinity when the angle gets near +/- 90°

        distance: distance to the source plane in meters
        """
        return int(np.round(PointSourceHelper.max_angle(self.length, distance)))

    def make_rms_list(self, signals, distance, window=False):
        """
        Compute RMS values for all valid positions on the sources
        positions plane.

        signals: numpy array containing the microphone signals
                 this has to be (L x N) array, with L being the
                 length of the signals and N being the number of signals
        distance: distance to the source plane in meters
        window: boolean flag that indicates to use a window function

        returns: list of rms values for the angles from <start_angle> to
                 <stop_angle> in <angle_steps>
        """
        N = signals.shape[1]

        if N != self.num_mics:
            msg = "Number of given signals must equal the specified number of" \
            "microphones({}, given: {})"
            raise ValueError(msg.format(self.num_mics, N))

        if distance <= 0:
            msg = "Distance to source plane must be bigger than zero!"
            raise ValueError(msg)

        max_angle = self.max_angle(distance)
        angles = np.arange(-max_angle, max_angle + 1)
        mic_positions = PointSourceHelper.mic_positions(self.length, self.delta_x)

        rms_values = []
        w = self._sp.hann_window(signals.shape[1])
        for ang in angles:
            src_pos = PointSourceHelper.src_position(ang, distance)
            mic_delays = PointSourceHelper.mic_delays(mic_positions, src_pos, self.fs)
            sigs_tmp = deepcopy(signals)
            if window:
                sigs_tmp *= w
            for s, d in zip(sigs_tmp.T, mic_delays):
                self._sp.delay_signal(s, np.round(d))
            rms_values.append(self._sp.get_rms(sigs_tmp.sum(1)))

        return self._sp.to_db(rms_values)
