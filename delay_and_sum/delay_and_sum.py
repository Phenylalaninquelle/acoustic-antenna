import numpy as np
from copy import deepcopy

from .signal_processing import SignalProcessor
from ._helper import SPEED_OF_SOUND, TO_RAD

class DelayAndSumPlane:
    """
    Offers methods for the delay and sum algorithm to realise
    an acoustical antenna. This implementation assumes the
    incoming sound wave to be a plane wave.
    """

    def __init__(self, delta_x, num_mics, fs, signal_processor=None):
        """
        Initialise new DelayAndSumPlane object.

        delta_x: distance between the microphones of the array izn meters
        num_mics: number of microphones in the array
        fs: sampling frequency in Hertz
        signal_processor: SignalProcessor object,
                          if not given, create new one
        """
        self._delta_x = delta_x
        self._num_mics = num_mics
        self._fs = fs
        if signal_processor is None:
            self._sp = SignalProcessor()
        else:
            self._sp = signal_processor


    def __repr__(self):
        desc = "<DelayAndSum Object with {nm} mics with distance of {dx}, fs: {fs}>"
        return desc.format(nm=self._num_mics, dx=self._delta_x, fs=self._fs)


    def delta_t_for_angle(self, angle):
        """
        Compute time delay between two adjacent microphones.

        angle: angle of incoming wave in degrees

        returns: delta_t value as float
        """
        if angle < -90 or angle > 90:
            raise ValueError("Angle must be in [-90, 90]!")

        delta_t = self._delta_x * np.sin(TO_RAD * angle) / SPEED_OF_SOUND
        return delta_t


    def make_rms_list(self, signals, start_angle=-90, stop_angle=90, angle_steps=1):
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
        L = signals.shape[0]
        N = signals.shape[1]

        if N != self._num_mics:
            msg = "Number of given signals must equal the specified number of" \
                  "microphones({}, given: {})"
            raise ValueError(msg.format(self._num_mics, N))

        if start_angle > stop_angle or stop_angle - start_angle < angle_steps:
            raise ValueError("Given angle range not valid")


        rms_values = []
        delay_if_pos = lambda n: self._num_mics - n
        delay_if_neg = lambda n: n - 1

        # go through all angles and delay and sum
        for angle in range(start_angle, stop_angle + 1, angle_steps):
            signals_tmp = deepcopy(signals)
            delta_t = self.delta_t_for_angle(angle)
            delay = delta_t * self._fs

            self._sp.delay_signals_with_baseDelay(signals_tmp, delay)

            # sum up everything and add rms value of sum to the list
            rms_values.append(self._sp.get_rms(signals_tmp.sum(1)))

        return self._sp.to_db(rms_values)
