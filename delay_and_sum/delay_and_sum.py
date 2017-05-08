import numpy as np
from copy import deepcopy

from .signal_processing import SignalProcessor
from ._helper import SPEED_OF_SOUND, TO_RAD

class DelayAndSumPlane:
    """
    Offers methods for the delay and sum algorithm to realise
    an acoustical antenna. This implementation assumes plane waves.
    """

    def __init__(self, delta_x, num_mics, fs, signal_processor=None):
        """
        Initialise new object.

        delta_x: distance between the microphones of the array
        num_mics: number of microphones in the array
        fs: sampling frequency
        signal_processor: SignalProcessor object
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
        """
        angle = self._delta_x * np.sin(TO_RAD * angle) / SPEED_OF_SOUND
        return angle


    def make_rms_list(self, signals, start_angle=-90, stop_angle=90, angle_steps=1):
        """
        Perform delay & sum algorithm for a given set of microphone signals
        to compute an array of rms values for the angles from -90 to 90 degrees
        """
        # TODO: error checking and stuff

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

        return rms_values
