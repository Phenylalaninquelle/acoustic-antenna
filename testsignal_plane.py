#!/usr/bin/env python3

"""
Create an artificial testsignal for the delay and sum algorithm
from a mono audio file assuming plane wave propagation.
"""

import argparse
import numpy as np

from delay_and_sum import DelayAndSumPlane
from delay_and_sum import TestsignalGenerator
from delay_and_sum.cli_base import CliHandler

class CliPlaneTestsignalHandler(CliHandler):
    """
    Command line handler class for plane wave testsignal generation.
    """

    def __init__(self):
        self._tg = TestsignalGenerator()

    def main(self, filename, array_len, num_mics, angle):
        s, fs = self.read_signals_from_wav(filename)
        sig_shape = s.shape
        if len(sig_shape) == 1:
            s = s.reshape((len(s), 1))
        elif sig_shape[1] != 1:
            print("Only mono signals can be used!")
            return

        dx = array_len / (num_mics - 1)
        das = DelayAndSumPlane(dx, num_mics, fs)
        dt = das.delta_t_for_angle(angle, in_samples=True)
        testsigs = self._tg.plane_wave_testsignals(num_mics, dt, s)

        add = "_{}deg_{}mics_{}m.wav".format(angle, num_mics, array_len)
        filename = filename.split('.wav')[0] + add
        self.write_signal_to_wav(filename, testsigs, fs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file",
                        help="Mono (!!) Wav-file containing the signal to form the testsignale from")
    parser.add_argument("numMics", type=int,
                        help="Number of microphones of the array")
    parser.add_argument("arrayLength", type=float,
                        help="Length of the microphone array in meters")
    parser.add_argument("angle", type=int,
                        help="Angle to place signal at")
    args = parser.parse_args()
    handler = CliPlaneTestsignalHandler()
    handler.main(args.file, args.arrayLength, args.numMics, args.angle)
