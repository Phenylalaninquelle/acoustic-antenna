#!/usr/bin/env python3

"""
Create an artificial testsignal for the delay and sum algorithm
from a mono audio file assuming plane wave propagation.
"""

import argparse
import numpy as np

from delay_and_sum import DelayAndSumPointSources
from delay_and_sum import TestsignalGenerator
from delay_and_sum.cli_base import CliHandler

class CliPointTestsignalHandler(CliHandler):
    """
    Command line handler class for plane wave testsignal generation.
    """

    def __init__(self):
        self._tg = TestsignalGenerator()

    def main(self, filename, array_len, num_mics, angle, distance):
        s, fs = self.read_signals_from_wav(filename)

        dx = array_len / (num_mics - 1)
        das = DelayAndSumPointSources(dx, num_mics, fs)
        testsigs = self._tg.point_source_testsignal(angle, distance, das, s)

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
    parser.add_argument("distance", type=float,
                        help="Distance to source plane in meters")
    args = parser.parse_args()
    handler = CliPointTestsignalHandler()
    handler.main(args.file, args.arrayLength, args.numMics, args.angle, args.distance)
