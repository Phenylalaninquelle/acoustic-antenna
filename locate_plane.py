#!/usr/bin/env python3

"""
Locate a single sound source with the delay and sum algorithm
assuming the sound source emits plane waves.
"""

import argparse
import numpy as np

from delay_and_sum import DelayAndSumPlane
from delay_and_sum.cli_base import CliHandler

class CliPlaneHandler(CliHandler):
    """
    Command line handler class for plane wave case.
    """

    def __init__(self):
        self._max_angle = 90

    def main(self, filename, num_mics, arr_len, use_win):
        dx = arr_len / (num_mics - 1)
        s, fs = self.read_signals_from_wav(filename)
        das = DelayAndSumPlane(dx, num_mics, fs)
        rms_list = das.make_rms_list(s, window = use_win)
        self.plot_results(self._max_angle, rms_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file",
                        help="Wav-file containing the array signal.")
    parser.add_argument("numMics", type=int,
                        help="Number of microphones of the array")
    parser.add_argument("arrayLength", type=float,
                        help="Length of the microphone array in meters")
    parser.add_argument("-w", "--window", action="store_true",
                        help="Use Hann-Window for microphone weights")
    args = parser.parse_args()
    handler = CliPlaneHandler()
    handler.main(args.file, args.numMics, args.arrayLength, args.window)
