#!/usr/bin/env python3

"""
Locate a single sound source with the delay and sum algorithm
assuming the sound source emits spherical waves.
"""

import argparse
import numpy as np

from delay_and_sum import DelayAndSumPointSources
from delay_and_sum.cli_base import CliHandler

class CliPlaneHandler(CliHandler):

    def __init__(self):
        self._max_angle = None


    def main(self, filename, num_mics, arr_len, dist, use_win):
        dx = arr_len / (num_mics - 1)
        s, fs = self.read_signals_from_wav(filename)
        das = DelayAndSumPointSources(dx, num_mics, fs)
        self._max_angle = das.max_angle(dist)
        rms_list = das.make_rms_list(s, dist, use_win)
        self.plot_results(self._max_angle, rms_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file",
                        help="Wav-file containing the array signal.")
    parser.add_argument("numMics", type=int,
                        help="Number of microphones of the array")
    parser.add_argument("arrayLength", type=float,
                        help="Length of the microphone array in meters")
    parser.add_argument("distance", type=int,
                        help="Distance to the source plane")
    parser.add_argument("-w", "--window", action="store_true",
                        help="Use Hann-Window for microphone weights")
    args = parser.parse_args()
    handler = CliPlaneHandler()
    handler.main(args.file, args.numMics, args.arrayLength, args.distance, args.window)
