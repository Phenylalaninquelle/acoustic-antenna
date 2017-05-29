#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Delay and Sum example that reads an array output signal
from a .wav file and draws a graph to illustrate the
source location process.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from delay_and_sum import DelayAndSumPlane

def find_angle(signals, array_len, num_mics, fs, useWin):
    delta_x = array_len / (num_mics - 1)
    das_plane = DelayAndSumPlane(delta_x, num_mics, fs)
    rms_list = das_plane.make_rms_list(signals, window=useWin)
    return rms_list

def read_signals_from_wav(filename):
    return sf.read(filename)

def main(filename, array_len, num_mics, useWin):
    s, fs = read_signals_from_wav(filename)
    rms_list = find_angle(s, array_len, num_mics, fs, useWin)
    max_val = np.amax(rms_list)
    angle = np.argmax(rms_list) - 90

    # plot RMS values and mark the maximum
    plt.figure()
    xmin = -90
    xmax = 90
    plt.plot(range(xmin, xmax + 1, 1), rms_list)
    plt.grid()
    plt.xlabel(r"$\alpha / °$")
    plt.ylabel(r"$s / dB$")
    plt.xlim(xmin, xmax + 1)
    plt.title("Source found at: {}°".format(angle))
    plt.vlines(angle, 0, max_val, 'r', '--')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file",
                        help="Wav-file containing the array signal.")
    parser.add_argument("numMics", type=int,
                        help="Number of microphones of the array")
    parser.add_argument("arrayLength", type=float,
                        help="Length of the microphone array in meters")
    parser.add_argument("-w", action="store_true",
                        help="Use Hann-Window for microphone weights")
    args = parser.parse_args()
    main(args.file, args.arrayLength, args.numMics, args.w)
