#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a testfile for the Delay And Sum script
using a single mono .wav file
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from os.path import exists

from delay_and_sum import TestsignalGenerator, \
                          DelayAndSumPlane

def read_signal_from_wav(filename):
    s, fs = sf.read(filename)
    # allow only one-channel files
    try:
        s = s.reshape(s.shape[0], 1)
    except:
        raise RuntimeError("Can only use mono (1-channel) files!")
    return s, fs


def write_signal(filename, signal, fs):
    # ask user before overwriting an existing file
    if exists(filename):
        msg = "File {} already exists. Overwrite? (y/n)"
        choice = input(msg.format(filename))
        do_overwrite = {'y': True,
                   'yes': True,
                   'j': True,
                   'ja': True,
                   'n': False,
                   'no': False,
                   'nein': False}
        if not do_overwrite[choice]:
            print("Abort!")
            return

    sf.write(filename, signal, fs)


def make_testsignal(delta_x, num_mics, s, fs, angle):
    das = DelayAndSumPlane(delta_x, num_mics, fs)
    tg = TestsignalGenerator()
    delta_t = das.delta_t_for_angle(angle, in_samples=True)
    return tg.plane_wave_testsignals(num_mics, delta_t, s)


def main(filename, array_len, num_mics, angle):
    try:
        s, fs = read_signal_from_wav(filename)
    except RuntimeError as e:
        msg = str(e)
        print("An error occured while reading input file: {}".format(msg))
        return

    delta_x = array_len / (num_mics - 1)
    testsignals = make_testsignal(delta_x, num_mics, s, fs, angle)
    # create new filename
    add = "_{}deg_{}mics_{}m.wav".format(angle, num_mics, array_len)
    filename = filename.split('.wav')[0] + add
    try:
        write_signal(filename, testsignals, fs)
    except Exception as e:
        msg = str(e)
        print("An error occured while writing the output file: {}".format(msg))


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
    main(args.file, args.arrayLength, args.numMics, args.angle)
