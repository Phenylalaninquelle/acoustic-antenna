import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import sys
import os

class CliHandler:
    """
    Base class for cli handler
    """

    def __init__(self):
        raise NotImplementedError("Subclass this!!")

    def plot_results(self, max_ang, rms_list):
        """
        Plot the directional function of a microphone array
        """
        plt.figure()
        xmin = -max_ang
        xmax = max_ang + 1
        max_val = np.amax(rms_list)
        src_angle = np.argmax(rms_list) - max_ang
        ang_range = range(xmin, xmax, 1)
        plt.plot(ang_range, rms_list)
        plt.grid()
        plt.xlabel(r"$\alpha / °$")
        plt.ylabel(r"$R / dB$")
        plt.xlim(xmin, xmax)
        plt.title("Source found at: {}°".format(src_angle))
        plt.vlines(src_angle, 0, max_val, 'r', '--')
        plt.show()

    def read_signals_from_wav(self, filename):
        """
        Safely read an audio signal from a .wav file.
        If an error occurs while reading the file the whole program is terminated.

        filename: path of the file

        returns: signal and sampling rate
        """
        try:
            s, fs = sf.read(filename)
            return s, fs
        except RuntimeError as e:
            msg = 'An error occured while reading the file {}:\n"{}"'
            print(msg.format(filename, str(e)))
            sys.exit()

    def write_signal_to_wav(self, filename, signal, fs):
        """
        Safely write an audio signal to a .wav file.
        If the file already exists the user is asked before the file is overwritten.
        An error during writing terminates the programm

        filename: path to write to
        signal: signal as (L x c) numpy array, with L begin the length of the signal
                and c being the channel count of the signal
        fs: sampling rate
        """
        if os.path.exists(filename):
            msg = "File {} already exists. Overwrite? (y/n)"
            choice = input(msg.format(filename))
            do_overwrite = {'y': True, 'yes': True, 'j': True, 'ja': True,
                            'n': False, 'no': False, 'nein': False}
            if not do_overwrite[choice.lower()]:
                print("Writing aborted!")
                return

        try:
            sf.write(filename, signal, fs)
        except RuntimeError as e:
            msg = 'An error occured while wrting the file {}:\n"{}"'
            print(msg.format(filename, str(e)))
            sys.exit()


    def main():
        raise NotImplementedError("This needs to be overwritten in subclass")
