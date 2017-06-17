import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import sys

class CliHandler:
    """
    Static namespace class for cli handling
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

    def main():
        raise NotImplementedError("This needs to be overwritten in subclass")
