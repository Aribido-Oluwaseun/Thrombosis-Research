import soundfile as sf
import numpy as np
import pylab as pl
import glob, os
import matplotlib.pyplot as plt
from scipy import signal as sg
import scipy as sp
import wave
import struct

FILE_PATH = '/home/joseph/Desktop/2017Research/LVAD_Dataset'
FILE_TYPE = '.wav'

class DataException(Exception):
    "This class prints exceptions for the Extract Data class"

class ProcessDataException(Exception):
    "This class prints exceptions for the Process Data class"

class ExtractData(object):
    "This class extracts data from the object file"
    def __init__(self):
        "Define some initial parameters"

        self.files = self._get_filenames(FILE_PATH)
        self.params = [self._get_params(os.path.join(FILE_PATH, x)) for x in
                       self.files]
        # note that the result of wave.getparams(signal) is a tuple. We
        # cannot modify the content since it is a list. We need to convert it
        #  to a list to make it more flexible.

        self.params = [list(x) for x in self.params]
        minframelength = []
        # Set the length of all extracted frames to the minimum frame length
        #  for all the files for uniformity. The frame length happens to be
        # the 4th item in the result: wave.getparams()

        for each_list in self.params:
            minframelength.append(each_list[3])
        minframelength = min(minframelength)
        self.params[0][3] = minframelength
        self.params = self.params[0]


    def _get_filenames(self, dir):
        filenames = os.listdir(dir)
        filenames = [x for x in filenames if x.endswith('.wav')]
        filenames.sort()
        return filenames


    def _struct_format(self, sample_width, nb_samples):
        return {1: "%db", 2: "<%dh", 4: "<%dl"}[sample_width] % nb_samples

    def _read_data(self, filename, nframes):
        wv = wave.open(filename, 'r')
        data = wv.readframes(nframes=nframes)
        if data:
            sample_width = wv.getsampwidth()
            nsamples = len(data) // sample_width
            format = self._struct_format(sample_width, nsamples)
            wv.close()
            return struct.unpack(format, data)
        else:
            raise DataException('No Data content found in file: %s' %(filename))

    def _get_params(self, filename):
        wv = wave.open(filename, 'r')
        return wv.getparams()

    def extract_all_data(self, dir, data_dict=False, data_matrix=False):
        data = {}
        data_mat = np.empty(shape=(len(self.files), self.params[3]))
        i = 0
        for each_file in self.files:
            tempf = self._read_data(os.path.join(dir, each_file),
                                    self.params[3])
            data[each_file] = tempf
            data_mat[i][:]  = tempf
            i += 1

        if data_dict and data_matrix is True:
            raise DataException('Please specify only one data type.')

        elif data_dict == True:
            return data

        elif  data_matrix == True:
            return data_mat

        else:
            raise DataException('No Data Type specified')

class ProcessData():
    """This class helps us visualize data and reduce it's dimensions"
     Make class objects at initialization
    """

    def __init__(self):
        dir = FILE_PATH
        self.extract_data = ExtractData()
        self.data_mat = self.extract_data.extract_all_data(dir,
                                                           data_matrix=True)
        self.filenames = self.extract_data._get_filenames(FILE_PATH)
        self.data = self.extract_data.extract_all_data(dir, data_dict=True)
        self.params = self.extract_data.params

    def _resample(self, newrate=None):
        """Re-sample Data"""

        if newrate is None:
            newrate = 10000
        resampled_data = np.empty(shape=(len(self.data_mat), newrate))
        for i in range(len(self.data_mat)):
            resampled_data[i,:] = signal.resample(self.data_mat[i,:], newrate)
        return resampled_data

    def normalize_data(self):
        """Convert Data to Zero-mean, Unit-Variance"""

        sampled_data = self.data_mat
        for each_row in range(len(self.data_mat)):
            sampled_data[each_row, :] = sampled_data[each_row, :] - np.mean(
                sampled_data[each_row, :])
            sampled_data[each_row, :] = sampled_data[each_row, :] / np.std(
                sampled_data[each_row, :])
        return sampled_data


    def  filter(self, signal, Fs, cutoff, type, order=9, fig=False):
        if type not in ['lowpass', 'highpass', 'bandpass']:
            print('Please specify: lowpass, highpass or bandpass')
            raise ProcessDataException('Invalid Filter Specified')
        Wn = 2 * cutoff / Fs
        if type in ['lowpass', 'highpass']:
            [b,a] = sg.butter(N=order, Wn=Wn, btype=type, output='ba')

        if type is 'bandpass' and isinstance(type) is list and len(type) is 2:
            [b, a] = sg.butter(N=order, Wn=Wn, btype=type, output='ba')

        if fig:
            w, h = signal.freqs(b, a)
            plt.plot(w, 20 * np.log10(abs(h)))
            plt.xscale('log')
            plt.title('Butterworth filter frequency response')
            plt.xlabel('Frequency [radians / second]')
            plt.ylabel('Amplitude [dB]')
            plt.margins(0, 0.1)
            plt.grid(which='both', axis='both')
            plt.axvline(100, color='green')  # cutoff frequency
            plt.show()
        return sg.filtfilt(b=b, a=a, x=signal)

    def _fft(self):
        """fft on data"""

        fft_result = {}
        for each_name in self.filenames:
            fft_result[each_name] = np.abs(np.fft.fft(self.data[each_name],
                                               self.params[3],
                                                      norm='ortho'))

        freq = np.fft.fftfreq(self.params[3])*self.params[2]
        for each_file in self.filenames:
            plt.plot(freq, fft_result[each_file])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('FFT of %s' % (each_file))
            plt.title('Normalized DFT (FFT) %s' % (each_file))
            plt.show()


    def _visualize_data(self, tone=None, hist=None, fft=None):
        """Help Visualize Data"""
        title = ['1787.wav No Thrombosis', '2353.wav No Thrombosis',
                 '4095.wav Possible Thrombosis', '7346 Likely Thrombosis',
                 '7452.wav Has Thrombosis', '7645.wav Thrombosis Resolved',
                 '7838.wav Has Thrombosis', '7976.wav Thrombosis Resolved']

        if tone:
            for i in range(len(self.data)):
                normalized_data = self.normalize_data()
                plt.plot(normalized_data[i, :])
                plt.title(title[i])
                plt.show()

        if hist:
            for i in range(len(self.data)):
                normalized_data = self.normalize_data()
                plt.hist(normalized_data[i, :], 100)
                plt.title(title[i])
            plt.show()

        if fft:
            self._fft()


def main():
    """Calls Classes and methods to implement logid"""
    extract_data = ProcessData()
    extract_data._visualize_data(fft=True)



if __name__ == "__main__":
    main()