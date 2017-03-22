import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal as sg
from scipy.stats import skew, kurtosis
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

    def run(self):
        order = 15
        filter_signal = {}
        cutoff = 100 # new Nyquist to recover signal would be 200 Hz

        data = self.normalize_data() # Normalize to zero mean (although heart
        #  beat data is non-stationary) and Normalized  Variance

        filter_result = self.filter_lowpass(order=order, signal=data,
                                            cutoff=cutoff)
        new_fs = 250 # Hz
        window_size = 500  # Corresponding to 2 seconds of heart beat using
        # a new sampling rate of 250 after filtering.
        window = sg.get_window('boxcar', int(self.params[3]/window_size))
        f,t,Sxx = self.spectrogram(signal=data, Fs=new_fs,
                                              window=window)
        skew_result = self.skew(data)
        kurtosis_result = self.kurtosis(signal=data)
        print [kurtosis_result[x] for x in kurtosis_result]
        plt.plot(f['2353.wav'], Sxx['2353.wav'])
        plt.title('1787.wav Spectrogram')
        plt.show()
        #self._visualize_data(filter_result,hist=True)
        #self._fft(filter_result,True)

    def normalize_data(self):
        """Convert Data to Zero-mean, Unit-Variance"""
        normal_data = {}
        for each_file in self.data:
            normal_data[each_file] = self.data[each_file] - np.mean(
                self.data[each_file])
            normal_data[each_file] = self.data[each_file] / np.std(
                self.data[each_file])
        return normal_data

    def filter_lowpass(self, order, signal, cutoff):
        filter_result = {}
        Wn = float(cutoff)*2/self.params[2]
        b,a = sg.butter(N=order, Wn=Wn, btype='lowpass')
        for each_file in signal:
            filter_result[each_file] = sg.lfilter(b, a, signal[each_file])
        return filter_result

    def spectrogram(self, signal, Fs, window):
        frequency_sample = {}
        segment_time = {}
        spectrogram_result = {}
        for each_file in signal:
            frequency_sample[each_file], segment_time[each_file], \
            spectrogram_result[each_file] = sg.spectrogram(x=signal[each_file],
                                            fs=Fs, window=window)
        return frequency_sample, segment_time, spectrogram_result

    def skew(self, signal):
        skew_result = {}
        for each_file in signal:
            skew_result[each_file] = skew(a=signal[each_file])
        return skew_result

    def kurtosis(self, signal):
        kurtosis_result = {}
        for each_file in signal:
            kurtosis_result[each_file] = kurtosis(a=signal[each_file])
        return kurtosis_result

    def _fft(self, signal, fig=False):
        """fft on data"""

        fft_result = {}
        freq = np.fft.fftfreq(self.params[3]) * self.params[2]
        for each_name in signal:
            fft_result[each_name] = np.abs(np.fft.fft(signal[each_name],
                                               self.params[3]))
            if fig:
                plt.plot(freq, fft_result[each_name])
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('FFT of %s' % (each_name))
                plt.title('Normalized DFT (FFT) %s' % (each_name))
                plt.show()
        return fft_result

    def _visualize_data(self, data, tone=None, hist=None):
        """Help Visualize Data"""
        title = ['1787.wav No Thrombosis', '2353.wav No Thrombosis',
                 '4095.wav Possible Thrombosis', '7346 Likely Thrombosis',
                 '7452.wav Has Thrombosis', '7645.wav Thrombosis Resolved',
                 '7838.wav Has Thrombosis', '7976.wav Thrombosis Resolved']
        count = 0
        if tone:
            for i in data:
                plt.plot(data[i])
                plt.title(title[count])
                plt.show()
                count += 1

        count = 0
        if hist:
            for i in data:

                plt.hist(data[i], 100)
                plt.title(title[count])
                plt.show()
                count += 1


def main():
    """Calls Classes and methods to implement log id"""
    extract_data = ProcessData()
    extract_data.run()



if __name__ == "__main__":
    main()