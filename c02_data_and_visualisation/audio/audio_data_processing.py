import librosa
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt

class AudioInfo:
    def __init__(self, filepath, sr=44100, n_fft=512, hop_length=256, keep_audio=False, keep_aux=False):
        if filepath.split('.')[-1] in ['wav', 'mp3']:
            self.name = filepath.split('.')[-2].split(os.sep)[-1]
            print('filepath: ', filepath)
            self.audio, sr = librosa.load( filepath, sr=sr )
            self.sr = sr
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.extract_power_spectrum()
            self.make_useful_audio_mask()
            self.make_useful_spectrum()
            self.make_useful_area_features()
            self.assign_category()
            if not keep_audio:
                del self.audio
            if not keep_aux:
                del self.power_spectrum
                del self.useful_spectrum
                del self.spectral_magnitude
                del self.useful_bandwidth
                del self.useful_centroid
                del self.useful_mask
        else:
            print('bad format')
    # end __init__

    def extract_power_spectrum(self):
        p = librosa.stft(self.audio, n_fft=self.n_fft, hop_length=self.hop_length)
        self.spectral_magnitude, _ = librosa.magphase(p)
        self.power_spectrum = librosa.amplitude_to_db( np.abs(p), ref=np.max )
    # end extract_power_spectrum
    
    def make_useful_audio_mask(self):
        self.rms = librosa.feature.rms(S=self.spectral_magnitude, frame_length=self.n_fft)
        rms = self.rms[0]
        self.useful_mask = np.zeros( rms.size )
        self.useful_mask[ rms > 0.005 ] = 1
        self.useful_mask = self.useful_mask.astype(int)
    # end extract_power_spectrum
    
    def make_useful_spectrum(self):
        self.useful_spectrum = self.power_spectrum[:,self.useful_mask == 1]
    # end extract_power_spectrum
    
    def plot_spectrum(self, range_low=20, range_high=5000):
        fig , plt_alias =  plt.subplots()
        librosa.display.specshow(self.power_spectrum, sr=self.sr, x_axis='time', y_axis='linear', ax=plt_alias)
        plt_alias.set_ylim([range_low, range_high])
    # end plot_spectrum
    
    def plot_save_spectrum(self, figure_file_name='test.png', range_low=20, range_high=5000):
        fig , plt_alias =  plt.subplots()
        librosa.display.specshow(self.power_spectrum, sr=self.sr, x_axis='time', y_axis='linear', ax=plt_alias)
        plt_alias.set_ylim([range_low, range_high])
        plt.savefig( figure_file_name , dpi=300 )
    # plot_save_spectrum
    
    def make_features(self):
        # centroid
        c = librosa.feature.spectral_centroid(self.audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        self.centroid = c[0]
        self.mean_centroid = np.mean( self.centroid )
        self.std_centroid = np.std( self.centroid )
        # bandwidth
        b = librosa.feature.spectral_bandwidth(self.audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        self.bandwidth = b[0]
        self.mean_bandwidth = np.mean( self.bandwidth )
        self.std_bandwidth = np.std( self.bandwidth )
        self.features = np.reshape( [self.mean_centroid, self.std_centroid, self.mean_bandwidth, self.std_bandwidth], (4,1) )
    # end make_features
    
    def make_useful_area_features(self):
        # centroid
        c = librosa.feature.spectral_centroid(self.audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        self.useful_centroid = c[0][ self.useful_mask == 1 ]
        self.mean_centroid = np.mean( self.useful_centroid )
        self.std_centroid = np.std( self.useful_centroid )
        # bandwidth
        b = librosa.feature.spectral_bandwidth(self.audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        self.useful_bandwidth = b[0][ self.useful_mask == 1 ]
        self.mean_bandwidth = np.mean( self.useful_bandwidth )
        self.std_bandwidth = np.std( self.useful_bandwidth )
        self.features = np.reshape( [self.mean_centroid, self.std_centroid, self.mean_bandwidth, self.std_bandwidth], (4,1) )
    # end make_useful_area_features
    
    def assign_category(self):
        if 'kick' in self.name.lower() or 'bass' in self.name.lower():
            self.category = 'kick'
        elif 'snare' in self.name.lower():
            self.category = 'snare'
        elif 'tom' in self.name.lower():
            self.category = 'tom'
        elif 'hat' in self.name.lower() or 'hh' in self.name.lower():
            self.category = 'hihat'
        elif 'bell' in self.name.lower():
            self.category = 'bell'
        elif 'conga' in self.name.lower():
            self.category = 'conga'
        elif 'bongo' in self.name.lower():
            self.category = 'bongo'
        elif 'ride' in self.name.lower():
            self.category = 'ride'
        elif 'ride' in self.name.lower():
            self.category = 'ride'
        elif 'crash' in self.name.lower():
            self.category = 'crash'
        elif 'agogo' in self.name.lower():
            self.category = 'agogo'
        elif 'clave' in self.name.lower():
            self.category = 'clave'
        elif 'clap' in self.name.lower():
            self.category = 'clap'
        elif 'rim' in self.name.lower():
            self.category = 'rim'
        elif 'stick' in self.name.lower():
            self.category = 'stick'
        elif 'tamb' in self.name.lower():
            self.category = 'tambourine'
        elif 'shak' in self.name.lower():
            self.category = 'shaker'
        else:
            self.category = 'unknown'
    # end assign_category
    
# end class AudioInfo
