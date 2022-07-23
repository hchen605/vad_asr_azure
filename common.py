# -*- coding: utf-8 -*-

import pyaudio
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def read_wav(audio_path):
    fs, data = wavfile.read(audio_path)
    data = data/32768
    return fs, data

def write_wav(file_name, data, fs):
    '''
    Input float data, write int16 wav
    '''
    data = data*32768
    data = data.astype(np.int16)
    wavfile.write(file_name, fs, data)

def show_spectrogram(data, fs, save_name=None):
    '''
    Unit 5
    Only accept mono channel
    '''
    
    plt.figure(figsize=(10, 6))
    plt.specgram(data, Fs=fs, cmap='jet', 
                 NFFT=int(fs*0.064),
                 noverlap=int(fs*0.032),
                 mode='magnitude')
    plt.xlabel('Time (sec)')
    plt.ylabel('Freq (Hz)')
    plt.title('Spectrogram')
    plt.colorbar()
    if save_name is not None:
        plt.savefig(f'spectrogram_{save_name}.jpg', dpi=150)
    plt.show()

def show_signal(data, fs, save_name=None):
    '''
    Unit 5
    Only accept mono channel
    '''
    
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(data))/fs, data)
    plt.xlabel('Time (sec)')
    plt.ylabel('Amplitude')
    plt.title('Time domain')
    plt.ylim([-1, 1])
    if save_name is not None:
        plt.savefig(f'time_{save_name}.jpg', dpi=150)
    plt.show()

def short_time_energy(ns, fs):
    ''' Unit 7 '''
    
    def round_p(x):
        return int(x+0.5)
    
    # pre
    ns_length = ns.shape[0]
    frame_size = round_p(0.032*fs)
    NFFT = 2*frame_size
    han_win = np.hanning(frame_size+2)[1:-1]
    
    # main
    shift_pct = 0.5
    overlap = round_p((1-shift_pct)*frame_size)
    offset = frame_size - overlap
    max_m = int(np.floor((ns_length - NFFT)/offset)) + 1
    
    frame_time = np.zeros(max_m)
    output = np.zeros(max_m)
    
    for m in range(max_m):
        
        begin = m*offset
        finish = m*offset + frame_size  
        s_frame = ns[begin:finish]

        s_frame_win = s_frame*han_win
        ste = np.sum(s_frame_win*s_frame_win)
        
        output[m] = ste
        frame_time[m] = ((begin+finish)/2)/fs

    return output, frame_time

def zero_cross_rate(ns, fs):
    ''' Unit 7 '''
    
    def round_p(x):
        return int(x+0.5)
    
    def sign(x):
        if x < 0:
            return -1
        elif x == 0:
            return 0
        elif x > 0:
            return 1
    
    # pre
    ns_length = ns.shape[0]
    frame_size = round_p(0.032*fs)
    NFFT = 2*frame_size
    han_win = np.hanning(frame_size+2)[1:-1]
    
    # main
    shift_pct = 0.5
    overlap = round_p((1-shift_pct)*frame_size)
    offset = frame_size - overlap
    max_m = int(np.floor((ns_length - NFFT)/offset)) + 1
    
    frame_time = np.zeros(max_m)
    output = np.zeros(max_m)
    
    for m in range(max_m):
        
        begin = m*offset
        finish = m*offset + frame_size  
        s_frame = ns[begin:finish]

        s_frame_win = s_frame*han_win
        
        zcr = 0
        for i in range(2, frame_size):
            zcr = zcr + 0.5*abs(sign(s_frame_win[i])-sign(s_frame_win[i-1]))
        
        output[m] = zcr
        frame_time[m] = ((begin+finish)/2)/fs

    return output, frame_time

def spectral_energy(ns, fs, freq_min, freq_max):
    ''' Unit 7 '''
    
    def round_p(x):
        return int(x+0.5)
    
    # pre
    ns_length = ns.shape[0]
    frame_size = round_p(0.032*fs)
    NFFT = 2*frame_size
    han_win = np.hanning(frame_size+2)[1:-1]
    
    # filter band
    freq_per = (fs/2)/(NFFT/2)
#    freq_min = 700
#    freq_max = 1200
    freq_min_idx = int(freq_min/freq_per)
    freq_max_idx = int(freq_max/freq_per)
    
    # main
    shift_pct = 0.5
    overlap = round_p((1-shift_pct)*frame_size)
    offset = frame_size - overlap
    max_m = int(np.floor((ns_length - NFFT)/offset)) + 1
    
    frame_time = np.zeros(max_m)
    SE = np.zeros(max_m)
    
    for m in range(max_m):
        
        begin = m*offset
        finish = m*offset + frame_size  
        s_frame = ns[begin:finish]

        s_frame_win = s_frame*han_win
        
        # FFT
        s_fft = np.fft.fft(s_frame_win, NFFT)
        s_mag = abs(s_fft)
        
        # select band
        select = s_mag[freq_min_idx:freq_max_idx]
        energy = 20*np.log10(select)
        energy = np.sum(energy[energy > 0])

        SE[m] = energy
        frame_time[m] = ((begin+finish)/2)/fs
        
    return SE, frame_time

def play_raw(speech, fs):
    '''
    Unit 8
    Play audio raw data
    '''

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=fs, output=True)
    stream.write(np.asarray(speech).astype(np.float32).tostring())
    stream.stop_stream()
    stream.close()
    p.terminate()