import json
import pyaudio
import datetime
import keyboard
import threading
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from scipy.ndimage import find_objects
import azure.cognitiveservices.speech as speechsdk

import common as cm

def voice_activity_detection(data, fs, show_img=False):
    '''
    Unit 9
    '''

    # Parameters
    freq_min = 300
    freq_max = 4000

    TH_is_active = 100
    TH_connect_max = 15
    TH_keep_min = 6
    buffer_num = 5

    # VAD Step 2 ~ 3
    SE, frame_time = cm.spectral_energy(data, fs, freq_min, freq_max)

    # VAD Step 4
    VAD = np.zeros(len(SE))
    for frame_idx in range(len(SE)):
        if SE[frame_idx] >= TH_is_active:
            VAD[frame_idx] = 1
        else:
            VAD[frame_idx] = 0

    # VAD Step 5
    labeled, ncomponents = label(VAD, np.ones((3), dtype=np.int))
    segments = find_objects(labeled)
    
    for i in range(len(segments)-1):
        idx_from = segments[i][0].stop
        idx_to = segments[i+1][0].start
        if idx_to-idx_from <= TH_connect_max:
            VAD[idx_from:idx_to] = 1
            
    # VAD Step 6
    labeled, ncomponents = label(VAD, np.ones((3), dtype=np.int))
    segments = find_objects(labeled)
    
    VAD_points = []
    for i in range(len(segments)):
        idx_from = segments[i][0].start
        idx_to = segments[i][0].stop
        if idx_to-idx_from+1 >= TH_keep_min:
            VAD_points.append([idx_from, idx_to])
        else:
            VAD[idx_from:idx_to] = 0
    
    # VAD Step 7
    for i in VAD_points:
        i[0] = max(0, i[0]-buffer_num)
        i[1] = min(len(VAD)-1, i[1]+buffer_num)
        VAD[i[0]:i[1]] = 1
    
    segments = []
    
    for i in VAD_points:
            
        segment = {}
        segment['frame_from'] = i[0]
        segment['frame_to'] = i[1]
        segment['sec_from'] = frame_time[i[0]]
        segment['sec_to'] = frame_time[i[1]]
        segment['samples'] = data[int(segment['sec_from']*fs):int(segment['sec_to']*fs)]
        
        segments.append(segment)
        
    if show_img:
        
        plt.figure(figsize=(10, 4))
        plt.plot(frame_time, SE)
        plt.title('Spectral Energy')
        plt.xlabel('Time (sec)')
        plt.ylabel('Energy')
        plt.show()
        
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(data))/fs, data)
        plt.plot(frame_time, VAD*max(data), linewidth=3)
        plt.title('VAD')
        plt.xlabel('Time (sec)')
        plt.ylabel('Amplitude')
        plt.show()
    
    return segments

def transcribe_azure(audio_path, azure_key, region, lang):
    '''
    Use Azure Speech-To-Text service. You can access the key from the following website.
    https://docs.microsoft.com/zh-tw/azure/cognitive-services/speech-service/get-started
    '''
    
    speech_config = speechsdk.SpeechConfig(subscription=azure_key, region=region, speech_recognition_language=lang)
    speech_config.request_word_level_timestamps()
    
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    result = speech_recognizer.recognize_once()
    
    full_str = ''
    word_dict = []
    
    if result.text != '':
        stt = json.loads(result.json)
        confidences_in_nbest = [item['Confidence'] for item in stt['NBest']]
        best_index = confidences_in_nbest.index(max(confidences_in_nbest))
        words = stt['NBest'][best_index]['Words']
        for word in words:
            full_str = full_str + word['Word']
            word_dict.append({'word': word['Word'], 'from': word['Offset']*1e-7, 'to': (word['Offset']+word['Duration'])*1e-7})
    
    return full_str, word_dict


'''
Realtime recording

遊戲頁面
http://www.i-gamer.net/play/15683.html

'''


FS = 16000
DETECTION_SEC = 50

chunk = 512
p = pyaudio.PyAudio()
stream = p.open( format=pyaudio.paFloat32, channels=1, rate=FS, input=True, frames_per_buffer=chunk)
han_win = np.hanning(chunk+2)[1:-1]

def recognize():
    
    # Write your code here
    # 
    azure_key = '5aa25ccfd41c42e9a4c590ce46a1d9fa'
    region = 'westus'
    # 對 tmp.wav 執行 transcribe_azure()
    full_str, words = transcribe_azure('tmp.wav', azure_key, region, 'zh-TW')
    # print 出 full_str
    #print(full_str)
    if full_str == '陳信宏':
        print('陳信宏 Someone call you at')
        now = datetime.datetime.now()
        print(now)
    elif full_str == '信宏':
        print('信宏 Someone call you at')
        now = datetime.datetime.now()
        print(now)
    elif full_str == '信宏哥':
        print('信宏哥 Someone call you at')
        now = datetime.datetime.now()
        print(now)
    elif full_str == '阿信':
        print('阿信 Someone call you at')
        now = datetime.datetime.now()
        print(now)
    elif full_str == '學友哥':
        print('學友哥 Someone call you at')
        now = datetime.datetime.now()
        print(now)
    
print('Start')

samples = []
now = datetime.datetime.now()
for i in range(int(DETECTION_SEC*FS/chunk)):
    reading_samples = np.frombuffer(stream.read(chunk, exception_on_overflow=False), dtype = np.float32).tolist()
    samples.extend(reading_samples)
    
    ''' Process Voice Avtivity Detection'''
    display_sec = 2
    
    # Write your code here
    #
    # [現在時間] 和 now 差距幾秒 ( 使用 .total_seconds() )
    diff_sec = (datetime.datetime.now() - now).total_seconds()
    # 若差距大於 display_sec 秒，則做下面事情
    if diff_sec > display_sec:
        # 將 samples 轉成 numpy 矩陣
        s_samples = np.asarray(samples)
        # 取出最新的 2 秒鐘 (2*16000個取樣點)
        s_samples = s_samples[-FS*display_sec:]
        # 將取出的訊號，執行 voice_activity_detection()，可得到 segments list
        segments = voice_activity_detection(s_samples, FS, show_img=False)
        # 若 len(segments) > 0 以及 segments[0]起點大於0.3秒、終點小於 1.7秒，則做下面事情
        if len(segments) > 0 and segments[0]['sec_from'] > 0.3 and segments[0]['sec_to'] < 1.7:
            # 因確認語音有活動且有效，可更新變數 now 至 [現在時間]
            now = datetime.datetime.now()
            #print('Recognizing...')
            # 儲存音檔，固定檔名為 tmp.wav
            cm.write_wav(f"tmp.wav", segments[0]['samples'], FS)
            t = threading.Thread(target=recognize) # 到這邊後，換寫 recognize()
            t.start()


stream.stop_stream()
stream.close()
p.terminate()
print('End')
