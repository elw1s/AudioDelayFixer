import face_recognition
import cv2
import time
from mouth_open_algorithm import get_lip_height, get_mouth_height
from moviepy.editor import VideoFileClip
import numpy as np                                                                                                         
import scipy.io.wavfile as wf                                                                                              
import matplotlib.pyplot as plt 
from numba import jit

import torch
from IPython.display import Audio
from pprint import pprint

import timeit

class AudioVideoDetection():                                                                                             

    def __init__(self, wave_input_filename, video_filepath):
        self.clip = VideoFileClip(video_filepath)
        self.video_capture = cv2.VideoCapture(video_filepath)
        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print( "Length of video in frames = " , total_frames)
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        print("FPS = " , fps)
        self.video_length = total_frames / fps
        self.frame_rate = fps
        self._read_wav(wave_input_filename)._convert_to_mono()
        self.sample_window = 0.02 #ms
        self.sample_overlap = 0.01 #ms                                                                                                                                               
        self.speech_window = 0.5 #half a second                                       
        self.speech_energy_threshold = 0.15 #30% of energy in voice band                                                    
        self.speech_start_band = 300                                                                                      
        self.speech_end_band = 3000
        self.lip_movements = []                                                                               
        

    def is_mouth_open(self,face_landmarks):
        top_lip = face_landmarks['top_lip']
        bottom_lip = face_landmarks['bottom_lip']

        top_lip_height = get_lip_height(top_lip)
        bottom_lip_height = get_lip_height(bottom_lip)
        mouth_height = get_mouth_height(top_lip, bottom_lip)
        
        # if mouth is open more than lip height * ratio, return true.
        ratio = 0.5
        #print('top_lip_height: %.2f, bottom_lip_height: %.2f, mouth_height: %.2f, min*ratio: %.2f' 
           # % (top_lip_height,bottom_lip_height,mouth_height, min(top_lip_height, bottom_lip_height) * ratio))
            
        if mouth_height > min(top_lip_height, bottom_lip_height) * ratio:
            return True
        else:
            return False
    

    #Returns audio, video data as tuple

    def getAudioVideoData(self):
        start = timeit.default_timer()
        video = self.getLipMovements()
        end = timeit.default_timer()
        print("Time taken for getting lip movements from video is ", end - start)
        start = timeit.default_timer()
        audio = self.getSpeech()
        end = timeit.default_timer()
        print("Time taken for getting speech from audio is ", end - start)
        return audio , video

    def getAudioData(self):
        audio_length = self.data.shape[0] / self.rate
        rate = self.rate
        return audio_length, rate
    
    def getVideoData(self):
        video_length = self.video_length
        frame_rate = self.frame_rate
        return video_length, frame_rate

    def getSpeech(self):

        SAMPLING_RATE = 16000

        torch.set_num_threads(1)

        USE_ONNX = False # change this to True if you want to test onnx model

        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                    model='silero_vad',
                                    force_reload=True,
                                    onnx=USE_ONNX)

        (get_speech_timestamps,
        save_audio,
        read_audio,
        VADIterator,
        collect_chunks) = utils

        #Buradaki filename değiş
        wav = read_audio(self.filename, sampling_rate=SAMPLING_RATE)
        self.rate = SAMPLING_RATE
        self.data = wav
        frames = (len(wav) // 512) + 1
        audio_length = wav.shape[0] / SAMPLING_RATE
        dber_audio = audio_length / frames

        # get speech timestamps from full audio file
        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE, return_seconds = True)

        speech = []

        for i in range(0 , frames):
            time = dber_audio * i
            detected = False
            for dic in speech_timestamps:
                if dic['start'] < time and time < dic['end']:
                    speech.append(1)
                    detected = True
                    break
            if detected == False:
                speech.append(0)

        print("Lenght of video = " , self.data.shape[0] / self.rate)
        print("Rate of audio = ", self.rate)
        print("Length of data = " , len(self.data))                                                              


        return speech

    @jit
    def getLipMovements(self):

        prev = 0

        while True:
            time_elapsed = time.time() - prev
            ret, frame = self.video_capture.read()
            while ret:
                if time_elapsed > 1./self.frame_rate:
                    
                    prev = time.time()

                    face_locations = face_recognition.face_locations(frame)
                    face_landmarks_list = face_recognition.face_landmarks(frame, face_locations)

                    # Loop through each face in this frame of video
                    for face_landmarks in face_landmarks_list:

                        # Display text for mouth open / close
                        ret_mouth_open = self.is_mouth_open(face_landmarks)
                        if ret_mouth_open is True:
                            self.lip_movements.append(1)
                            text = 'Open'
                        else:
                            self.lip_movements.append(0)
                            text = 'Close'
                        cv2.putText(frame, text, (0,50) , cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1)
                        
                    if(len(face_locations) == 0):
                        self.lip_movements.append(-1)
                    # Display the resulting image
                    #cv2.imshow('Video', frame)

                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time_elapsed = time.time() - prev
                ret, frame = self.video_capture.read()
            
            break
        
        self.video_capture.release()
        cv2.destroyAllWindows()
        return self.lip_movements
                                                                                        
                                                                                                                           
    def _read_wav(self, wave_file):                                                                                        
        self.rate, self.data = wf.read(wave_file)                                                                          
        self.channels = len(self.data.shape)                                                                               
        self.filename = wave_file                                                                                          
        return self                                                                                                        
                                                                                                                           
    def _convert_to_mono(self):                                                                                            
        if self.channels == 2 :                                                                                            
            self.data = np.mean(self.data, axis=1, dtype=self.data.dtype)                                                  
            self.channels = 1                                                                                              
        return self                                                                                                        
                                                                                                                           
    def _calculate_frequencies(self, audio_data):                                                                          
        data_freq = np.fft.fftfreq(len(audio_data),1.0/self.rate)                                                          
        data_freq = data_freq[1:]                                                                                          
        return data_freq                                                                                                   
                                                                                                                           
    def _calculate_amplitude(self, audio_data):                                                                            
        data_ampl = np.abs(np.fft.fft(audio_data))                                                                         
        data_ampl = data_ampl[1:]                                                                                          
        return data_ampl                                                                                                   
                                                                                                                           
    def _calculate_energy(self, data):                                                                                     
        data_amplitude = self._calculate_amplitude(data)                                                                   
        data_energy = data_amplitude ** 2                                                                                  
        return data_energy                                                                                                 
                                                                                                                           
    def _znormalize_energy(self, data_energy):                                                                             
        energy_mean = np.mean(data_energy)                                                                                 
        energy_std = np.std(data_energy)                                                                                   
        energy_znorm = (data_energy - energy_mean) / energy_std                                                            
        return energy_znorm                                                                                                
                                                                                                                           
    def _connect_energy_with_frequencies(self, data_freq, data_energy):                                                    
        energy_freq = {}                                                                                                   
        for (i, freq) in enumerate(data_freq):                                                                             
                energy_freq[abs(freq)] = data_energy[i]                                                                    
        return energy_freq                                                                                                 
                                                                                                                           
    def _calculate_normalized_energy(self, data):                                                                          
        data_freq = self._calculate_frequencies(data)                                                                      
        data_energy = self._calculate_energy(data)                                                                         
        #data_energy = self._znormalize_energy(data_energy) #znorm brings worse results                                    
        energy_freq = self._connect_energy_with_frequencies(data_freq, data_energy)                                        
        return energy_freq                                                                                                 
                                                                                                                           
    def _sum_energy_in_band(self,energy_frequencies, start_band, end_band):                                                
        sum_energy = 0                                                                                                     
        for f in energy_frequencies.keys():                                                                                
            if start_band<f<end_band:                                                                                      
                sum_energy += energy_frequencies[f]
        return sum_energy                                                                                                  
                                                                                                                           
    def _median_filter (self, x, k):                                                                                       
        assert k % 2 == 1, "Median filter length must be odd."                                                             
        assert x.ndim == 1, "Input must be one-dimensional."                                                               
        k2 = (k - 1) // 2                                                                                                  
        y = np.zeros ((len (x), k), dtype=x.dtype)                                                                         
        y[:,k2] = x                                                                                                        
        for i in range (k2):                                                                                               
            j = k2 - i                                                                                                     
            y[j:,i] = x[:-j]                                                                                               
            y[:j,i] = x[0]                                                                                                 
            y[:-j,-(i+1)] = x[j:]                                                                                          
            y[-j:,-(i+1)] = x[-1]                                                                                          
        return np.median (y, axis=1)                                                                                       
                                                                                                                           
    def _smooth_speech_detection(self, detected_windows):                                                                  
        median_window=int(self.speech_window/self.sample_window)                                                           
        if median_window%2==0: median_window=median_window-1                                                               
        median_energy = self._median_filter(detected_windows[:,1], median_window)                                          
        return median_energy                                                                                               
                                                                                                                           
    def convert_windows_to_readible_labels(self, detected_windows):                                                        
        """ Takes as input array of window numbers and speech flags from speech                                            
        detection and convert speech flags to time intervals of speech.                                                    
        Output is array of dictionaries with speech intervals.                                                             
        """                                                                                                                
        speech_time = []                                                                                                   
        is_speech = 0                                                                                                      
        for window in detected_windows:                                                                                    
            if (window[1]==1.0 and is_speech==0):                                                                          
                is_speech = 1                                                                                              
                speech_label = {}                                                                                          
                speech_time_start = window[0] / self.rate                                                                  
                speech_label['speech_begin'] = speech_time_start                                                           
                print (window[0], speech_time_start)                                                                         
                #speech_time.append(speech_label)                                                                          
            if (window[1]==0.0 and is_speech==1):                                                                          
                is_speech = 0                                                                                              
                speech_time_end = window[0] / self.rate                                                                    
                speech_label['speech_end'] = speech_time_end                                                               
                speech_time.append(speech_label)                                                                           
                print (window[0], speech_time_end)                                                                           
        return speech_time                                                                                                 
                                                                                                                           
    def plot_detected_speech_regions(self):                                                                                
        """ Performs speech detection and plot original signal and speech regions.                                         
        """                                                                                                                
        data = self.data                                                                                                   
        detected_windows = self.detect_speech()                                                                            
        data_speech = np.zeros(len(data))                                                                                  
        it = np.nditer(detected_windows[:,0], flags=['f_index'])                                                           
        while not it.finished:                                                                                             
            data_speech[int(it[0])] = data[int(it[0])] * detected_windows[it.index,1]                                      
            it.iternext()                                                                                                  
        plt.figure()                                                                                                       
        plt.plot(data_speech)                                                                                              
        plt.plot(data)                                                                                                     
        plt.show()                                                                                                                                                                                                               
        print (data_speech)                                                                                                
        return self                                                                                                        
                                                                                                                           
    def detect_speech(self):                                                                                               
        """ Detects speech regions based on ratio between speech band energy                                               
        and total energy.                                                                                                  
        Output is array of window numbers and speech flags (1 - speech, 0 - nonspeech).                                    
        """                                                                                                                
        detected_windows = np.array([])
        audio_detection = []                                                                                    
        sample_window = int(self.rate * self.sample_window)
        sample_overlap = int(self.rate * self.sample_overlap)
        print("Lenght of video = " , self.data.shape[0] / self.rate)
        print("Rate of audio = ", self.rate)
        print("Length of data = " , len(self.data))                                                              
        data = self.data
        sample_start = 0                                                                                                   
        start_band = self.speech_start_band                                                                                
        end_band = self.speech_end_band
        speech_ratio_list = []
        i = 0
        while (sample_start < (len(data) - sample_window)):                                                                
            sample_end = sample_start + sample_window                                                                      
            if sample_end>=len(data): sample_end = len(data)-1
            #print(sample_start , " ~ " , sample_end , "   -  " , len(data))                                                             
            data_window = data[sample_start:sample_end]                                                                    
            energy_freq = self._calculate_normalized_energy(data_window)                                                   
            sum_voice_energy = self._sum_energy_in_band(energy_freq, start_band, end_band)                                 
            sum_full_energy = sum(energy_freq.values())                                                                    
            speech_ratio = sum_voice_energy/sum_full_energy
            speech_ratio_list.append(speech_ratio)                                                                
            # Hipothesis is that when there is a speech sequence we have ratio of energies more than Threshold
            #if(i < len(self.lip_movements)):
            #    print("Speech Ratio = " , speech_ratio ," - threshold = " , self.speech_energy_threshold , " - Detected = " , bool(speech_ratio>self.speech_energy_threshold) , " - Mouth = " , bool(self.lip_movements[i]))             
            i += 1
            speech_ratio = speech_ratio>self.speech_energy_threshold
            audio_detection.append(int(speech_ratio))                                                       
            #detected_windows = np.append(detected_windows, speech_ratio)    #Changed                                
            sample_start += sample_overlap                                                                                 
                                                
        return audio_detection

    def getMaximumScore(self , audio_window , video_window):
        GAP = -2 #Cost of Substition, Deletion and Insertion
        MATCH = 1
        MISS = -1

        n = len(audio_window) + 1
        m = len(video_window) + 1

        matrix = np.zeros((n,m))

        max_score = -1

        for i in range(1,n):
            for j in range(1,m):
                
                if audio_window[i - 1] == video_window[j - 1]:
                    match_value = MATCH
                else:
                    match_value = MISS

                matrix[i,j] = max(0 , matrix[i - 1, j - 1] + match_value , matrix[i, j-1] + GAP, matrix[i-1, j] + GAP)

                if matrix[i, j] >= max_score:
                        max_score = matrix[i, j]
                        
        return max_score

    def preprocessVideo(self , video_data):

        processed_video = []
        temp = []
        for i in range(0 , len(video_data)):
            if video_data[i] == -1:

                if len(temp) != 0:
                    processed_video.append((temp.copy(), i))
                temp.clear()
                
            else:
                temp.append(video_data[i])

        return processed_video
