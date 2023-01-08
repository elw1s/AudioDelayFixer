import face_recognition
import cv2
from mouth_open_algorithm import get_lip_height, get_mouth_height
from moviepy.editor import VideoFileClip
import numpy as np                                                                                                         
import scipy.io.wavfile as wf                                                                                              
import matplotlib.pyplot as plt 
import torch
import timeit

class AudioVideoDetection():                                                                                             

    def __init__(self, wave_input_filename, video_filepath):
        self.clip = VideoFileClip(video_filepath)
        self.video_capture = cv2.VideoCapture(video_filepath)
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print( "Length of video in frames = " , self.total_frames)
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        print("FPS = " , fps)
        self.video_length = self.total_frames / fps #
        print("Duration of video = " , self.video_length) #
        self.frame_rate = fps
        self._read_wav(wave_input_filename)._convert_to_mono()
        self.sample_window = 0.02 #ms
        self.sample_overlap = 0.01 #ms                                                                                                                                               
        self.speech_window = 0.5 #half a second                                       
        self.speech_energy_threshold = 0.15 #30% of energy in voice band                                                    
        self.speech_start_band = 300                                                                                      
        self.speech_end_band = 3000
        self.lip_movements = []                                                                               
        

    def calculate_mouth_distance(self,face_landmarks):
        top_lip = face_landmarks['top_lip']
        bottom_lip = face_landmarks['bottom_lip']

        top_lip_height = get_lip_height(top_lip)
        bottom_lip_height = get_lip_height(bottom_lip)
        mouth_height = get_mouth_height(top_lip, bottom_lip)
            
        if mouth_height > min(top_lip_height, bottom_lip_height) * 1:
            return 4
        elif mouth_height > min(top_lip_height, bottom_lip_height) * 0.75:
            return 3
        elif mouth_height > min(top_lip_height, bottom_lip_height) * 0.5:
            return 2
        elif mouth_height > min(top_lip_height, bottom_lip_height) * 0.25:
            return 1
        else: 
            return 0
    

    def getAudioVideoData(self):
        start = timeit.default_timer()
        video = self.getLipMovements()
        end = timeit.default_timer()
        print("Time taken for getting lip movements from video is ", end - start)
        print("Video arrayinin uzunlugu = ", int(self.frame_rate * self.video_length) , " - " , len(video))
        print("Yeni hesaplanan video_length = " , len(video) / self.frame_rate)
        print("Video Chunks (ms) = ", 1000 * (self.video_length / len(video)))
        start = timeit.default_timer()
        audio = self.detect_speech()
        end = timeit.default_timer()
        print("Time taken for getting speech from audio is ", end - start)
        return audio , video

    def getAudioData(self):
        audio_length = self.video_length 
        rate = self.rate
        return audio_length, rate
    
    def getVideoData(self):
        video_length = self.video_length
        frame_rate = self.frame_rate
        return video_length, frame_rate


    def getLipMovements(self):

        delay = 2
        count = 0
        frames = []
        start = timeit.default_timer()
        while True:
            success, frame = self.video_capture.read()
            if not success:
                break
            elif success and count % delay == 0:
                frames.append(frame)
            count += 1
        end = timeit.default_timer()
        print("Time taken for reading video frame by frame = " , end - start)
        startProcess = timeit.default_timer()
        for frame in frames:
            face_locations = face_recognition.face_locations(frame , number_of_times_to_upsample = 0 , model= "hog")
            if(len(face_locations) == 0):
                    self.lip_movements.append(-1)
            else:
                    face_landmarks_list = face_recognition.face_landmarks(frame, face_locations)
                    for face_landmarks in face_landmarks_list:
                        mouth_distance = self.calculate_mouth_distance(face_landmarks)
                        self.lip_movements.append(mouth_distance)

        endProcess = timeit.default_timer()
        print("Time taken for processing frame = " , endProcess - startProcess)
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
                                                                                                                                                                                                                                                      
    def _connect_energy_with_frequencies(self, data_freq, data_energy):                                                    
        energy_freq = {}                                                                                                   
        for (i, freq) in enumerate(data_freq):                                                                             
                energy_freq[abs(freq)] = data_energy[i]                                                                    
        return energy_freq                                                                                                 
                                                                                                                           
    def _calculate_normalized_energy(self, data):                                                                          
        data_freq = self._calculate_frequencies(data)                                                                      
        data_energy = self._calculate_energy(data)                                                                         
        energy_freq = self._connect_energy_with_frequencies(data_freq, data_energy)                                        
        return energy_freq                                                                                                 
                                                                                                                           
    def _sum_energy_in_band(self,energy_frequencies, start_band, end_band):                                                
        sum_energy = 0                                                                                                     
        for f in energy_frequencies.keys():                                                                                
            if start_band<f<end_band:                                                                                      
                sum_energy += energy_frequencies[f]
        return sum_energy                                                                                                  

                                                                                                                           
    def detect_speech(self):                                                                                               
        """ Detects speech regions based on ratio between speech band energy                                               
        and total energy.                                                                                                  
        Output is array of window numbers and speech flags (1 - speech, 0 - nonspeech).                                    
        """                                                                                                                
        detected_windows = np.array([])
        audio_detection = []                                                                                    
        #sample_window = int(self.rate * self.sample_window)
        sample_window = int(len(self.data) / len(self.lip_movements))
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

            
            if speech_ratio>0.6:
                speech_ratio = 4
            elif speech_ratio>0.45:
                speech_ratio = 3
            elif speech_ratio>0.3:
                speech_ratio = 2
            elif speech_ratio>0.15:
                speech_ratio = 1
            else:
                speech_ratio = 0
                
            audio_detection.append(int(speech_ratio))

            #speech_ratio = speech_ratio>self.speech_energy_threshold #Yorumu kaldır                                                       
            #sample_start += sample_overlap                                                                                 
            sample_start += sample_window

        return audio_detection

    def preprocessVideo(self , video_data):

        processed_video = []
        temp = []
        inRange = True
        start = 0
        for i in range(0 , len(video_data)):
            if video_data[i] == -1:
                if len(temp) > 5:
                    end = i-1
                    processed_video.append((temp.copy(), start , end))
                temp.clear()
                inRange = True
            else:
                if inRange:
                    start = i
                temp.append(video_data[i])
                inRange = False
        return processed_video

    def getMaximumScore(self , audio_window , video_window):
        GAP = -2
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

    """def getMaximumScore(self, audio_window, video_window):
        score = 0
        n = len(audio_window)
        m = len(video_window)
        
        #print("Audio = " , audio_window)
        #print("Video = " , video_window)

        for i in range(n):
            if (audio_window[i] == 0 and video_window[i] != 0):
                score -= 2
            elif(audio_window[i] != 0 and video_window[i] == 0):
                score -= 2
            elif(audio_window[i] == video_window[i]):
                score += 5
            elif(abs(audio_window[i] - video_window[i]) == 1):
                score += 3
            elif(abs(audio_window[i] - video_window[i]) == 2):
                score += 2
            elif(abs(audio_window[i] - video_window[i]) == 3):
                score += 1
        #print("Score = " , score)
        return score """


