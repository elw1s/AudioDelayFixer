from audio_video_detection import AudioVideoDetection
import matplotlib.pyplot as plt
import numpy as np
import moviepy.editor as mp
import timeit

AUDIO_PATH = "sound_extracted.wav"


class AudioDelayFixer():

    def __init__(self, video_filepath):

        self.extractAudioFromVideo(video_filepath)
        self.video_filepath = video_filepath

    def extractAudioFromVideo(self, video_filepath):
        self.my_clip = mp.VideoFileClip(video_filepath)
        self.my_clip.audio.write_audiofile(AUDIO_PATH)

    def getAudioVideoDetection(self,video_filepath, audio_path = AUDIO_PATH):
        return AudioVideoDetection(AUDIO_PATH, video_filepath)


    def fixAudioDelay(self):
        
        start = timeit.default_timer()
        self.audio_video_detection = self.getAudioVideoDetection(self.video_filepath)
        end = timeit.default_timer()

        print("Time taken for creating audiovideoDetection object is " , end - start)

        start = timeit.default_timer()
        self.audio_data , self.video_data = self.audio_video_detection.getAudioVideoData()
        end = timeit.default_timer()

        print("Time taken for getting audio and video data is " , end - start)

        """ fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].step(self.video_data , 'r')
        ax[1].step(self.audio_data , 'g')
        plt.show() """
        

        start = timeit.default_timer()
        audio_length , audio_rate = self.audio_video_detection.getAudioData()
        end = timeit.default_timer()
        print("Time taken for getting audio data details is " , end - start)
        start = timeit.default_timer()
        video_length , video_frame_rate = self.audio_video_detection.getVideoData()
        end = timeit.default_timer()
        print("Time taken for video data details is " , end - start)


        self.dber_audio = self.calculateDurationBetweenEachRate(audio_length , self.audio_data)
        self.dber_video = self.calculateDurationBetweenEachRate(video_length , self.video_data)
        self.createAudioClip()
        start = timeit.default_timer()

        #Bu kısımda process edilen videonun süre aralığını ve başlangıç saniyesini belirle. Ardından audio içinde bu aralığı
        #bul ve birleştir.

        self.createClipForEachSpeech(self.test())


        #self.createClipForEachSpeech(self.compareAudioVideo())
        end = timeit.default_timer()
        #print("Time taken for creating clip for each speech is " , end - start)
        print("Time taken for testing is " , end - start)


    
    def test(self):
        self.processed = self.audio_video_detection.preprocessVideo(self.video_data)
        length_of_processed = len(self.processed)
        values = []
        max_index = 0
        for k in range(0 , length_of_processed):
            video , start_index , end_index = self.processed[k]

            print(k+1 , ". video = ", video)
            print(k+1, ". Video araligi = " , start_index * self.dber_video , " - " , end_index * self.dber_video)

            video_time = int(end_index * self.dber_video - start_index * self.dber_video)
            print(k+1, ". Video süresi = " , video_time)
            oran = self.dber_video / self.dber_audio
            length_audio_window = int(len(video) * oran)
            #Find local alignment
            max = 0
            last_stop = max_index
            print('Length of video window = ' , len(video))
            print('Length of audio window = ' , length_audio_window)
            print('Oran = ', oran)
            print('Last stop = ', last_stop)
            print('from ', len(self.audio_data) - 1 , ' to ' , length_audio_window - 1 , ' by ' , -1 )
            """ for i in range(len(self.audio_data) - 1, last_stop + length_audio_window - 1 , -1 * length_audio_window):
                temp = self.audio_video_detection.getMaximumScore(self.audio_data[i - length_audio_window:i] , video)
                if(temp > max):
                    max = temp
                    max_index = i
            values.append((max , max_index)) """
            for i in range(len(self.audio_data) - 1, length_audio_window - 1, -1):
                temp = self.audio_video_detection.getMaximumScore(self.audio_data[i - length_audio_window:i] , video)
                if(temp > max):
                    max = temp
                    max_index = i
            values.append((max, max_index))

        return values

    
    def calculateDurationBetweenEachRate(self, length , data):
        return length / len(data)

    def createAudioClip(self):
        clip = mp.AudioFileClip(AUDIO_PATH)
        self.new_audioclip = mp.CompositeAudioClip([clip])
    

    def compareAudioVideo(self):

        self.processed = self.audio_video_detection.preprocessVideo(self.video_data) 
        values = []
        self.t = []
        print("AUDIO DATA ====== " , self.audio_data)
        print("Number of Processed videos = " , len(self.processed))
        for k in range(0 , len(self.processed)):
            max = 0
            max_index = -1
            processed_video , first_index, last_index = self.processed[k] #Processed değişti burayı düzelt error alırsın
            print("Processed video = " , processed_video)
            print("t = " , int(len(processed_video) * self.dber_video / self.dber_audio))
            self.t.append(int((len(processed_video) * self.dber_video) / self.dber_audio))
            start = timeit.default_timer()
            print((len(self.audio_data) - 1 - len(processed_video) - 1), " iterations for FOR LOOP")
            for i in range(len(self.audio_data) - 1, len(processed_video) - 1 , -1):
                temp = self.audio_video_detection.getMaximumScore(self.audio_data[i - self.t[k]:i] , processed_video)
                if(temp > max):
                    max = temp
                    max_index = i
            end = timeit.default_timer()
            print("Time taken for getting max index is " , end - start)
            print("max_index = ", max_index)
            values.append((max , max_index))
        
        return values

    def createClipForEachSpeech(self, values):
        for i in range(0, len(values)):

            (max , max_index) = values[i]
            video , start , end = self.processed[i]
            oran = self.dber_video / self.dber_audio
            length_audio_window = int(len(video) * oran)

            print(i+1, 'th Max score = ' , max)
            print(i+1, 'th Max index = ', max_index)
            #print(i+1, 'th Audio data = ' , self.audio_data[max_index - length_audio_window: max_index])

            newVideoClip = self.my_clip.subclip((start *   self.dber_video) , (end *   self.dber_video))
            newAudioClip = self.new_audioclip.subclip(((max_index - length_audio_window) * self.dber_audio) , max_index * self.dber_audio)
            newVideoClip.audio = newAudioClip
            newVideoClip.write_videofile("output/fixed{0}.mp4".format(i))

            """ processed_video , first_index, last_index = self.processed[i]
            (max , max_index) = values[i]

            video_time = int(last_index * self.dber_video - first_index * self.dber_video)
            oran = self.dber_video / self.dber_audio
            length_audio_window = int(len(processed_video) * oran)

            print(len(self.audio_data) , " - " , len(processed_video) , " - " , len(self.audio_data[max_index - length_audio_window : max_index]))
            print("MAX SCORE = " , max)
            print("VIDEO DATA = " , processed_video)
            print("------------------------------------------------------")
            print("AUDIO DATA = " , self.audio_data[max_index - length_audio_window : max_index])

            print(self.dber_audio)
            print(self.dber_video)
            print(((max_index - length_audio_window) * self.dber_audio), " is the starting point of audio window")
            print(max_index * self.dber_audio , " is the ending point of audio window")
            print("Audio window time = " , (max_index * self.dber_audio - ((max_index - length_audio_window) * self.dber_audio)))
            print("Video window time = " , (((last_index) *   self.dber_video) - ((last_index - len(processed_video)) *   self.dber_video)))
            print(((last_index - len(processed_video)) *   self.dber_video), " is the starting point of video window")
            print(((last_index) *   self.dber_video), " is the ending point of video window")

            newVideoClip = self.my_clip.subclip(((last_index - len(processed_video)) *   self.dber_video) , ((last_index) *   self.dber_video))
            newAudioClip = self.new_audioclip.subclip(((max_index - length_audio_window) * self.dber_audio) , max_index * self.dber_audio)
            newVideoClip.audio = newAudioClip
            newVideoClip.write_videofile("output/fixed{0}.mp4".format(i)) """