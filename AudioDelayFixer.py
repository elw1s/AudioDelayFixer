from audio_video_detection import AudioVideoDetection
import matplotlib.pyplot as plt
import numpy as np
import moviepy.editor as mp
import timeit

AUDIO_PATH = "sound/sound_extracted.wav"


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

        print(len(self.audio_data))
        print(self.audio_data)
        print(" -------------------------------- ")
        print(len(self.video_data))
        print(self.video_data)
        print("Time taken for getting audio and video data is " , end - start)

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

        self.createClipForEachSpeech(self.compareAudioVideo())

        end = timeit.default_timer()
        print("Time taken for compareAudioVideo is " , end - start)


    def align_video(self, start, to , video, length_audio_window):
        overlap = 1
        max = 0
        while start < to:

                end = start + length_audio_window

                if(end >= to):
                    break
                    
                temp = self.audio_video_detection.getMaximumScore(self.audio_data[start:end] , video)
                if(temp > max):
                    max = temp
                    min_index = start
                    max_index = end

                start += overlap

        return (min_index, max_index)

    def compareAudioVideo(self):
        self.processed = self.audio_video_detection.preprocessVideo(self.video_data)
        length_of_processed = len(self.processed)
        print("Length of processed = " , length_of_processed)
        values = []
        max_index = 0

        for k in range(0 , length_of_processed):
            video , start_index , end_index = self.processed[k]

            #print("\n\n",k+1 , ". video = ", video)
            video_time = int(end_index * self.dber_video - start_index * self.dber_video)
            oran = self.dber_video / self.dber_audio
            #length_audio_window = int(len(video) * oran)
            length_audio_window = int(len(video))
            #Find local alignment
            max = 0
            last_stop = max_index

            ## REMAINING LENGTH
            remaining_video_length = 0
            for i in range(length_of_processed - 1 , k , -1):
                remaining_video_length += len(self.processed[i][0])

            
            print("while ", max_index , " < ", len(self.audio_data) - remaining_video_length , " + " , length_audio_window)
            print("Remaining video length = " , remaining_video_length)
            print("\n\n\n\n",k+1 , ". Video işleniyor")
            print("İncelenen audio= " , self.audio_data[max_index : len(self.audio_data) - remaining_video_length])
            print("\nAranan video= " , video)
            result = self.align_video(max_index, len(self.audio_data) - remaining_video_length, video , length_audio_window)
            min , max = result
            max_index = max
            values.append(result)

        return values
    
    def calculateDurationBetweenEachRate(self, length , data):
        return length / len(data)

    def createAudioClip(self):
        clip = mp.AudioFileClip(AUDIO_PATH)
        self.new_audioclip = mp.CompositeAudioClip([clip])
    

    def createClipForEachSpeech(self, values):
        kayma = 0
        for i in range(len(values)):
            (min_index , max_index) = values[i]
            video , start , end = self.processed[i]
            kayma += min_index - start
        
        kayma = int(kayma / len(values))

        audio_clip = self.new_audioclip.subclip((kayma * self.dber_audio) , (len(self.audio_data) - 1) * self.dber_audio)
        self.my_clip.audio = audio_clip
        self.my_clip.write_videofile("output/OUTPUT.mp4")
