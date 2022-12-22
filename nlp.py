import numpy as np
from audio_video_detection import AudioVideoDetection
import moviepy.editor as mp


video_path = "AudioDelay1.mp4"
audio_path = "sound_extracted.wav"

my_clip = mp.VideoFileClip(video_path)
my_clip.audio.write_audiofile(audio_path)

audio_video_detection = AudioVideoDetection(audio_path, video_path)
audio_data , video_data = audio_video_detection.getAudioVideoData()


GAP = -2 #Cost of Substition, Deletion and Insertion
MATCH = 1
MISS = -1

def backtracking(str1, str2,matrix, i, j, curr_sequence1, curr_sequence2):
        
    if curr_sequence1 == None:
        curr_sequence1 = []
    if curr_sequence2 == None:
        curr_sequence2 = []
    
    if matrix[i][j] == 0 or i == 0 or j == 0:
        return curr_sequence1,curr_sequence2, (i,j)
    
    max_score = max(0, matrix[i-1][j-1], matrix[i][j-1], matrix[i-1][j])
    if max_score == matrix[i-1][j-1]:
        curr_sequence1.append(str1[i-1])
        curr_sequence2.append(str2[j-1])
        return backtracking(str1, str2,matrix, i-1, j-1, curr_sequence1, curr_sequence2)
    if max_score == matrix[i][j-1]:
        curr_sequence1.append('-')
        curr_sequence2.append(str2[j-1])
        return backtracking(str1, str2,matrix, i, j-1, curr_sequence1, curr_sequence2)
    if max_score == matrix[i-1][j]:
        curr_sequence1.append(str1[i-1])
        curr_sequence2.append('-')
        return backtracking(str1, str2,matrix, i-1, j, curr_sequence1, curr_sequence2)

n = len(audio_data) + 1
m = len(video_data) + 1

matrix = np.zeros((n,m))

max_score = -1
max_index = (-1, -1) 

for i in range(1,n):
    for j in range(1,m):
        
        if audio_data[i - 1] == video_data[j - 1]:
            match_value = MATCH
        else:
            match_value = MISS

        matrix[i,j] = max(0 , matrix[i - 1, j - 1] + match_value , matrix[i, j-1] + GAP, matrix[i-1, j] + GAP)

        if matrix[i, j] >= max_score:
                max_index = (i,j)
                max_score = matrix[i, j]
                
(max_i, max_j) = max_index


aligned_str1 , aligned_str2, min_index = backtracking(audio_data,video_data,matrix,max_i,max_j,None,None)
print("MIN - MAX INDEX= " , min_index , " - " , max_index)
aligned_str1.reverse()
aligned_str2.reverse()
print(aligned_str1)
print(aligned_str2)
print(len(aligned_str1))
print(len(aligned_str2))
print('MATRIX')
print(matrix)
print('Matrix size: ', n , 'x' , m)
print('Max Score: ' ,max_score)
ratio = (max_score) / (max(len(audio_data), len(video_data)))
print('Similarity: %' , ratio * 100)

(min_i , min_j) = min_index



audio_length , audio_rate = audio_video_detection.getAudioData()
video_length , video_frame_rate = audio_video_detection.getVideoData()

dbi_audio = audio_length / len(audio_data)
dbi_video = video_length / len(video_data)

print(dbi_audio)
print(dbi_video)
print((min_i * dbi_audio), " is the starting point of audio")
print((min_j * dbi_video), " is the starting point of video")
