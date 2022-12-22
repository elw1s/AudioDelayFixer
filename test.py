import torch
from IPython.display import Audio
from pprint import pprint


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

import moviepy.editor as mp

video_path = "input/AudioDelay2.mp4"
audio_path = "sound_extracted.wav"

my_clip = mp.VideoFileClip(video_path)
my_clip.audio.write_audiofile(audio_path)

wav = read_audio(audio_path, sampling_rate=SAMPLING_RATE)

frames = len(wav) // 512
audio_length = wav.shape[0] / SAMPLING_RATE
dbi_audio = audio_length / (len(wav) // 512)

# get speech timestamps from full audio file
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE, return_seconds = True)
pprint(speech_timestamps)

speech = []

for i in range(0 , frames):
    time = dbi_audio * i
    detected = False
    for dic in speech_timestamps:
        if dic['start'] < time and time < dic['end']:
            speech.append(1)
            detected = True
            break
    print(time , ": " , detected)
    if detected == False:
        speech.append(0)
    

print(speech)
print(SAMPLING_RATE)

