from AudioDelayFixer import AudioDelayFixer
import timeit

start = timeit.default_timer()

adf = AudioDelayFixer('input/30sn360p.mp4')
adf.fixAudioDelay()

stop = timeit.default_timer()

print('Time: ', stop - start)  
