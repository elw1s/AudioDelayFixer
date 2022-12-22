"""
Detect a face in webcam video and check if mouth is open.
"""
import face_recognition
import cv2
import time
from mouth_open_algorithm import get_lip_height, get_mouth_height
from moviepy.editor import VideoFileClip

class MouthDetection():
    
    def __init__(self, video_filepath):
        self.clip = VideoFileClip(video_filepath)
        self.video_capture = cv2.VideoCapture(video_filepath)
        length = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print( "Length of video in frames = " , length )
        self.frame_rate = 25

    def is_mouth_open(self,face_landmarks):
        top_lip = face_landmarks['top_lip']
        bottom_lip = face_landmarks['bottom_lip']

        top_lip_height = get_lip_height(top_lip)
        bottom_lip_height = get_lip_height(bottom_lip)
        mouth_height = get_mouth_height(top_lip, bottom_lip)
        
        # if mouth is open more than lip height * ratio, return true.
        ratio = 0.75
        #print('top_lip_height: %.2f, bottom_lip_height: %.2f, mouth_height: %.2f, min*ratio: %.2f' 
           # % (top_lip_height,bottom_lip_height,mouth_height, min(top_lip_height, bottom_lip_height) * ratio))
            
        if mouth_height > min(top_lip_height, bottom_lip_height) * ratio:
            return True
        else:
            return False


    def getLipMovements(self):

        prev = 0
        lip_movements = []

        start_time = time.time()
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
                            lip_movements.append(1)
                            text = 'Open'
                        else:
                            lip_movements.append(0)
                            text = 'Close'
                        cv2.putText(frame, text, (0,50) , cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1)
                        
                    if(len(face_locations) == 0):
                        lip_movements.append(0)
                    # Display the resulting image
                    #cv2.imshow('Video', frame)

                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time_elapsed = time.time() - prev
                ret, frame = self.video_capture.read()
            total_time = (time.time() - start_time)
            print("Total time for program = ", total_time)
            print("Duration of video = " , self.clip.duration)
            print("Processed Frames = " , len(lip_movements))
            #print("Duration of the Video / Processed Frames = " , self.clip.duration / len(lip_movements))
            break
        
        self.video_capture.release()
        cv2.destroyAllWindows()
        return lip_movements