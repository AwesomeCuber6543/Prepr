import numpy as np
import mediapipe as mp
import math
import threading
from flask import Flask, jsonify, request
import os, sys
import openai
import cv2 as cv
import multiprocessing
import speech_recognition as sr
import wave
import whisper
import time
import secretkey


model = whisper.load_model('base')

# Constants for audio recording
FORMAT = sr.AudioData
CHANNELS = 1
RATE = 22000
RECORD_SECONDS = 45  

recognizer = sr.Recognizer()





mp_face_mesh = mp.solutions.face_mesh

app = Flask(__name__)

eyePos = []
openai.api_key = secretkey.SECRET_KEY
conversation = [{"role": "system", "content": "You are an interviewer for a company. You will ask behavioural questions similar to What is your biggest flaw or why do you want to work here. The first message you will say is Hello my name is Prepr and I will be your interviewer. Make sure to ask the questions one at a time and wait for the response. Make it seem like a natural conversation. Make sure the questions do not get too technical and if they do and you believe you cannot continue anymore say Alright and ask another behavioral question"}]



class chattingWork:
    def addUserConvo(self, message):
        conversation.append({"role": "user", "content": message})


    def addGPTConvo(self, response):
        conversation.append({"role": "user", "content": response["choices"][0]["message"]["content"]})


    def runConvo(self):


        while True:
            # message = input()
            print("recording ... ")
            with sr.Microphone(sample_rate=RATE) as source:
                print("Recording...")
                # audio = recognizer.listen(source, timeout=None, phrase_time_limit=RECORD_SECONDS)
                audio = recognizer.listen(source)
            print("Recording stopped.")

            # Save the recorded audio to a WAV file
            with wave.open("assets/shamzy.mp3", 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.sample_width)
                wf.setframerate(RATE)
                wf.writeframes(audio.frame_data)

            result = model.transcribe('assets/shamzy.mp3', fp16 = False)
            self.addUserConvo(result['text'])
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=conversation,
                temperature=0,
            )
            self.addGPTConvo(response)
            print(response["choices"][0]["message"]["content"])
            time.sleep(3)
            os.remove("assets/shamzy.mp3")




class iris_recognition:

    cap = cv.VideoCapture(0)

    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    L_H_LEFT = [33]     
    L_H_RIGHT = [133]  
    R_H_LEFT = [362]    
    R_H_RIGHT = [263]  

    def euclidean_distance(self, point1, point2):
        x1, y1 =point1.ravel()
        x2, y2 =point2.ravel()
        distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        return distance

    def iris_position(self, iris_center, right_point, left_point):
        center_to_right_dist = self.euclidean_distance(iris_center, right_point)
        total_distance = self.euclidean_distance(right_point, left_point)
        ratio = center_to_right_dist/total_distance
        iris_position =""
        if ratio >= 2.2 and ratio <= 2.65:
            iris_position = "right"
        elif ratio >= 2.8 and ratio <= 3.2:
            iris_position = "left"
        else:
            iris_position = "center"
        return iris_position, ratio
    
    def runFullIris(self):

        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            count = 0
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame = cv.flip(frame, 1)
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  
                img_h, img_w = frame.shape[:2]
                results = face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

                    (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[self.LEFT_IRIS])
                    (r_cx,r_cy), r_radius = cv.minEnclosingCircle(mesh_points[self.RIGHT_IRIS])

                    center_left = np.array([l_cx, l_cy], dtype=np.int32)
                    center_right = np.array([r_cx, r_cy], dtype=np.int32)

                    cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
                    cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv.LINE_AA)

                    cv.circle(frame, mesh_points[self.R_H_RIGHT][0], 3, (255, 255, 255), -1, cv.LINE_AA)
                    cv.circle(frame, mesh_points[self.R_H_LEFT][0], 3, (0, 255, 255), -1, cv.LINE_AA)

                    iris_pos, ratio = self.iris_position(center_right, mesh_points[self.R_H_RIGHT], mesh_points[self.R_H_LEFT][0])

                    # print(iris_pos)
                    # print(count)
                    count += 1
                if count % 30 == 0 and count != 0:
                    eyePos.append(iris_pos)
                cv.imshow("img", frame)
                key = cv.waitKey(1)
                if key ==ord("q"):
                    x = calcPercentage(eyePos, "center")
                    print("THE ACCURACY IS ", x * 100, "%")
                    break
        self.cap.release()
        cv.destroyAllWindows()

def calcPercentage(arr, target):
    num = 0
    if len(arr) > 0:
        for x in arr:
            if x == target:
                num += 1
        return num/len(arr)
    else:
        return 0


def runIris():
    ir = iris_recognition()
    ir.runFullIris()

def runGPT():
    gpt = chattingWork()
    gpt.runConvo()


@app.route('/GetContactPercentage', methods = ['POST'])
def getContactPercentage():
    try:
        return jsonify(calcPercentage(eyePos, "center")), 200
    except:
        return jsonify({'message': 'There was a problem getting the eye contact accuracy'}), 400



if __name__ == "__main__":
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=2516))
    gpt_thread = threading.Thread(target=runGPT)


    flask_thread.daemon = True
    gpt_thread.daemon = True

    gpt_process = multiprocessing.Process(target=runGPT)

    flask_thread.start()
    gpt_thread.start()
    runIris()
    
