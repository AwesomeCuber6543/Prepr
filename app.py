import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import threading
from flask import Flask, jsonify, request
import os, sys
mp_face_mesh = mp.solutions.face_mesh





app = Flask(__name__)

eyePos = []





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

                    print(iris_pos)
                    print(count)
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


@app.route('/GetContactPercentage', methods = ['POST'])
def getContactPercentage():
    try:
        return jsonify(calcPercentage(eyePos, "center")), 200
    except:
        return jsonify({'message': 'There was a problem getting the eye contact accuracy'}), 400



if __name__ == "__main__":
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=2516))

    flask_thread.daemon = True


    flask_thread.start()
    runIris()
