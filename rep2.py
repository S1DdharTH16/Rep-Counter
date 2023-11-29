from flask import Flask, render_template, Response, request
import cv2
import math
import cv2
import mediapipe as mp
import numpy as np


app = Flask(__name__)
exercise = 'pushups'  # Default exercise selection


@app.route('/', methods=['GET', 'POST'])
def index():
    global exercise
    if request.method == 'POST':
        exercise = request.form.get('exercise')
    return render_template('index.html')


class poseDetector() :
    def __init__(self, mode=False, complexity=1, smooth_landmarks=True,
                 enable_segmentation=False, smooth_segmentation=True,
                 detectionCon=0.5, trackCon=0.5):
        
        self.mode = mode 
        self.complexity = complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth_landmarks,
                                     self.enable_segmentation, self.smooth_segmentation,
                                     self.detectionCon, self.trackCon)
        
        
    def findPose (self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
                
        return img
    
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                #finding height, width of the image printed
                h, w, c = img.shape
                #Determining the pixels of the landmarks
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
        return self.lmList
        
    def findAngle(self, img, p1, p2, p3, draw=True):   
        #Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        
        #Calculate Angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - 
                             math.atan2(y1-y2, x1-x2))
        if angle < 0:
            angle += 360
            if angle > 180:
                angle = 360 - angle
        elif angle > 180:
            angle = 360 - angle
        # print(angle)
        
        #Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255,255,255), 3)

            
            cv2.circle(img, (x1, y1), 5, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0,0,255), 2)
            cv2.circle(img, (x2, y2), 5, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0,0,255), 2)
            cv2.circle(img, (x3, y3), 5, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0,0,255), 2)
            
            cv2.putText(img, str(int(angle)), (x2-50, y2+50), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
        return angle



def perform_pushup(detector, img, lmList):
    elbow = detector.findAngle(img, 11, 13, 15)
    shoulder = detector.findAngle(img, 13, 11, 23)
    hip = detector.findAngle(img, 11, 23, 25)

    per = np.interp(elbow, (90, 160), (0, 100))
    bar = np.interp(elbow, (90, 160), (380, 50))

    form = 0
    direction = 0
    count = 0
    feedback = "Fix Form"

    if elbow > 160 and shoulder > 40 and hip > 160:
        form = 1

    if form == 1:
        if per == 0:
            if elbow <= 90 and hip > 160:
                feedback = "Up"
                if direction == 0:
                    count += 0.5
                    direction = 1
            else:
                feedback = "Fix Form"

        if per == 100:
            if elbow > 160 and shoulder > 40 and hip > 160:
                feedback = "Down"
                if direction == 1:
                    count += 0.5
                    direction = 0
            else:
                feedback = "Fix Form"

    if form == 1:
        cv2.rectangle(img, (580, 50), (600, 380), (0, 255, 0), 3)
        cv2.rectangle(img, (580, int(bar)), (600, 380), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (565, 430), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0), 2)

    cv2.rectangle(img, (0, 380), (100, 480), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(int(count)), (25, 455), cv2.FONT_HERSHEY_PLAIN, 5,
                (255, 0, 0), 5)

    cv2.rectangle(img, (500, 0), (640, 40), (255, 255, 255), cv2.FILLED)
    cv2.putText(img, feedback, (500, 40), cv2.FONT_HERSHEY_PLAIN, 2,
                (0, 255, 0), 2)



def perform_squats(detector, img, lmList):
    left_hip = lmList[11]
    left_knee = lmList[13]
    left_ankle = lmList[15]
    right_hip = lmList[12]
    right_knee = lmList[14]
    right_ankle = lmList[16]

    left_leg_angle = detector.findAngle(img, 11, 13, 15, draw=False)
    right_leg_angle = detector.findAngle(img, 12, 14, 16, draw=False)

    per = np.interp(left_leg_angle, (90, 160), (0, 100))
    bar = np.interp(left_leg_angle, (90, 160), (380, 50))

    form = 0
    direction = 0
    count = 0
    feedback = "Performing Squats"

    left_threshold_angle = 160
    right_threshold_angle = 160

    if left_leg_angle > left_threshold_angle and right_leg_angle > right_threshold_angle:
        form = 1

    if form == 1:
        if per == 0:
            if left_leg_angle <= 90 and right_leg_angle <= 90:
                feedback = "Up"
                if direction == 0:
                    count += 0.5
                    direction = 1
            else:
                feedback = "Fix Form"

        if per == 100:
            if left_leg_angle > 160 and right_leg_angle > 160:
                feedback = "Down"
                if direction == 1:
                    count += 0.5
                    direction = 0
            else:
                feedback = "Fix Form"

    if form == 1:
        cv2.rectangle(img, (580, 50), (600, 380), (0, 255, 0), 3)
        cv2.rectangle(img, (580, int(bar)), (600, 380), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (565, 430), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0), 2)

    cv2.rectangle(img, (0, 380), (100, 480), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(int(count)), (25, 455), cv2.FONT_HERSHEY_PLAIN, 5,
                (255, 0, 0), 5)

    cv2.rectangle(img, (500, 0), (640, 40), (255, 255, 255), cv2.FILLED)
    cv2.putText(img, feedback, (500, 40), cv2.FONT_HERSHEY_PLAIN, 2,
                (0, 255, 0), 2)

    return form, direction, count



# Function to perform different exercises based on exercise name
def perform_exercise(exercise, detector, img, lmList):
    if exercise == 'pushups':
        perform_pushup(detector, img, lmList)

    elif exercise == 'squats':
        print("doing sw")
        perform_squats(detector, img, lmList)
    else:
        pass  # Handle other exercises if necessary


def generate_frames():
    cap = cv2.VideoCapture(0)
    detector = poseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)

        if len(lmList) != 0:
            perform_exercise(exercise, detector, img, lmList)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()


@app.route('/video_feed', methods=['POST'])
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)