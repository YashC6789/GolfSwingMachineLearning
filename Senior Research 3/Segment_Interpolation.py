import cv2
import mediapipe as mp
import numpy as np
import time as time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
#from scipy import PPoly
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
l_angle_array = []
r_angle_array = []
prof_l = []
prof_r = []
startingFrame = 0

counter = 0
stage = None


def calculate_angle(a,b,c):
    a = np.array(a) #First
    b = np.array(b) #Mid
    c = np.array(c) #End
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1] - b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle>180.0:
        angle = 360-angle
        
    return angle

def remove_duplicates(test_list):
    res = []
    [res.append(x) for x in test_list if x not in res]
    return res

def find_StartingFrame(leftElbowArray):
   for frame in range(len(leftElbowArray)):
       if leftElbowArray[frame] < 170:
           startingFrame = frame
           break
   return startingFrame

def slope(coordinate_one_x,coordinate_one_y, coordinate_two_x,coordinate_two_y):

    slope = (coordinate_two_y - coordinate_one_y)/(coordinate_two_x-coordinate_one_x)
    return slope

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

#leftElbowArray
leftElbowArray = []
leftWristArray = []
rightWristArray = []
leftShoulderArray = []
rightShoulderArray = []
leftHipArray = []
rightHipArray = []

frameArray = []

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)


# For webcam input: 0 + cv2.CAP_DSHOW GolfSwing_MOV_SparkVideo.mp4
#cap = cv2.VideoCapture("GolfSwing_MOV_SparkVideo.mp4")
#cap = cv2.VideoCapture("JustinThomasProSwing.mp4")

cap = cv2.VideoCapture("TigerWoodsVideo.mp4")


#Start of determining the framenumber 
frame_rate = 10
prev = 0
count = 0
elapsedTime = 0
prevTime = 0
prevPosMsec = 0
startTimeStamp = time.time()
sumTime = 0
shoulder_slope = []

with mp_pose.Pose(
        
    
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    
    time_elapsed = time.time() - prev
    success, image = cap.read()
    
    if time_elapsed > 1./frame_rate:
        prev = time.time()

        # Do something with your image here.
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break
  
    

    count += 1
    frameNum = cap.get(cv2.CAP_PROP_POS_FRAMES)
    posMsec = cap.get(cv2.CAP_PROP_POS_MSEC)
    
    elapsedPosTime = posMsec - prevPosMsec
    prevPosMsec = posMsec
    
    elapsedTime = (time.time() - prevTime) * 1000
    prevTime = time.time()
    sumTime += elapsedTime
    
# =============================================================================
#     if frameNum == 155:
#         cv2.imwrite("frame%d.jpg" % frameNum, image)    
# =============================================================================

# =============================================================================
#     if frameNum >= 130 and frameNum <= 145:
#         cv2.imwrite("frame%d.jpg" % frameNum, image)
# =============================================================================
    
    #print("count: ", count, 'frame: ', frameNum, "deltamsecs: ", elapsedTime, "Time: ", sumTime)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    #image = cv2.flip(image,1)
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image,1)
    results = pose.process(image)
    
    # Extract landmarks (coordinates)
    try:
        landmarks = results.pose_landmarks.landmark
        #print(landmarks)
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        left_wrist = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_wrist = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_shoulder = calculate_angle(left_elbow, left_shoulder, left_hip)
        right_shoulder = calculate_angle(right_elbow, right_shoulder, right_hip)
        left_hip = calculate_angle(left_knee, left_hip, left_shoulder)
        right_hip = calculate_angle(right_knee, right_hip, right_shoulder)
        
        
        left_leg = (calculate_angle(left_hip, left_knee, left_ankle))
        right_leg = (calculate_angle(right_hip, right_knee, right_ankle))
        left_elbow = (calculate_angle(left_shoulder, left_elbow, left_wrist))
        right_elbow = (calculate_angle(right_shoulder, right_elbow, right_wrist))
        
        shoulder_slope.append(slope(left_shoulder[0], left_shoulder[1], right_shoulder[0], right_shoulder[1]))
        #print(frameNum, left_elbow, left_leg)
        
        
        
        #Visualize
# =============================================================================
#         cv2.putText(image, str(left_elbow), 
#                     tuple(np.multiply(left_elbow, [640,480]).astype(int)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2,cv2.LINE_AA)
# =============================================================================
        
        
        
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,500)
        fontScale              = 1
        fontColor              = (255,255,255)
        thickness              = 1
        lineType               = 2
        
        cv2.putText(image,str(frameNum), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)
        
        leftElbowArray.append(left_elbow)
        leftWristArray.append(left_wrist)
        rightWristArray.append(right_wrist)
        leftShoulderArray.append(left_shoulder)
        rightShoulderArray.append(right_shoulder)
        leftHipArray.append(left_hip)
        rightHipArray.append(right_hip)
        
        frameArray.append(image)      
        
       # print(frameNum, left_elbow, right_leg, left_leg)
        
    except: 
        pass
    
    
    # Draw the pose annotation on the image.q
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()

for lndmrk in mp_pose.PoseLandmark:
    print(lndmrk)
    
    
    
    
#Start of the Segmentation    
    
startingFrame = find_StartingFrame(leftElbowArray) 
swingFrames = []
backswingFrames = []
impactFrames = []
followThroughFrames = []
impactFrame = 0
swingEndFrame = 0

    

for frameNumber in range(startingFrame, len(leftElbowArray)):
    if(leftElbowArray[frameNumber] <= 169):
        backswingFrames.append(frameNumber)
    else:
        break

cv2.imwrite("backswingstart%d.jpg" % (startingFrame-5), frameArray[startingFrame-5])
cv2.imwrite("backswingend%d.jpg" % backswingFrames[-1], frameArray[backswingFrames[-1]])

for frameNumber in range(backswingFrames[-1], len(shoulder_slope)):
    if shoulder_slope[frameNumber] < 0 and shoulder_slope[frameNumber+1] > 0.3:
        impactFrame = frameNumber+1
        break
    
cv2.imwrite("impactFrame%d.jpg" % impactFrame, frameArray[impactFrame])

for frameNumber in range(backswingFrames[-1], impactFrame+1):
    impactFrames.append(frameNumber)

for frameNumber in range(impactFrame, len(shoulder_slope)):
    if shoulder_slope[frameNumber] < 0 and shoulder_slope[frameNumber+1] > 0:
        swingEndFrame = frameNumber+1
        break
    
cv2.imwrite("swingEndFrame%d.jpg" % swingEndFrame, frameArray[swingEndFrame])


for frameNumber in range(impactFrame, swingEndFrame+1):
    followThroughFrames.append(frameNumber)


# =============================================================================
# for i in range(len(shoulder_slope)):
#     print(shoulder_slope[i], i)
# =============================================================================
    
#x_range = np.linspace(1,len(leftElbowArray) + 1,len(leftElbowArray))
#y_range = np.array(leftElbowArray)

#x2_range = np.linspace(1,len(rightElbowArray) + 1,len(rightElbowArray))
#y2_range = np.array(rightElbowArray)

#plt.scatter(x_range, y_range)
#plt.scatter(x2_range, y2_range)
plt.show()

    #Start of the Interpolation using Scipy Spline

plt.figure()
#plt.plot(x, y, 'x', xnew, ynew, xnew, np.sin(xnew), x, y, 'b')
plt.legend(['Linear', 'Cubic Spline', 'True'])
x = np.arange(len(leftWristArray))
y = np.array(leftWristArray)
cs = CubicSpline(x,y)
xs = np.arange(-0.5, len(leftWristArray), 0.1)
#fig, ax = plt.subplots(figsize=(6.5, 4))
#ax.plot(x, y, 'o', label='data')
#ax.plot(xs, cs(xs), label="S")

#ax.set_xlim(-0.5, len(leftElbowArray))
#ax.legend(loc='lower left')
#plt.title('Cubic-spline interpolation')

for i in range(len(cs.x)-1):
    xs = np.linspace(cs.x[i], cs.x[i+1], 100)
    plt.plot(xs, np.polyval(cs.c[:,i], xs - cs.x[i]))

#sci = scipy.interpolate.PPoly.from_spline(cs,xs, extrapolate=True)
#plt.show()

majorSwingFrames = [startingFrame -5, backswingFrames[-1], impactFrame, swingEndFrame]
#print(majorSwingFrames)

#splineBackSwing = []
#splineImpact = []
#splineFollowThrough = []
l_w = []
for i in np.arange(majorSwingFrames[0], majorSwingFrames[1], (majorSwingFrames[1]-majorSwingFrames[0])/30.0):
    l_w.append(cs(i))
    #print(i,cs(i))
for i in np.arange(majorSwingFrames[1], majorSwingFrames[2], (majorSwingFrames[2]-majorSwingFrames[1])/15.0):
    l_w.append(cs(i))  
for i in np.arange(majorSwingFrames[2], majorSwingFrames[3], (majorSwingFrames[3]-majorSwingFrames[2])/15.0):
    l_w.append(cs(i))
    

plt.figure()
#plt.plot(x, y, 'x', xnew, ynew, xnew, np.sin(xnew), x, y, 'b')
plt.legend(['Linear', 'Cubic Spline', 'True'])
x = np.arange(len(rightWristArray))
y = np.array(rightWristArray)
cs = CubicSpline(x,y)
xs = np.arange(-0.5, len(rightWristArray), 0.1)
#fig, ax = plt.subplots(figsize=(6.5, 4))
#ax.plot(x, y, 'o', label='data')
#ax.plot(xs, cs(xs), label="S")

#ax.set_xlim(-0.5, len(rightWristArray))
#ax.legend(loc='lower left')
plt.title('Cubic-spline interpolation')

for i in range(len(cs.x)-1):
    xs = np.linspace(cs.x[i], cs.x[i+1], 100)
    plt.plot(xs, np.polyval(cs.c[:,i], xs - cs.x[i]))

#sci = scipy.interpolate.PPoly.from_spline(cs,xs, extrapolate=True)
#plt.show()

majorSwingFrames = [startingFrame -5, backswingFrames[-1], impactFrame, swingEndFrame]
#print(majorSwingFrames)

#splineBackSwing2 = []
#splineImpact2 = []
#splineFollowThrough2 = []
r_w = []
for i in np.arange(majorSwingFrames[0], majorSwingFrames[1], (majorSwingFrames[1]-majorSwingFrames[0])/30.0):
    r_w.append(cs(i))
    #print(i,cs(i))
for i in np.arange(majorSwingFrames[1], majorSwingFrames[2], (majorSwingFrames[2]-majorSwingFrames[1])/15.0):
    r_w.append(cs(i))  
for i in np.arange(majorSwingFrames[2], majorSwingFrames[3], (majorSwingFrames[3]-majorSwingFrames[2])/15.0):
    r_w.append(cs(i))

plt.figure()
#plt.plot(x, y, 'x', xnew, ynew, xnew, np.sin(xnew), x, y, 'b')
plt.legend(['Linear', 'Cubic Spline', 'True'])
x = np.arange(len(leftShoulderArray))
y = np.array(leftShoulderArray)
cs = CubicSpline(x,y)
xs = np.arange(-0.5, len(leftShoulderArray), 0.1)
#fig, ax = plt.subplots(figsize=(6.5, 4))
#ax.plot(x, y, 'o', label='data')
#ax.plot(xs, cs(xs), label="S")

#ax.set_xlim(-0.5, len(rightWristArray))
#ax.legend(loc='lower left')
plt.title('Cubic-spline interpolation')

for i in range(len(cs.x)-1):
    xs = np.linspace(cs.x[i], cs.x[i+1], 100)
    plt.plot(xs, np.polyval(cs.c[:,i], xs - cs.x[i]))

#sci = scipy.interpolate.PPoly.from_spline(cs,xs, extrapolate=True)
#plt.show()

majorSwingFrames = [startingFrame -5, backswingFrames[-1], impactFrame, swingEndFrame]
#print(majorSwingFrames)

#splineBackSwing2 = []
#splineImpact2 = []
#splineFollowThrough2 = []
l_s = []
for i in np.arange(majorSwingFrames[0], majorSwingFrames[1], (majorSwingFrames[1]-majorSwingFrames[0])/30.0):
    l_s.append(cs(i))
    #print(i,cs(i))
for i in np.arange(majorSwingFrames[1], majorSwingFrames[2], (majorSwingFrames[2]-majorSwingFrames[1])/15.0):
    l_s.append(cs(i))  
for i in np.arange(majorSwingFrames[2], majorSwingFrames[3], (majorSwingFrames[3]-majorSwingFrames[2])/15.0):
    l_s.append(cs(i))
    
plt.figure()
#plt.plot(x, y, 'x', xnew, ynew, xnew, np.sin(xnew), x, y, 'b')
plt.legend(['Linear', 'Cubic Spline', 'True'])
x = np.arange(len(rightShoulderArray))
y = np.array(rightShoulderArray)
cs = CubicSpline(x,y)
xs = np.arange(-0.5, len(rightShoulderArray), 0.1)
#fig, ax = plt.subplots(figsize=(6.5, 4))
#ax.plot(x, y, 'o', label='data')
#ax.plot(xs, cs(xs), label="S")

#ax.set_xlim(-0.5, len(rightWristArray))
#ax.legend(loc='lower left')
plt.title('Cubic-spline interpolation')

for i in range(len(cs.x)-1):
    xs = np.linspace(cs.x[i], cs.x[i+1], 100)
    plt.plot(xs, np.polyval(cs.c[:,i], xs - cs.x[i]))

#sci = scipy.interpolate.PPoly.from_spline(cs,xs, extrapolate=True)
#plt.show()

majorSwingFrames = [startingFrame -5, backswingFrames[-1], impactFrame, swingEndFrame]
#print(majorSwingFrames)

#splineBackSwing2 = []
#splineImpact2 = []
#splineFollowThrough2 = []
r_s = []
for i in np.arange(majorSwingFrames[0], majorSwingFrames[1], (majorSwingFrames[1]-majorSwingFrames[0])/30.0):
    r_s.append(cs(i))
    #print(i,cs(i))
for i in np.arange(majorSwingFrames[1], majorSwingFrames[2], (majorSwingFrames[2]-majorSwingFrames[1])/15.0):
    r_s.append(cs(i))  
for i in np.arange(majorSwingFrames[2], majorSwingFrames[3], (majorSwingFrames[3]-majorSwingFrames[2])/15.0):
    r_s.append(cs(i))
    
plt.figure()
#plt.plot(x, y, 'x', xnew, ynew, xnew, np.sin(xnew), x, y, 'b')
plt.legend(['Linear', 'Cubic Spline', 'True'])
x = np.arange(len(leftHipArray))
y = np.array(leftHipArray)
cs = CubicSpline(x,y)
xs = np.arange(-0.5, len(leftHipArray), 0.1)
#fig, ax = plt.subplots(figsize=(6.5, 4))
#ax.plot(x, y, 'o', label='data')
#ax.plot(xs, cs(xs), label="S")

#ax.set_xlim(-0.5, len(rightWristArray))
#ax.legend(loc='lower left')
plt.title('Cubic-spline interpolation')

for i in range(len(cs.x)-1):
    xs = np.linspace(cs.x[i], cs.x[i+1], 100)
    plt.plot(xs, np.polyval(cs.c[:,i], xs - cs.x[i]))

#sci = scipy.interpolate.PPoly.from_spline(cs,xs, extrapolate=True)
#plt.show()

majorSwingFrames = [startingFrame -5, backswingFrames[-1], impactFrame, swingEndFrame]
#print(majorSwingFrames)

#splineBackSwing2 = []
#splineImpact2 = []
#splineFollowThrough2 = []
l_h = []
for i in np.arange(majorSwingFrames[0], majorSwingFrames[1], (majorSwingFrames[1]-majorSwingFrames[0])/30.0):
    l_h.append(cs(i))
    #print(i,cs(i))
for i in np.arange(majorSwingFrames[1], majorSwingFrames[2], (majorSwingFrames[2]-majorSwingFrames[1])/15.0):
    l_h.append(cs(i))  
for i in np.arange(majorSwingFrames[2], majorSwingFrames[3], (majorSwingFrames[3]-majorSwingFrames[2])/15.0):
    l_h.append(cs(i))

plt.figure()
#plt.plot(x, y, 'x', xnew, ynew, xnew, np.sin(xnew), x, y, 'b')
plt.legend(['Linear', 'Cubic Spline', 'True'])
x = np.arange(len(rightHipArray))
y = np.array(rightHipArray)
cs = CubicSpline(x,y)
xs = np.arange(-0.5, len(rightHipArray), 0.1)
#fig, ax = plt.subplots(figsize=(6.5, 4))
#ax.plot(x, y, 'o', label='data')
#ax.plot(xs, cs(xs), label="S")

#ax.set_xlim(-0.5, len(rightWristArray))
#ax.legend(loc='lower left')
plt.title('Cubic-spline interpolation')

for i in range(len(cs.x)-1):
    xs = np.linspace(cs.x[i], cs.x[i+1], 100)
    plt.plot(xs, np.polyval(cs.c[:,i], xs - cs.x[i]))

#sci = scipy.interpolate.PPoly.from_spline(cs,xs, extrapolate=True)
#plt.show()

majorSwingFrames = [startingFrame -5, backswingFrames[-1], impactFrame, swingEndFrame]
#print(majorSwingFrames)

#splineBackSwing2 = []
#splineImpact2 = []
#splineFollowThrough2 = []
r_h = []
for i in np.arange(majorSwingFrames[0], majorSwingFrames[1], (majorSwingFrames[1]-majorSwingFrames[0])/30.0):
    r_h.append(cs(i))
    #print(i,cs(i))
for i in np.arange(majorSwingFrames[1], majorSwingFrames[2], (majorSwingFrames[2]-majorSwingFrames[1])/15.0):
    r_h.append(cs(i))  
for i in np.arange(majorSwingFrames[2], majorSwingFrames[3], (majorSwingFrames[3]-majorSwingFrames[2])/15.0):
    r_h.append(cs(i))

