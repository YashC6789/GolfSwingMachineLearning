# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 14:38:54 2022

@author: 815182
"""

import cv2
import csv
import mediapipe as mp
import numpy as np
import time as time
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from pandas import *
l_angle_array = []
r_angle_array = []
combine1 = []
combine2 = []
original1 = []
original2 = []
prof_l = []
prof_r = []
percentage = 0
addition = 0
p_array = []
starter1 = []
histo_array = []

#from scipy import PPoly


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

#Start of the Interpolation using Scipy Spline
def scipy_spline(angle_array):

    plt.legend(['Linear', 'Cubic Spline', 'True'])
    x = np.arange(len(angle_array))
    y = np.array(angle_array)
    cs = CubicSpline(x,y)
    xs = np.arange(-0.5, len(angle_array), 0.1)
    #fig, ax = plt.subplots(figsize=(6.5, 4))
    #ax.plot(x, y, 'o', label='data')
    #ax.plot(xs, cs(xs), label="S")
    
    #ax.set_xlim(-0.5, len(angle_array))
    #ax.legend(loc='lower left')
    #plt.title('Cubic-spline interpolation')
    
# =============================================================================
#     for i in range(len(cs.x)-1):
#         xs = np.linspace(cs.x[i], cs.x[i+1], 100)
#         plt.plot(xs, np.polyval(cs.c[:,i], xs - cs.x[i]))
# =============================================================================
    
    #sci = scipy.interpolate.PPoly.from_spline(cs,xs, extrapolate=True)
    #plt.show()
    
    majorSwingFrames = [startingFrame -5, backswingFrames[-1], impactFrame, swingEndFrame]
    print(majorSwingFrames)
    
    splineBackSwing = []
    splineImpact = []
    splineFollowThrough = []
    for i in np.arange(majorSwingFrames[0], majorSwingFrames[1], (majorSwingFrames[1]-majorSwingFrames[0])/30.0):
        splineBackSwing.append(cs(i))
        print(i,cs(i))
    for i in np.arange(majorSwingFrames[1], majorSwingFrames[2], (majorSwingFrames[2]-majorSwingFrames[1])/15.0):
        splineImpact.append(cs(i))  
    for i in np.arange(majorSwingFrames[2], majorSwingFrames[3], (majorSwingFrames[3]-majorSwingFrames[2])/15.0):
        splineFollowThrough.append(cs(i))
        
    return splineBackSwing, splineImpact, splineFollowThrough


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

#leftElbowArray
leftElbowArray = []
rightLegArray = []
leftLegArray = []

leftElbow = []
leftShoulder = []
leftHip = []
rightElbow = []
rightShoulder = []
rightHip = []
l_array = []
r_array = []

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
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        
        left_leg_angle = (calculate_angle(left_hip, left_knee, left_ankle))
        right_leg_angle = (calculate_angle(right_hip, right_knee, right_ankle))
        
        left_shoulder_angle = (calculate_angle(left_elbow, left_shoulder, left_hip))
        right_shoulder_angle = (calculate_angle(right_elbow, right_shoulder, right_hip))
        
        left_elbow_angle = (calculate_angle(left_shoulder, left_elbow, left_wrist))
        right_elbow_angle = (calculate_angle(right_shoulder, right_elbow, right_wrist))
        
        left_hip_angle = (calculate_angle(left_shoulder, left_hip, left_knee))
        right_hip_angle = (calculate_angle(right_shoulder, right_hip, right_knee))

        
        
        leftElbow.append(left_elbow_angle)
        leftHip.append(left_hip_angle)
        leftShoulder.append(left_shoulder_angle)
        
        rightElbow.append(right_elbow_angle)
        rightHip.append(right_hip_angle)
        rightShoulder.append(right_shoulder_angle)
        
        shoulder_slope.append(slope(left_shoulder[0], left_shoulder[1], right_shoulder[0], right_shoulder[1]))
        print(frameNum, left_elbow_angle, left_leg_angle)
        
        
        
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
        
        leftElbowArray.append(left_elbow_angle)
        rightLegArray.append(right_leg_angle)
       # leftLegArray.append()
        
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
    
    
swingFrames = []
startingFrame = find_StartingFrame(leftElbowArray)
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


#Call the scipy_spline method from above for interpolation

backswing_leftHip, impact_leftHip, followthrough_leftHip = scipy_spline(leftHip)
backswing_leftElbow, impact_leftElbow, followthrough_leftElbow = scipy_spline(leftElbow)
backswing_leftShoulder, impact_leftShoulder, followthrough_leftShoulder = scipy_spline(leftShoulder)

backswing_rightHip, impact_rightHip, followthrough_rightHip = scipy_spline(rightHip)
backswing_rightElbow, impact_rightElbow, followthrough_rightElbow = scipy_spline(rightElbow)
backswing_rightShoulder, impact_rightShoulder, followthrough_rightShoulder = scipy_spline(rightShoulder)

with open('oldTiger.csv','r') as file:
    i = 0
    reader = csv.reader(file)
    for row in reader:  
        if i == 0:
            og_1 = row
            i=i+1
        else:
            og_2 = row
           
original2 = []
for m in range(len(og_1)):
    original2.append(float(og_1[m]))
    original2.append(0)
for j in range(len(og_2)):
    original2.append(float(og_2[j]))
        
with open('oldTiger.csv','r') as file:
    i = 0
    reader = csv.reader(file)
    for row in reader:  
        if i == 0:
            og1 = row
            i=i+1
        else:
            og2 = row
           
original1 = []
for m in range(len(og1)):
    original1.append(float(og1[m]))
    original1.append(0)
for j in range(len(og2)):
    original1.append(float(og2[j]))



val2 = input('Would you like to analyze a frame(1) or entire swing(2)?')

if(int(val2) == 1):
    
    for index1 in range(0,60):
        # reading data from a csv file 'Data.csv'
        with open('fixedangles.csv', newline='') as file:
        
            reader = csv.reader(file, delimiter = ',')
          
            # output list to store all rows
            Output = []
            for row in reader:
                starter1.append(float(row[index1]))
            
            
        with open('youngTiger.csv', newline='') as file:
        
            reader = csv.reader(file, delimiter = ',')
          
            # output list to store all rows
            Output = []
            for row in reader:
                combine2.append(float(row[index1]))
            
            
        if(int(index1) > 29 and int(index1) < 45):
            index1 = index1 - 30
            combine1.append(impact_leftHip[index1])
            combine1.append(impact_leftElbow[index1])
            combine1.append(impact_leftShoulder[index1])
            combine1.append(impact_rightHip[index1])
            combine1.append(impact_rightElbow[index1])
            combine1.append(impact_rightShoulder[index1])
            index1 = index1 + 30
        if(int(index1) > 44):
            index1 = index1 - 45
            combine1.append(followthrough_leftHip[index1])
            combine1.append(followthrough_leftElbow[index1])
            combine1.append(followthrough_leftShoulder[index1])
            combine1.append(followthrough_rightHip[index1])
            combine1.append(followthrough_rightElbow[index1])
            combine1.append(followthrough_rightShoulder[index1])
            index1 = index1 + 45
        if(int(index1) < 30):
            combine1.append(backswing_leftHip[index1])
            combine1.append(backswing_leftElbow[index1])
            combine1.append(backswing_leftShoulder[index1])
            combine1.append(backswing_rightHip[index1])
            combine1.append(backswing_rightElbow[index1])
            combine1.append(backswing_rightShoulder[index1])
            
            #Added to create a heatmap and plot it
        H1, xedges1, yedges1 = np.histogram2d(combine1, combine2, bins=(3,3))
        H2, xedges2, yedges2 = np.histogram2d(combine1, starter1, bins=(3,3))
        
        for i in range(len(H1)):
            if(H1[i,i] == 0 and H2[i,i] == 0):
                percentage = 100
            else:
                if(H2[i,i] != 0):
                    percentage = (H1[i,i]/H2[i,i])*100
                else:
                    percentage = 0
            p_array.append(percentage)
            
        for y in range(len(p_array)):
            addition+=p_array[y]
        histo_array.append(int(addition/3))
        addition = 0
        
        combine1 = []
        combine2 = []
        starter1 = []
        p_array = []
    
    p1 = plt.bar(np.arange(len(histo_array)), histo_array)
    
    for rect1 in p1:
        height = rect1.get_height()
        plt.annotate( "{}%".format(height),(rect1.get_x() + rect1.get_width()/2, height+.05),ha="center",va="bottom",fontsize=5)
    
    plt.show()
    
else:
#with open('fixedangles.csv','r') as file:
#    i = 0
#    reader = csv.reader(file)
#    for row in reader:  
#        if i == 0:
#            prof_l = row
#            i=i+1
#        else:
#            prof_r = row

    with open('oldTiger.csv','r') as file:
        i = 0
        reader = csv.reader(file)
        for row in reader:  
            if i == 0:
                prof_l = row
                i=i+1
            else:
                prof_r = row
           
    combine2 = []
    for m in range(len(prof_l)):
        combine2.append(float(prof_l[m]))
    combine2.append(0)
    for j in range(len(prof_r)):
        combine2.append(float(prof_r[j]))
        
    with open('Tiger_Young.csv','r') as file:
        i = 0
        reader = csv.reader(file)
        for row in reader:  
            if i == 0:
                amat_l = row
                i=i+1
            else:
                amat_r = row
           
    combine1 = []
    for m in range(len(amat_l)):
        combine1.append(float(amat_l[m]))
    combine1.append(0)
    for j in range(len(amat_r)):
        combine1.append(float(amat_r[j]))
        
       
#        combine2 = []
#    for m in range(len(prof_l)):
#        combine2.append(float(prof_l[m]))
#        combine2.append(0)
#    for j in range(len(prof_r)):
#        combine2.append(float(prof_r[j]))


    #Added to create a heatmap and plot it
    
    H1, xedges1, yedges1 = np.histogram2d(combine1, combine2, bins=(5,5))
    #H2, xedges2, yedges2 = np.histogram2d(original1, original2, bins=(5,5))


    plt.pcolormesh(xedges1, yedges1, H1, cmap = 'inferno')
    plt.plot(xedges1[4], 2*np.log(xedges1[4]), 'k-')
    plt.show()
    plt.set_xlabel('Professional')
    plt.set_ylabel('Amateur')
    plt.set_title('histogram2d')
    
  
cap.release()
cv2.destroyAllWindows()

