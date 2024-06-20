#Heatmapping by Yash C 2/8/22
#Most of the code is based off of Nicholas Renotte's videos on Mediapipe/Pose Estimation
#Adding the angles to a csv file is found on pythontutorial.net


#Added a couple of imports myself
import cv2
import mediapipe as mp
import csv
import numpy as np
import matplotlib.pyplot as plt
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
l_angle_array = []
r_angle_array = []
prof_l = []
prof_r = []

counter = 0
stage = None

def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
        
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
        
    if angle>180.0:
        angle = 360 - angle
            
    return angle

# For webcam input:
cap = cv2.VideoCapture("Golf_Swing.mp4")
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    ret, frame = cap.read()


    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    
    #Extract landmarks
    try:
        landmarks = results.pose_landmarks.landmark
        
        #I added more landmarks and angles
        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    
        l_wrist_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_wrist_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        l_shoulder_angle = calculate_angle(l_elbow, l_shoulder, l_hip)
        r_shoulder_angle = calculate_angle(r_elbow, r_shoulder, r_hip)
        l_hip_angle = calculate_angle(l_knee, l_hip, l_shoulder)
        r_hip_angle = calculate_angle(r_knee, r_hip, r_shoulder)
    
        #I added this section for comparison
        l_angle_array.append(l_wrist_angle)
        l_angle_array.append(l_shoulder_angle)
        l_angle_array.append(l_hip_angle)
        r_angle_array.append(r_wrist_angle)
        r_angle_array.append(r_shoulder_angle)
        r_angle_array.append(r_hip_angle)
    
        #cv2.putText(image, str(l_angle),tuple(np.multiply(l_elbow, [680, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        #cv2.putText(image, str(r_angle),tuple(np.multiply(r_elbow, [680, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        
        #Added to count how many swings occur
        if r_wrist_angle < 100:
            stage = "up"
        if r_wrist_angle > 130 and r_wrist_angle < 170 and stage == "up":
            counter+=1
            stage = "down"
            print(counter)
            
    except:
        pass
    
    

    
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
         mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color =(245, 66, 230), thickness=2, circle_radius = 2)
        )
    
    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    
    
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
      break
  
    #Added to read the professional angles from the csv
with open('golf_angles.csv','r') as file:
    i = 0
    reader = csv.reader(file)
    for row in reader:  
        if i == 0:
            prof_l = row
            i=i+1
        else:
            prof_r = row
 
#Added to write the professional angles into a csv
combine_array = [[l_angle_array,r_angle_array]]


with open('angles_golf.csv', 'w', newline='') as file:
    mywriter = csv.writer(file, delimiter=',')
    mywriter.writerows(combine_array)      

#Added to make it so the heatmap can actually compare the two different arrays of angles
combine1 = l_angle_array
combine1.append(0)
for i in range(len(r_angle_array)):
    combine1.append(r_angle_array[i])
       
combine2 = []
for x in range(len(prof_l)):
    combine2.append(float(prof_l[x]))
combine2.append(0)
for j in range(len(prof_r)):
    combine2.append(float(prof_r[j]))
    
if len(combine1) < len(combine2):
    x = int(len(combine1))
    extra = combine2[0:x]
    combine2 = extra
else:
    x = int(len(combine2))
    extra = combine1[0:x]
    combine1 = extra

#Added to create a heatmap and plot it
H, xedges, yedges = np.histogram2d(combine1, combine2, bins=10)

plt.pcolormesh(xedges, yedges, H)
plt.plot(x, 2*np.log(x), 'k-')
plt.set_xlabel('Professional')
plt.set_ylabel('Amateur')
plt.set_title('histogram2d')
  
cap.release()
cv2.destroyAllWindows()