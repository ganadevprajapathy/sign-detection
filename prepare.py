import cv2
import csv
import mediapipe as mp
from pyparsing import Each
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
from app import calc_bounding_rect, calc_landmark_list, pre_process_landmark, pre_process_point_history, logging_csv
import copy

from os import listdir
from os.path import isfile, join
path = '/home/ganadev/Documents/sign-detection/training2/'
folder = [d for d in listdir(path)]
count = 0
for each in folder:
  IMAGE_FILES = [path + each + '/' + f for f in listdir(path + '/' + each) if isfile(join(path + '/' + each, f))]
#   IMAGE_FILES = [path + each + '/' + f for f in listdir(path + '/' + each) if (isfile(join(path + '/' + each, f) and f.endswith('.jpg')))]
  print('Folder: ' + each)
  
  with mp_hands.Hands(
      static_image_mode=True,
      max_num_hands=2,
      min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):
      if not file.endswith('.jpg'):
        continue
      print('file: ' + file)
      # Read image #####################################################
      
      # Read an image, flip it around y-axis for correct handedness output 
      image = cv2.flip(cv2.imread(file), 1) # Mirror display
      # cv2.imshow('image ', cv2.flip(image, 1))
      debug_image = copy.deepcopy(image)

      # Detection implementation #############################################################
      
      # Convert the BGR image to RGB before processing.
      # image.flags.writeable = False
      results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      # image.flags.writeable = True

      #  ####################################################################
      if results.multi_hand_landmarks is not None:
          for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
              # Bounding box calculation
              brect = calc_bounding_rect(debug_image, hand_landmarks)
              # Landmark calculation
              landmark_list = calc_landmark_list(debug_image, hand_landmarks)

              # Conversion to relative coordinates / normalized coordinates
              pre_processed_landmark_list = pre_process_landmark(
                  landmark_list)
              # pre_processed_point_history_list = pre_process_point_history(
              #     debug_image, point_history)
              # # Write to the dataset file

          csv_path = 'model/keypoint_classifier/keypoint.csv'
          with open(csv_path, 'a', newline="") as f:
              writer = csv.writer(f)
              writer.writerow([count, *pre_processed_landmark_list])
    csv_label = 'model/keypoint_classifier/keypoint_classifier_label.csv'
    with open(csv_label, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([each])
    count+=1