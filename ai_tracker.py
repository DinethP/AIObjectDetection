import cv2

# Input Image
# img_file = 'car_image.jpg'

# Input video
video = cv2.VideoCapture('video2.mp4')


# Our pre-trained car & pedestrian haar features
car_tracker_features = 'car_detector.xml'
pedestrian_tracker_features = 'haarcascade_fullbody.xml'
face_cascade = 'haarcascade_frontalface_default.xml'

# create car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_features)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_features)
face_tracker = cv2.CascadeClassifier(face_cascade)



while True:
  # Read video frame by frame 
  (read_successful, frame) = video.read() 

  # check if read was successful
  if read_successful:
    #convert to grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  else:
    break

  # detect cars & pedestrians in each frame
  cars = car_tracker.detectMultiScale(grayscaled_frame)
  pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)
  faces = face_tracker.detectMultiScale(grayscaled_frame)


  # draw rectangles around cars(coordinates, colour, thickness)
  for (x, y, w, h) in cars:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

  # draw rectangles around pedestrians(coordinates, colour, thickness)
  for (x, y, w, h) in pedestrians:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

  for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)


  # Display the fram (only shows for a split second)
  cv2.imshow("Car Detector", frame)

  # Stop autoclose of image, listen for key press
  key = cv2.waitKey(1)

  # Stop if Q key is pressed
  if key == 81 or key == 113:
    break
  
#Relase video capture object
video.release()
# # create opencv image
# img = cv2.imread(img_file)

# #convert to grayscale (needed for haar cascade)
# black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # draw rectangles around cars(coordinates, colour, thickness)
# for (x, y, w, h) in cars:
#   cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# # Display the image (only shows for a split second)
# cv2.imshow("Car Detector", black_n_white)

# # Stop autoclose of image, listen for key press
# cv2.waitKey()

