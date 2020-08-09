import cv2

# Our pre-trained car & pedestrian haar features
car_tracker_features = 'car_detector.xml'
pedestrian_tracker_features = 'haarcascade_fullbody.xml'
face_cascade = 'haarcascade_frontalface_default.xml'

# create car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_features)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_features)
face_tracker = cv2.CascadeClassifier(face_cascade)
  
  
def scan_image(filename):
  # Input image
  img = cv2.imread(filename)
  print("Processing "+ filename+"...")
  # convert to grayscale
  grayscaled_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # call tracker to track objects
  tracker(img, grayscaled_image)
  # Display the fram (only shows for a split second)
  cv2.imshow("AI Tracker", img)
  print("Processing complete!\n")

  # Stop autoclose of image, listen for key press
  cv2.waitKey()
  # close all open windows
  cv2.destroyAllWindows()


def scan_video(filename):
  # Input video
  video = cv2.VideoCapture(filename)
  print("Processing "+ filename+"...")

  while True:
    # Read video frame by frame 
    (read_successful, frame) = video.read() 

    # check if read was successful
    if read_successful:
      #convert to grayscale
      grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # call tracker to track objects
      tracker(frame, grayscaled_frame)
      # Display the fram (only shows for a split second)
      cv2.imshow("AI Tracker", frame)
      # Stop autoclose of image, listen for key press
      key = cv2.waitKey(1)

      # Stop if Q key is pressed
      if key == 81 or key == 113:
        print("Aborted by user\n")
        cv2.destroyAllWindows()
        break
    else:
      print("Processing complete!\n")
      break
  #Relase video capture object
  video.release()


def tracker(image, grayscaled_image):
  # detect cars & pedestrians in each frame
  cars = car_tracker.detectMultiScale(grayscaled_image)
  pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_image)
  faces = face_tracker.detectMultiScale(grayscaled_image)

  # draw rectangles around cars(coordinates, colour, thickness)
  for (x, y, w, h) in cars:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
  # draw rectangles around pedestrians(coordinates, colour, thickness)
  for (x, y, w, h) in pedestrians:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)
  # draw rectangles around faces(coordinates, colour, thickness)
  for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)


if __name__ == "__main__":
  print("Welcome to the AI video tracker\n")
  
  input_val = ''
  while input_val != "q":
    print("Do you want to track an image or video?\nEnter 1 for image\nEnter 2 for video\nEnter q to exit\n")
    input_val = input("Choice: ")
    if input_val == "1":
      input_name = input("Enter the filename: ")
      scan_image(input_name)
    elif input_val == "2":
      input_name = input("Enter the filename: ")
      scan_video(input_name)
  print("Done")