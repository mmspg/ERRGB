"""
Script for capturing video from the camera module on the raspberry pi.
"""


# import the opencv library
import cv2
import os
from datetime import datetime

# define a video capture object
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
now = datetime.now()
dt_string = now.strftime("%d%m%Y_%H%M%S")
data_path = os.path.join(os.getcwd(), 'data')

if not os.path.exists(data_path):
    os.mkdir(data_path)

out = cv2.VideoWriter(os.path.join(data_path, dt_string + '.avi'),
                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height))

print("Recording video " + dt_string)

while(True):

    # Capture the video frame
    # by frame
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    out.write(frame)

    # Display the resulting frame
    # cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
