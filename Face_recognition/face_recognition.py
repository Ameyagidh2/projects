# Here we use haar cascade classifier which moves through the entire image and finds the lights and facial features
# and finds the components of the face rgb values and main features
import cv2
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # ret to return true if a frame is returned
    # frame is the next picture frame
    if ret == False:
        continue
        # skip the element if no frame seen ie camera not working
        # ret checks if camera works or not
        

    cv2.imshow("video frame",frame)
    #showing the image
    key_pressed = cv2.waitKey(1) & 0xFF

    if key_pressed == ord('q'):
        break
        #ending the program at q key pressed

cap.release()
cv2.destroyAllWindows()
#delete the cache and all the data

# store face data as an array

