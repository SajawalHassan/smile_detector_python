import cv2 as cv

smile_detector = cv.CascadeClassifier("haarcascade_smile.xml")

# Read video (change filename to 0 if need webcam)
video = cv.VideoCapture("videos/vid_test_smile.3gp")

while True:
    # Read currunt frame
    successful, frame = video.read()

    # If can't read currunt frame come out of loop
    if not successful:
        break

    # Convert to grayscale
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imshow("Smile Detector", grayscale)

    key = cv.waitKey(1)

    if key==27:
        break # If "Esc" is pressed, come out of loop

video.release()
cv.destroyAllWindows()
