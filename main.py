import cv2 as cv

face_detector = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_detector = cv.CascadeClassifier("haarcascade_smile.xml")

# Read video
video = cv.VideoCapture(0)

while True:
    # Read currunt frame
    successful, frame = video.read()

    # If can't read currunt frame come out of loop
    if not successful:
        break

    # Convert to grayscale
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Getting smile and face coordinates
    smiles = smile_detector.detectMultiScale(grayscale, 1.7, 10)
    faces = face_detector.detectMultiScale(grayscale)

    # Drawing rectangle around face and smile
    for (x, y, w, h) in smiles:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)

    cv.imshow("Smile Detector", frame)

    key = cv.waitKey(1)

    if key==27:
        break # If "Esc" is pressed, come out of loop

video.release()
cv.destroyAllWindows()
