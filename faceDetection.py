# Importing required packages
import argparse
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-vw", "--isVideoWriter", type=bool, default=False)
args = vars(ap.parse_args())

cap = cv2.VideoCapture(0)

if args["isVideoWriter"] == True:
    fourrcc = cv2.VideoWriter_fourcc("M","J","P","G")
    capWidth = int(cap.get(3))
    capHeight = int(cap.get(4))
    videoWrite = cv2.VideoWriter("output.avi", fourrcc, 22, (capWidth, capHeight))


def putText(frame, text, x, y):
    color = (193, 69, 42) # It is the color of text string to be drawn
    font = cv2.FONT_HERSHEY_SIMPLEX # It denotes the font type
    thickness = 2 # It is the thickness of the line in px
    fontScale = 0.75 # Font scale factor that is multiplied by the font-specific base size
    org = (x, y) # It is the coordinates of the bottom-left corner of the text string in the image
    lineType = cv2.LINE_AA # It gives the type of the line to be used
    cv2.putText(frame, text, org, font, fontScale , color, thickness, lineType)
    return frame


# pre-trained model
modelFile = "models/dnn/res10_300x300_ssd_iter_140000.caffemodel"
# prototxt has the information of where the training data is located.
configFile = "models/dnn/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


def detectFacesWithDNN(frame):
    # A neural network that really supports the input value
    size = (300,300)

    # After executing the average reduction, the image needs to be scaled
    scalefactor = 1.0

    # These are our mean subtraction values. They can be a 3-tuple of the RGB means or
    # they can be a single value in which case the supplied value is subtracted from every
    # channel of the image.
    swapRB = (104.0, 117.0, 123.0)

    height, width = frame.shape[:2]
    resizedFrame = cv2.resize(frame, size)
    blob = cv2.dnn.blobFromImage(resizedFrame, scalefactor, size, swapRB)
    net.setInput(blob)
    dnnFaces = net.forward()
    for i in range(dnnFaces.shape[2]):
        confidence = dnnFaces[0, 0, i, 2]
        if confidence > 0.5:
            box = dnnFaces[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(frame, (x, y), (x1, y1), (193, 69, 42), 2)
            frame = putText(frame, "Face Detection", x+5, y-5)
    return frame


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = detectFacesWithDNN(frame)

    if args["isVideoWriter"] == True:
        videoWrite.write(frame)

    cv2.imshow("Face Detection Model", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
if args["isVideoWriter"] == True:
    videoWrite.release()
cv2.destroyAllWindows()
