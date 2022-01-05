# Zeeshan Khan
# Age and (Binary) Gender Detection Program
# 1/4/2022

import cv2, sys                                                                                         # cv2 is OpenCV lib, sys for argv

if len(sys.argv) == 1:                                                                                  # Help Message is argv[1] doesn't exist
    print("\nUsage: 'python ageDetect.py [path/to/image]\nUsage: 'python ageDetect.py camera'\n")       # Tells user that they can input file path or "camera" for Live capture
    sys.exit(0)                                                                                         # Exit Program

# OpenCV Models - https://talhassner.github.io/home/projects/Adience/Adience-data.html
faceProto = "files/face_detector.pbtxt"                                                                 # face detection prototxt file
faceModel = "files/face_detector_uint8.pb"                                                              # face detection model file
ageProto = "files/age_deploy.prototxt"                                                                  # age detection prototxt file
ageModel = "files/age_net.caffemodel"                                                                   # age detection model file
genderProto = "files/gender_deploy.prototxt"                                                            # gender detection prototxt file
genderModel = "files/gender_net.caffemodel"                                                             # gender detection model file
faceNet = cv2.dnn.readNet(faceModel, faceProto)                                                         # DNN with face model and proto
ageNet = cv2.dnn.readNet(ageModel, ageProto)                                                            # DNN with age model and proto
genderNet = cv2.dnn.readNet(genderModel, genderProto)                                                   # DNN with gender model and proto
ageList = ['(0-2)', '(3-8)', '(8-14)', '(15-22)', '(23-32)', '(33-42)', '(43-52)', '(60-100)']          # List of attributable ages
genderList = ['Male','Female']                                                                          # The (binary) genders
mean1 = (78.4263377603, 87.7689143744, 114.895847746)                                                   # Mean values for blobs
mean2 = (104, 117, 123)

# Function to find faces using blob (Binary Long Object Classification)
# Inspired from https://github.com/smahesh29/Gender-and-Age-Detection/blob/master/detect.py
def findFace(faceNet, image):
    copy = image.copy()                                                                                 # Copies image
    height = copy.shape[0]
    width = copy.shape[1]
    b = cv2.dnn.blobFromImage(copy, 1.0, (299, 299), mean2, True)                                       # Makes a blob
    faceNet.setInput(b)                                                                                 # Sets input to blob
    det = faceNet.forward()
    boxes = []
    for i in range(det.shape[2]):                                                                       # Calculate box shape around face
        con = det[0, 0, i, 2]
        if con > 0.7:
            x1 = int(det[0, 0, i, 3] * width)
            y1 = int(det[0, 0, i, 4] * height)
            x2 = int(det[0, 0, i, 5] * width)
            y2 = int(det[0, 0, i, 6] * height)
            boxes.append([x1, y1, x2, y2])                                                              # Adds box
            cv2.rectangle(copy, (x1, y1), (x2, y2), (255, 0, 255), int(round(height / 150)), 8)         # Adds box to image
    return copy, boxes                                                                                  # Returns values

# Function for age and gender detection via camera
def Live():
    capture = cv2.VideoCapture(0)                                                                       # Turns on camera for video capture
    while cv2.waitKey(1) < 0:                                                                           # While no button clicked
        a, image = capture.read()                                                                       # Capture input
        result, box = findFace(faceNet, image)                                                          # Sends faceNet DNN and current frame to findFace fn
        for faces in box:
            face = image[max(0, faces[1]) : min(faces[3], image.shape[0]), max(0, faces[0]) : min(faces[2], image.shape[1])]
            b = cv2.dnn.blobFromImage(face, 1.0, (227, 227), mean1, swapRB = False)                     # Creates blob for gender and age
            genderNet.setInput(b)                                                                       # Sets inputs for gender and age DNN to blob
            ageNet.setInput(b)
            Pgender = genderNet.forward()                                                               # forwards
            Page = ageNet.forward()
            gender = genderList[Pgender[0].argmax()]                                                    # Determines gender
            age = ageList[Page[0].argmax()]                                                             # Determines age
            cv2.putText(result, f'{gender}, {age}', (faces[0], faces[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)
            cv2.imshow("Hit any button to exit", result)                                                # Displays image

# Function for age and gender detection via inputted image
def Photo():
    capture = cv2.VideoCapture(sys.argv[1])                                                             # Passes file location into capture
    while cv2.waitKey(1) < 0:                                                                           # While no button clicked
        a, image = capture.read()                                                                       # Capture input
        if not a:
            cv2.waitKey()                                                                               # Waitkey to not close image
            break                                                                                       # Exit loop
        result, box = findFace(faceNet, image)                                                          # Sends faceNet DNN and current frame to findFace fn
        for faces in box:
            face = image[max(0, faces[1]) : min(faces[3], image.shape[0]), max(0, faces[0]) : min(faces[2], image.shape[1])]
            b = cv2.dnn.blobFromImage(face, 1.0, (227, 227), mean1, swapRB = False)                     # Creates blob for gender and age
            genderNet.setInput(b)                                                                       # Sets inputs for gender and age DNN to blob
            ageNet.setInput(b)
            Pgender = genderNet.forward()                                                               # forwarding
            Page = ageNet.forward()
            gender = genderList[Pgender[0].argmax()]                                                    # Determines gender
            age = ageList[Page[0].argmax()]                                                             # Determines age
            cv2.putText(result, f'{gender}, {age}', (faces[0], faces[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)
            cv2.imshow("Hit any button to exit", result)                                                # Display image

# Main Function
if sys.argv[1] == "camera":                                                                             # If "camera" after "python ageDetect.py"
    print("Opening Camera...")
    Live()                                                                                              # Go to Live() function
else:                                                                                                   # Else go to Photo() function
    print("Opening Photo...")
    Photo()