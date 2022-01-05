# Zeeshan Khan
# Age and (Binary) Gender Detection Program
# 1/4/2022

import cv2, sys                                                                                         # cv2 is OpenCV lib, sys for argv

if len(sys.argv) == 1:                                                                                  # Help Message is argv[1] doesn't exist
    print("\nUsage: 'python ageDetect.py [path/to/image]\nUsage: 'python ageDetect.py camera'\n")       # Tells user that they can input file path or "camera" for Live capture
    sys.exit(0)                                                                                         # Exit Program

faceProto = "files/face_detector.pbtxt"
faceModel = "files/face_detector_uint8.pb"
ageProto = "files/age_deploy.prototxt"
ageModel = "files/age_net.caffemodel"
genderProto = "files/gender_deploy.prototxt"
genderModel = "files/gender_net.caffemodel"
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male','Female']
mean1 = (78.4263377603, 87.7689143744, 114.895847746)
mean2 = (104, 117, 123)

def findFace(faceNet, image):
    copy = image.copy()
    height = copy.shape[0]
    width = copy.shape[1]
    b = cv2.dnn.blobFromImage(copy, 1.0, (299, 299), mean2, True)
    faceNet.setInput(b)
    det = faceNet.forward()
    boxes = []
    for i in range(det.shape[2]):
        con = det[0, 0, i, 2]
        if con > 0.7:
            x1 = int(det[0, 0, i, 3] * width)
            y1 = int(det[0, 0, i, 4] * height)
            x2 = int(det[0, 0, i, 5] * width)
            y2 = int(det[0, 0, i, 6] * height)
            boxes.append([x1, y1, x2, y2])
            cv2.rectangle(copy, (x1, y1), (x2, y2), (0, 255, 0), int(round(height / 150)), 8)
    return copy, boxes

def Live():
    capture = cv2.VideoCapture(0)
    while cv2.waitKey(1) < 0:
        a, image = capture.read()        
        result, box = findFace(faceNet, image)
        for faces in box:
            face = image[max(0, faces[1]):min(faces[3], image.shape[0]), max(0, faces[0]):min(faces[2], image.shape[1])]
            b = cv2.dnn.blobFromImage(face, 1.0, (227, 227), mean1, swapRB = False)
            genderNet.setInput(b)
            ageNet.setInput(b)
            genderPreds=genderNet.forward()
            agePreds=ageNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            age=ageList[agePreds[0].argmax()]
            cv2.putText(result, f'{gender}, {age}', (faces[0], faces[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)
            cv2.imshow("Hit any button to exit", result)

def Photo():
    capture = cv2.VideoCapture(sys.argv[1])
    while cv2.waitKey(1) < 0:
        a, image = capture.read()
        cv2.waitKey()                                           # Waitkey to not close image
        result, box = findFace(faceNet, image)
        for faces in box:
            face = image[max(0, faces[1]):min(faces[3], image.shape[0]), max(0, faces[0]):min(faces[2], image.shape[1])]
            b = cv2.dnn.blobFromImage(face, 1.0, (227, 227), mean1, swapRB = False)
            genderNet.setInput(b)
            ageNet.setInput(b)
            genderPreds=genderNet.forward()
            agePreds=ageNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            age=ageList[agePreds[0].argmax()]
            cv2.putText(result, f'{gender}, {age}', (faces[0], faces[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)
            cv2.imshow("Hit any button to exit", result)

if sys.argv[1] == "camera":
    print("Opening Camera...")
    Live()
else:
    print("Opening Photo...")
    Photo()