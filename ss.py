import numpy as np
import time
import cv2
import pyttsx3

INPUT_FILE='demo.jpg'
OUTPUT_FILE='predicted.jpg'
LABELS_FILE='coco.names'
CONFIG_FILE='yolov3.cfg'
WEIGHTS_FILE='yolov3.weights'
CONFIDENCE_THRESHOLD=0.3

engine = pyttsx3.init()
LABELS = open(LABELS_FILE).read().strip().split("\n")

np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
dtype="uint8")

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loadin
        continue

    net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

    # image = cv2.imread(INPUT_FILE)
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i-1] for i in net.getUnconnectedOutLayers()]


    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()


    print("[INFO] YOLO took {:.6f} seconds".format(end - start))


    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs 
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

    # filter out weak predictions by ensuring the detected
    # probability is greater than the minimum probability
    if confidence > CONFIDENCE_THRESHOLD:
    # scale the bounding box coordinates back relative to the
    # size of the image, keeping in mind that YOLO actually
    # returns the center (x, y)-coordinates of the bounding
    # box followed by the boxes' width and height
        box = detection[0:4] * np.array([W, H, W, H])
    (centerX, centerY, width, height) = box.astype("int")

    # use the center (x, y)-coordinates to derive the top and
    # and left corner of the bounding box
    x = int(centerX - (width / 2))
    y = int(centerY - (height / 2))

    # update our list of bounding box coordinates, confidences,
    # and class IDs
    boxes.append([x, y, int(width), int(height)])
    confidences.append(float(confidence))
    classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
    CONFIDENCE_THRESHOLD)
    # image = cv2.flip(image,1)
    # ensure at least one detection exists
    if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():

    # extract the bounding box coordinates
    (x, y) = (boxes[i][0], boxes[i][1])
    (w, h) = (boxes[i][2], boxes[i][3])

    color = [int(c) for c in COLORS[classIDs[i]]]

    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
    0.5, color, 2)

if len(idxs) > 0:
for i in idxs.flatten():
if (boxes[i][0] + boxes[i][2])/2 <= 200:
pos = "right" 
elif (boxes[i][0] + boxes[i][2])/2 >= 300:
pos = "left"
else:
pos = "center"
print(pos)

speeck = LABELS[classIDs[i]]+" "+ pos
engine.say(speeck)
engine.runAndWait()

# show the output image
# image = cv2.flip(image,1)
cv2.imshow("example.png", image)
if cv2.waitKey(5) & 0xFF == 27:
break
cap.release()