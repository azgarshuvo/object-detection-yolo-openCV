import cv2
import numpy as np

webcam = False
# read the yolo configuration file, its a deep-neural-network formula
net = cv2.dnn.readNet('yolo-config/yolov3.weights', 'yolo-config/yolov3.cfg') 

classes = []
with open("data/coco.names", "r") as f: # read the coco dataset
    classes = f.read().splitlines() 

if webcam:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture("file/cam3_004.mp4")

while True:
    # capture the video frame
    _, frame = cap.read()

    # capture the height and weight of every frame that we are going to use it scale back to the original image size
    height, width, _ = frame.shape

    # creating a blob input (image, scaling, size of the image)
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)

    """
    # to see the R G B channels of the file
    for b in blob:
        for n, img_blob in enumerate(b):
            cv2.imshow(str(n), img_blob)
    """
    # passing the blob into input function
    net.setInput(blob)

    # getting the output layers name
    output_layers_names = net.getUnconnectedOutLayersNames()

    # getting the output layer
    layerOutputs = net.forward(output_layers_names)  

    boxes = []
    confidences = []
    class_ids = [] # represent the predicted classes

    for output in layerOutputs: # extract the information from each of the input
        for detection in output: # extract the information from each of the output
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
                # first 4 coeffcient is the location of the bounding box and the 5th element is the box confidence 
    
    # an object has multiple boxes, NMSBoxes method get the highest score of the boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # print(indexes.flatten())

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    # passing all the information to show over the video/pictures
    if len(indexes)>0:
        for i in indexes.flatten():
            # extracting the box coordinates
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            cv2.putText(frame, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

    cv2.imshow('Image', frame)

    # print the object coefficient
    print(boxes)

    key = cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()