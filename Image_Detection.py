import cv2
import numpy as np

net = cv2.dnn.readNet(
    'resources/normal/yolov3.weights', 'resources/normal/yolov3.cfg')
classes = []
with open('resources/normal/coco.names', 'r') as f:
    classes = f.read().splitlines()

img = cv2.imread('resources/image.png')
height, width, _ = img.shape

# prepare the image blob

# normalizaing and channel swapping
# end result is black and white squares
# This is the preferred format for the deep learning algo
blob = cv2.dnn.blobFromImage(
    img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

# passing the input blob into our network
net.setInput(blob)

# OUTPUTS
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

# results are stored in lists
# bounding box, confidences, prediction class
boxes = []
confidences = []
class_ids = []

# nested forloops for layers outputs, second forloop extracts info in each output
for output in layerOutputs:
    for detection in output:
        # get all predictions starting from the 6th element, first 5 are locations and such, unrelated
        scores = detection[5:]
        # get the location of the highest score
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        # valid confidence threshold
        if confidence > 0.5:
            # plotting the bounding box
            # to scale the output back, we are using our image height and width
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            # subtrack from output, which needs offset from the center
            x = int(center_x - w/2)
            y = int(center_y - h/2)

            # appending results
            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

results = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# picking the font and draw boxes colors
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(boxes), 3))

# checking for null values
if len(results) > 0:
    # identify each object
    for i in results.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        # create a rectangle
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        # put output texts back onto the image, upper left
        cv2.putText(img, label + " " + confidence,
                    (x, y+20), font, 2, (255, 255, 255), 2)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
