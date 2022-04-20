# cv2 (opencv) is a library of programming functions mainly aimed at real-time computer vision. 
import cv2
# NumPy is a large collection of high-level mathematical functions to operate on these arrays
import numpy as np

# importing the pre-trained weights file and config file
net = cv2.dnn.readNet(
    'resources/yolov2-tiny.weights', 'resources/yolov3.cfg')
# classes variable will contain identified object names (ie. bike, car, bottle, person)
classes = []
# reading the predefined names from coco file and assigning it to the classes variable
with open('resources/coco.names', 'r') as f:
    classes = f.read().splitlines()

# input file location, input images/videos that you want identified
img = cv2.imread('resources/inputs/rock.jpg') 
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
# Returns indexes of layers with unconnected outputs
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

# results
boxes = [] # bounding box to draw onto the image
confidences = [] # how confident is the prediction
class_ids = [] # labeling of the prediction object

# nested forloops for layers outputs, second forloop extracts info in each output
for output in layerOutputs:
    for detection in output:
        # get all predictions starting from the 6th element, first 5 are locations and such, unrelated data
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

# checking for null values, or empty results
if len(results) > 0:
    # identify each object
    # flatten(): [[5, 6], [7, 8]] => [5 6 7 8]
    for i in results.flatten():
        x, y, w, h = boxes[i]
        # stringify the results
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        # draw the rectangle
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        # put output texts back onto the image, upper left
        cv2.putText(img, label + " " + confidence,
                    (x, y+20), font, 2, (255, 255, 255), 2)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
