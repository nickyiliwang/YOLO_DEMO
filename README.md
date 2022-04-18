# DEMO Real time object detection
Description:
This is the DEMO for You Only Look Once (YOLO) with the use of python-opencv

## Prerequisites: CLI setup
   1. MacOS: needs homebrew
   2. brew install python
   3. pip/pip3 install opencv-python

## Prerequisites: Files in resources folder
1. create an coco.names file from this [link](https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names)
2. From this [link](https://pjreddie.com/darknet/yolo/), search for: Performance on the COCO Dataset, and download the cfg and weights file for either:
   1. Slower (45 FPS): YOLOv3-320
   2. Fast (244 FPS): Tiny YOLO
3. Lastly have some images and videos that you would like to input for this code to work, ie. office.png, people.jpg, dog.mp4

## Commands
```
Change the input image file location in Image_Detection.py and then run:
python3 Image_Detection.py

Change the input video file location in Video_Detection.py and then run:
python3 Video_Detection.py
```

## Credits
https://www.youtube.com/watch?v=1LCb1PVqzeY&ab_channel=eMasterClassAcademy
https://pjreddie.com/darknet/yolo/
https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names