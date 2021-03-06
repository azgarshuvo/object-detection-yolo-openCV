# object-detection-yolo-openCV
Object Detection using YOLO algorithm classified by Deep Neural Network, Library using openCV by Python

## Installation

```
pip install numpy
pip install opencv-python
```

## Download YOLO config and weights file

Given the below link you will find several version of YOLO. Select a version and download the weights as well as the cfg file and place it into the **yolo-config** directory. 
* For faster processing you may download the YOLOv3-tiny 
* For better accuracy you may download other version-3 configuration file
* Here I have used YOLOv3-320
* Refer to: <https://pjreddie.com/darknet/yolo/>


## COCO Dataset
The COCO dataset is used to label the object after detection. The dataset is already in **data** folder.
If you wish you can download the dataset from the following link <https://github.com/pjreddie/darknet/blob/master/data/coco.names/>


## Input File
Using YOLO you can Detect the object from an Image or from a live video or using your webcam. Place your input file into the **file** folder
