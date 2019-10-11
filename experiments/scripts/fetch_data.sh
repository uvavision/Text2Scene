#!/bin/bash

ABSTRACT=https://vision.ece.vt.edu/clipart/dataset/AbstractScenes_v1.1.zip
COCO_TRAIN=http://images.cocodataset.org/zips/train2017.zip
COCO_VAL=http://images.cocodataset.org/zips/val2017.zip
COCO_ANNOTATION_17=http://images.cocodataset.org/annotations/annotations_trainval2017.zip
COCO_ANNOTATION_14=http://images.cocodataset.org/annotations/annotations_trainval2014.zip
COCO_STUFF_17=http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

echo "Downloading Abstract Scene ..."
wget $ABSTRACT -O AbstractScenes_v1.1.zip
echo "Unzipping..."
unzip -q AbstractScenes_v1.1.zip -d data/

mkdir data/coco
mkdir data/coco/images

echo "Downloading coco images ..."
wget $COCO_TRAIN -O train2017.zip
wget $COCO_VAL -O val2017.zip
echo "Unzipping..."
unzip -q train2017.zip -d data/coco/images
unzip -q val2017.zip -d data/coco/images

echo "Downloading coco annotation data ..."
wget $COCO_ANNOTATION_17 -O annotations_trainval2017.zip
wget $COCO_ANNOTATION_14 -O annotations_trainval2014.zip
wget $COCO_STUFF_17 -O stuff_annotations_trainval2017.zip
echo "Unzipping..."
unzip -q annotations_trainval2017.zip -d data/coco
unzip -q annotations_trainval2014.zip -d data/coco
unzip -q stuff_annotations_trainval2017.zip -d data/coco

rm AbstractScenes_v1.1.zip train2017.zip val2017.zip annotations_trainval2017.zip annotations_trainval2014.zip stuff_annotations_trainval2017.zip
