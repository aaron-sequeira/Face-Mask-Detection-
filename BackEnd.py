import os
import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280,720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def test(detections):
    test = np.array([])
    for test in detections:
        print(test)


def main():
    args = parse_arguments()
    frame_width , frame_height = args.webcam_resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    model_path = os.path.join('.', 'runs', 'detect', 'train3', 'weights', 'last.pt')  
    label_annotator = sv.LabelAnnotator()  
    model = YOLO(model_path)
    box_annotator = sv.BoxAnnotator(
        thickness=2,
    )
    ClassData = []

    while True:
        ret, frame = cap.read()
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        ClassData = []
        for ClassData in detections:
            ClassIndex = ClassData[3]
        if int(ClassIndex) == 1 or int(ClassIndex) == 2: 
            test(detections)
            AxisData = np.array([])
            for AxisData in detections:
                XaxisIndex = AxisData[0][0]
                YaxisIndex = AxisData[0][1]
                WidthIndex = AxisData[0][0]
                HeightIndex = AxisData[0][1]
            frame = cv2.line(frame,(int(XaxisIndex),int(YaxisIndex)),(int(WidthIndex),int(HeightIndex)), (255,0,0),10)
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(
            scene=frame,
            detections= detections
        )
        cv2.imshow("yolov8", frame)
        if (cv2.waitKey(30) == 27):
            break
    

if __name__ == "__main__":
    main()