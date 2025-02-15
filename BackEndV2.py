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
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args


def test(detections):
    for detection in detections:
        print(detection)


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    model_path = os.path.join('.', 'runs', 'detect', 'train3', 'weights', 'last.pt')  
    label_annotator = sv.LabelAnnotator()  
    model = YOLO(model_path)
    box_annotator = sv.BoxAnnotator(thickness=2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        if len(detections) >= 2:
            detection1 = detections.xyxy[0]  # assuming detections is an object with an xyxy attribute
            detection2 = detections.xyxy[1]  # same here

            x1_center = int(detection1[0] + (detection1[2] - detection1[0]) / 2)
            y1_center = int(detection1[1] + (detection1[3] - detection1[1]) / 2)
            x2_center = int(detection2[0] + (detection2[2] - detection2[0]) / 2)
            y2_center = int(detection2[1] + (detection2[3] - detection2[1]) / 2)

            frame = cv2.line(frame, (x1_center, y1_center), (x2_center, y2_center), (255, 0, 0), 2)

        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections)
        cv2.imshow("yolov8", frame)
        
        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
