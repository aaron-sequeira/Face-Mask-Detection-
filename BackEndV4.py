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


def calculate_center(bbox):
    x_center = int(bbox[0] + (bbox[2] - bbox[0]) / 2)
    y_center = int(bbox[1] + (bbox[3] - bbox[1]) / 2)
    return x_center, y_center


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
            detection1 = detections.xyxy[0]
            detection2 = detections.xyxy[1]

            x1_center, y1_center = calculate_center(detection1)
            x2_center, y2_center = calculate_center(detection2)

            frame = cv2.line(frame, (x1_center, y1_center), (x2_center, y2_center), (255, 0, 0), 2)

            distance = np.sqrt((x2_center - x1_center) ** 2 + (y2_center - y1_center) ** 2)
            distance_text = f'Distance: {distance:.2f} pixels'
            cv2.putText(frame, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Safe if distance >= 2 meters (assuming 1 pixel = 1 meter for illustration; adjust as necessary)
            if distance >= 400:  # Adjust this threshold based on actual scale
                safety_status = "Safe"
            elif int(detections.class_id[0]) == 1 and int(detections.class_id[0]) == 1:
                safety_status = "Safe"
            else:
                safety_status = "Not safe"

            cv2.putText(frame, safety_status, (frame_width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections)
        cv2.imshow("yolov8", frame)

        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
