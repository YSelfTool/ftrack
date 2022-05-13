#!/usr/bin/env python3

import numpy as np
import cv2
import mediapipe
import pyvirtualcam

mp_face_detection = mediapipe.solutions.face_detection
mp_drawing = mediapipe.solutions.drawing_utils


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="/dev/video0", help="Input device path")
    parser.add_argument("--output", "-o", default="/dev/video2", help="Output device path")
    parser.add_argument("--input-size", type=int, nargs=2, default=(1920, 1080), help="Resolution of input video")
    parser.add_argument("--output-size", type=int, nargs=2, default=(1280, 720), help="Resolution of output video")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for input and output.")
    parser.add_argument("--smoothness", type=float, default=0.9, help="Tracking smoothing factor.")


    args = parser.parse_args()

    capture = cv2.VideoCapture(args.input)

    input_width, input_height = args.input_size
    output_width, output_height = args.output_size

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, input_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, input_height)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    capture.set(cv2.CAP_PROP_FPS, args.fps)

    success, frame1 = capture.read()
    if not success:
        print(f"Failed to read frame from source: {args.input}")
        return

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        with pyvirtualcam.Camera(width=output_width, height=output_height, fps=args.fps, device=args.output) as camera:
            print(f"Using virtual camera: {camera.device}")

            center_x, center_y = input_width / 2, input_height / 2

            while capture.isOpened():
                success, image = capture.read()
                if not success:
                    break

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                detection_results = face_detection.process(image)

                image.flags.writeable = True

                if detection_results.detections:
                    result = detection_results.detections[0]
                    nose_tip = mp_face_detection.get_key_point(result, mp_face_detection.FaceKeyPoint.NOSE_TIP)
                    new_center_x = nose_tip.x * input_width
                    new_center_y = nose_tip.y * input_height
                    center_x = args.smoothness * center_x + (1 - args.smoothness) * new_center_x
                    center_y = args.smoothness * center_y + (1 - args.smoothness) * new_center_y

                offset_x = min(max(int(center_x - output_width / 2), 0), input_width - output_width)
                offset_y = min(max(int(center_y - output_height / 2), 0), input_height - output_height)
                image = image[offset_y:offset_y + output_height, offset_x:offset_x + output_width]

                camera.send(image)
                camera.sleep_until_next_frame()

if __name__ == "__main__":
    main()
