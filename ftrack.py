#!/usr/bin/env python3

import time
from multiprocessing import Process, Value

import numpy as np
import cv2
import mediapipe
import pyvirtualcam

from inotify.adapters import Inotify

mp_face_detection = mediapipe.solutions.face_detection
mp_drawing = mediapipe.solutions.drawing_utils
mp_background = mediapipe.solutions.selfie_segmentation
mp_pose = mediapipe.solutions.pose


def watch_file(filename, mp_value):
    inotify = Inotify()
    inotify.add_watch(filename)
    for event, flags, filename, _ in inotify.event_gen(yield_nones=False):
        if "IN_OPEN" in flags:
            mp_value.value += 1
        if "IN_CLOSE_WRITE" in flags:
            mp_value.value -= 1
        if mp_value.value < -10:
            break

def point_to_arr(point):
    return np.array([point.x, point.y])


def draw_point(x, y, image, color, size):
    height, width, _ = image.shape
    min_x = np.maximum(x - size, 0)
    max_x = np.minimum(x + size, width)
    min_y = np.maximum(y - size, 0)
    max_y = np.minimum(y + size, height)
    image[min_y:max_y,min_x:max_x] = color


def capture_image(capture, face_detection):
    success, image = capture.read()
    if not success:
        raise ValueError(f"Cannot read image from device {capture}")

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detection_results = face_detection.process(image)
    image.flags.writeable = True
    return image, detection_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", "-i", nargs="+", default=["/dev/video0"], help="Input device paths")
    parser.add_argument("--output", "-o", default="/dev/video2", help="Output device path")
    parser.add_argument("--input-size", type=int, nargs=2, default=(1920, 1080), help="Resolution of input video")
    parser.add_argument("--output-size", type=int, nargs=2, default=(1280, 720), help="Resolution of output video")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for input and output.")
    parser.add_argument("--smoothness", type=float, default=0.95, help="Tracking smoothing factor.")
    parser.add_argument("--zoom-smoothness", type=float, default=0.975, help="Tracking smoothing factor.")
    parser.add_argument("--zoom-target", type=float, default=2.0, help="Target value for how far to zoom in.")
    parser.add_argument("--blur-background", action="store_true", help="Blur background")
    parser.add_argument("--pose", action="store_true", help="Pose detection")


    args = parser.parse_args()

    input_width, input_height = args.input_size
    output_width, output_height = args.output_size

    min_zoom_factor = 1
    max_zoom_factor = min(input_width / output_width, input_height / output_height)

    for input_device in args.inputs:
        print(f"Testing input device {input_device}")
        capture = cv2.VideoCapture(input_device)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, input_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, input_height)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        capture.set(cv2.CAP_PROP_FPS, args.fps)

        success, frame1 = capture.read()
        if not success:
            print(f"Failed to read frame from source: {input_device}")
            return
        capture.release()
        print("Test successful")


    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection, mp_background.SelfieSegmentation() as change_bg, mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        print("Detection set up.")
        with pyvirtualcam.Camera(width=output_width, height=output_height, fps=args.fps, device=args.output) as camera:
            print(f"Using virtual camera: {camera.device}")

            camera.send(np.zeros((output_height, output_width, 3), dtype=np.uint8))

            readers = Value('i', 0)

            inotify_process = Process(target=watch_file, args=(args.output, readers))
            inotify_process.start()

            while True:
                while readers.value == 0:
                    time.sleep(1)

                captures = []
                for input_device in args.inputs:
                    capture = cv2.VideoCapture(input_device)
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, input_width)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, input_height)
                    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    capture.set(cv2.CAP_PROP_FPS, args.fps)
                    captures.append(capture)

                center_x, center_y = input_width / 2, input_height / 2
                zoom_factor = (min_zoom_factor + max_zoom_factor) / 2

                current_capture = None
                frames_on_current = 0

                while all(capture.isOpened() for capture in captures) and readers.value > 0:
                    found = False
                    old_capture = current_capture
                    old_image = None
                    if current_capture is not None:
                        image, detection_results = capture_image(current_capture, face_detection)
                        old_image = image
                        frames_on_current += 1
                        if detection_results.detections:
                            found = True
                    if (not found and frames_on_current >= args.fps) or current_capture is None:
                        for capture in captures:
                            if capture != current_capture:
                                image, detection_results = capture_image(capture, face_detection)
                                if detection_results.detections and any(d.score[0] > 0.9 for d in detection_results.detections):
                                    print(f"Switching to {capture}")
                                    found = True
                                    current_capture = capture
                                    capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                                    capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                                    center_x = None
                                    frames_on_current = 0
                                    break
                    if not found and old_image is not None:
                        image = old_image
                    if found:
                        image.flags.writeable = False
                        result = detection_results.detections[0]
                        image.flags.writeable = True
                        result = detection_results.detections[0]
                        img_size = np.array([input_width, input_height])
                        nose_tip = point_to_arr(mp_face_detection.get_key_point(result, mp_face_detection.FaceKeyPoint.NOSE_TIP)) * img_size
                        left_eye = point_to_arr(mp_face_detection.get_key_point(result, mp_face_detection.FaceKeyPoint.LEFT_EYE)) * img_size
                        right_eye = point_to_arr(mp_face_detection.get_key_point(result, mp_face_detection.FaceKeyPoint.RIGHT_EYE)) * img_size
                        mouth_center = point_to_arr(mp_face_detection.get_key_point(result, mp_face_detection.FaceKeyPoint.MOUTH_CENTER)) * img_size
                        left_ear = point_to_arr(mp_face_detection.get_key_point(result, mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION)) * img_size
                        right_ear = point_to_arr(mp_face_detection.get_key_point(result, mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION)) * img_size
                        ear_center = (left_ear + right_ear) / 2
                        head_center = (ear_center + nose_tip) / 2
                        new_center_x = head_center[0]
                        new_center_y = head_center[1]

                        bbox = result.location_data.relative_bounding_box
                        bbox_width = bbox.width
                        bbox_height = bbox.height
                        new_zoom_factor = args.zoom_target * max(bbox_width * input_width, bbox_height * input_height) / min(output_width, output_height)

                        if center_x is not None:
                            center_x = args.smoothness * center_x + (1 - args.smoothness) * new_center_x
                            center_y = args.smoothness * center_y + (1 - args.smoothness) * new_center_y
                            zoom_factor = max(min(args.zoom_smoothness * zoom_factor + (1 - args.zoom_smoothness) * new_zoom_factor, max_zoom_factor), min_zoom_factor)
                        else:
                            center_x = new_center_x
                            center_y = new_center_y
                            zoom_factor = new_zoom_factor
                    else:
                        new_center_x = input_width / 2
                        new_center_y = input_height / 2
                        new_zoom_factor = max_zoom_factor
                        center_x = 0.995 * center_x + (1 - 0.995) * new_center_x
                        center_y = 0.995 * center_y + (1 - 0.995) * new_center_y
                        zoom_factor = max(min(0.995 * zoom_factor + (1 - 0.995) * new_zoom_factor, max_zoom_factor), min_zoom_factor)

                    offset_x = min(max(int(center_x - output_width * zoom_factor / 2), 0), input_width - int(output_width * zoom_factor))
                    offset_y = min(max(int(center_y - output_height * zoom_factor / 2), 0), input_height - int(output_height * zoom_factor))
                    image = image[offset_y:int(offset_y + output_height * zoom_factor), offset_x:int(offset_x + output_width * zoom_factor)]
                    # zoom
                    image = cv2.resize(image, (output_width, output_height))

                    camera.send(image)
                    camera.sleep_until_next_frame()

                for capture in captures:
                    capture.release()
                camera.send(np.zeros((output_height, output_width, 3), dtype=np.uint8))

            mp_value.value = -100
            inotify_process.join()

if __name__ == "__main__":
    main()
