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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="/dev/video0", help="Input device path")
    parser.add_argument("--output", "-o", default="/dev/video2", help="Output device path")
    parser.add_argument("--input-size", type=int, nargs=2, default=(1920, 1080), help="Resolution of input video")
    parser.add_argument("--output-size", type=int, nargs=2, default=(1280, 720), help="Resolution of output video")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for input and output.")
    parser.add_argument("--smoothness", type=float, default=0.95, help="Tracking smoothing factor.")
    parser.add_argument("--zoom-smoothness", type=float, default=0.95, help="Tracking smoothing factor.")
    parser.add_argument("--zoom-target", type=float, default=1.0, help="Target value for how far to zoom in.")
    parser.add_argument("--blur-background", action="store_true", help="Blur background")
    parser.add_argument("--pose", action="store_true", help="Pose detection")


    args = parser.parse_args()

    input_width, input_height = args.input_size
    output_width, output_height = args.output_size

    min_zoom_factor = 1
    max_zoom_factor = min(input_width / output_width, input_height / output_height)

    capture = cv2.VideoCapture(args.input)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, input_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, input_height)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    capture.set(cv2.CAP_PROP_FPS, args.fps)

    success, frame1 = capture.read()
    if not success:
        print(f"Failed to read frame from source: {args.input}")
        return
    capture.release()


    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection, mp_background.SelfieSegmentation() as change_bg, mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        with pyvirtualcam.Camera(width=output_width, height=output_height, fps=args.fps, device=args.output) as camera:
            print(f"Using virtual camera: {camera.device}")

            camera.send(np.zeros((output_height, output_width, 3), dtype=np.uint8))

            readers = Value('i', 0)

            inotify_process = Process(target=watch_file, args=(args.output, readers))
            inotify_process.start()

            while True:
                while readers.value == 0:
                    time.sleep(1)

                capture = cv2.VideoCapture(args.input)
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, input_width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, input_height)
                capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                capture.set(cv2.CAP_PROP_FPS, args.fps)

                center_x, center_y = input_width / 2, input_height / 2
                zoom_factor = (min_zoom_factor + max_zoom_factor) / 2

                while capture.isOpened() and readers.value > 0:
                    success, image = capture.read()
                    if not success:
                        break

                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    detection_results = face_detection.process(image)
                    image.flags.writeable = True

                    if detection_results.detections:
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

                        center_x = args.smoothness * center_x + (1 - args.smoothness) * new_center_x
                        center_y = args.smoothness * center_y + (1 - args.smoothness) * new_center_y
                        zoom_factor = max(min(args.zoom_smoothness * zoom_factor + (1 - args.zoom_smoothness) * new_zoom_factor, max_zoom_factor), min_zoom_factor)

                    if args.pose:
                        image.flags.writeable = False
                        pose_results = pose.process(image)
                        image.flags.writeable = True
                        mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mediapipe.solutions.drawing_styles.get_default_pose_landmarks_style())

                    if args.blur_background:
                        image.flags.writeable = False
                        result = change_bg.process(image)
                        image.flags.writeable = True
                        fgmask = np.expand_dims(result.segmentation_mask, 2)
                        inverted = cv2.GaussianBlur(image, (25, 25), 0)
                        image = (image * fgmask + inverted * (1 - fgmask)).astype(np.uint8)


                    offset_x = min(max(int(center_x - output_width * zoom_factor / 2), 0), input_width - int(output_width * zoom_factor))
                    offset_y = min(max(int(center_y - output_height * zoom_factor / 2), 0), input_height - int(output_height * zoom_factor))
                    image = image[offset_y:int(offset_y + output_height * zoom_factor), offset_x:int(offset_x + output_width * zoom_factor)]
                    # zoom
                    image = cv2.resize(image, (output_width, output_height))

                    camera.send(image)
                    camera.sleep_until_next_frame()

                capture.release()
                camera.send(np.zeros((output_height, output_width, 3), dtype=np.uint8))

            mp_value.value = -100
            inotify_process.join()

if __name__ == "__main__":
    main()
