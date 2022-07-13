#!/usr/bin/env python3

import os
import time
from multiprocessing import Process, Value
import subprocess as sp

import numpy as np
import cv2
import mediapipe
import pyvirtualcam

from inotify.adapters import Inotify

mp_face_detection = mediapipe.solutions.face_detection
mp_drawing = mediapipe.solutions.drawing_utils
mp_background = mediapipe.solutions.selfie_segmentation
mp_pose = mediapipe.solutions.pose


class InputDevice:
    def __init__(self, device_name, capture, input_size, fps):
        self.device_name = device_name
        self.capture = capture
        self.input_size = input_size
        self.fps = fps

    def setup(self):
        input_width, input_height = self.input_size
        self.capture = cv2.VideoCapture(self.device_name)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, input_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, input_height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

    def release(self):
        self.capture.release()


    @staticmethod
    def create(device_name, max_resolution, fps=30):
        capture = cv2.VideoCapture(device_name)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        max_input_width, max_input_height = max_resolution
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, max_input_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, max_input_height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        success, frame = capture.read()
        capture.release()
        if success:
            input_size = (frame.shape[1], frame.shape[0])
            print(f"Using input size {input_size} for {device_name}")
            return InputDevice(device_name, capture, input_size, fps)
        raise ValueError(f"Cannot find resolution for input device {device_name}.")

    def get_max_zoom_factor(self, output_width, output_height):
        input_width, input_height = self.input_size
        return min(input_width / output_width, input_height / output_height)

    def capture_image(self, face_detection):
        success, image = self.capture.read()
        if not success:
            raise ValueError(f"Cannot read image from device {capture}")

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detection_results = face_detection.process(image)
        image.flags.writeable = True
        return image, detection_results

    def __repr__(self):
        return f"InputDevice(device_name={self.device_name!r}, capture={self.capture!r}, input_size={self.input_size!r}, fps={self.fps!r})"

    def __str__(self):
        return self.device_name



def watch_file(filename, mp_value):
    pgid = os.getpgid(os.getpid())
    lsof_output = sp.run(["lsof", "-F", "-g", f"^{pgid}", filename], text=True, stdout=sp.PIPE).stdout.splitlines()
    initial_watchers = len([line for line in lsof_output if line.startswith("p")])
    mp_value.value = initial_watchers
    inotify = Inotify()
    inotify.add_watch(filename)
    for event, flags, filename, _ in inotify.event_gen(yield_nones=False):
        if "IN_OPEN" in flags:
            mp_value.value += 1
        if "IN_CLOSE_WRITE" in flags or "IN_CLOSE_NOWRITE" in flags:
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
    parser.add_argument("--inputs", "-i", nargs="+", default=["/dev/video0"], help="Input device paths")
    parser.add_argument("--output", "-o", default="/dev/video2", help="Output device path")
    parser.add_argument("--output-size", type=int, nargs=2, default=(1280, 720), help="Resolution of output video")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for input and output.")
    parser.add_argument("--max-resolution", type=int, nargs=2, default=(2560, 1440), help="Maximum resolution to capture video at.")
    parser.add_argument("--smoothness", type=float, default=0.95, help="Tracking smoothing factor.")
    parser.add_argument("--zoom-smoothness", type=float, default=0.975, help="Tracking smoothing factor.")
    parser.add_argument("--zoom-target", type=float, default=2.0, help="Target value for how far to zoom in.")
    parser.add_argument("--digital-zoom", action="store_false", dest="limit_zoom", help="Do not limit zoom to input video resolution.")


    args = parser.parse_args()

    input_devices = [InputDevice.create(input_device_name, args.max_resolution, args.fps) for input_device_name in args.inputs]

    output_width, output_height = args.output_size

    min_zoom_factor = 1
    max_zoom_factor = min(input_device.get_max_zoom_factor(output_width, output_height) for input_device in input_devices)

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
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

                print("Detected readers, starting output.")

                center_x, center_y = None, None
                zoom_factor = min_zoom_factor

                for input_device in input_devices:
                    input_device.setup()
                current_input_device = None
                frames_on_current = 0

                while all(input_device.capture.isOpened() for input_device in input_devices) and readers.value > 0:
                    found = False
                    previous_input_device = current_input_device
                    image_from_current_input_device = None
                    if current_input_device is not None:
                        image, detection_results = current_input_device.capture_image(face_detection)
                        image_from_current_input_device = image
                        frames_on_current += 1
                        if detection_results.detections:
                            found = True
                    if (not found and frames_on_current >= args.fps) or current_input_device is None:
                        for input_device in input_devices:
                            if input_device != current_input_device:
                                image, detection_results = input_device.capture_image(face_detection)
                                if detection_results.detections and any(d.score[0] > 0.9 for d in detection_results.detections):
                                    print(f"Switching to {input_device}")
                                    found = True
                                    current_input_device = input_device
                                    input_device.capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                                    input_device.capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                                    center_x = None
                                    frames_on_current = 0
                                    max_zoom_factor = input_device.get_max_zoom_factor(output_width, output_height)
                                    break
                    if not found and image_from_current_input_device is not None:
                        image = image_from_current_input_device
                    input_height, input_width = image.shape[:2]
                    if found:
                        image.flags.writeable = False
                        result = detection_results.detections[0]
                        image.flags.writeable = True
                        result = detection_results.detections[0]
                        img_size = np.array([image.shape[1], image.shape[0]])
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
                            zoom_factor = min(args.zoom_smoothness * zoom_factor + (1 - args.zoom_smoothness) * new_zoom_factor, max_zoom_factor)
                            if args.limit_zoom:
                                zoom_factor = max(min(zoom_factor, max_zoom_factor), min_zoom_factor)
                        else:
                            center_x = new_center_x
                            center_y = new_center_y
                            zoom_factor = new_zoom_factor
                    else:
                        new_center_x = input_width / 2
                        new_center_y = input_height / 2
                        new_zoom_factor = max_zoom_factor
                        if center_x is not None:
                            center_x = 0.995 * center_x + (1 - 0.995) * new_center_x
                            center_y = 0.995 * center_y + (1 - 0.995) * new_center_y
                            zoom_factor = min(0.995 * zoom_factor + (1 - 0.995) * new_zoom_factor, max_zoom_factor)
                            if args.limit_zoom:
                                zoom_factor = max(min(zoom_factor, max_zoom_factor), min_zoom_factor)
                        else:
                            center_x = new_center_x
                            center_y = new_center_y
                            zoom_factor = new_zoom_factor

                    offset_x = min(max(int(center_x - output_width * zoom_factor / 2), 0), input_width - int(output_width * zoom_factor))
                    offset_y = min(max(int(center_y - output_height * zoom_factor / 2), 0), input_height - int(output_height * zoom_factor))
                    image = image[offset_y:int(offset_y + output_height * zoom_factor), offset_x:int(offset_x + output_width * zoom_factor)]
                    # zoom
                    image = cv2.resize(image, (output_width, output_height))

                    camera.send(image)
                    camera.sleep_until_next_frame()

                for input_device in input_devices:
                    input_device.release()
                camera.send(np.zeros((output_height, output_width, 3), dtype=np.uint8))

                print("No more readers, pausing output.")

            mp_value.value = -100
            inotify_process.join()

if __name__ == "__main__":
    main()
