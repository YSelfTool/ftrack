#!/usr/bin/env python3

import contextlib
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
mp_face_mesh = mediapipe.solutions.face_mesh
mp_face_mesh_connections = mediapipe.solutions.face_mesh_connections
mp_drawing = mediapipe.solutions.drawing_utils
mp_background = mediapipe.solutions.selfie_segmentation
mp_pose = mediapipe.solutions.pose
from mediapipe.python.solutions import drawing_styles

def collect_landmark_indices(connections):
    indices = set()
    for first, second in connections:
        indices.add(first)
        indices.add(second)
    return sorted(indices)

LANDMARKS_LEFT_IRIS = collect_landmark_indices(mp_face_mesh_connections.FACEMESH_LEFT_IRIS)
LANDMARKS_RIGHT_IRIS = collect_landmark_indices(mp_face_mesh_connections.FACEMESH_RIGHT_IRIS)
LANDMARKS_FACE_OUTLINE = collect_landmark_indices(mp_face_mesh_connections.FACEMESH_FACE_OVAL)

def extract_landmarks_position_and_size(landmarks, indices, input_size):
    input_width, input_height = input_size
    x = []
    y = []
    z = []
    for index in indices:
        landmark = landmarks.landmark[index]
        x.append(landmark.x * input_width)
        y.append(landmark.y * input_height)
        z.append(landmark.z)
    min_x = np.min(x)
    max_x = np.max(x)
    center_x = np.mean(x)
    min_y = np.min(y)
    max_y = np.max(y)
    center_y = np.mean(y)
    min_z = np.min(z)
    max_z = np.max(z)
    center_z = np.mean(z)
    return (center_x, center_y, center_z), (max_x - min_x, max_y - min_y, max_z - min_z)

def extract_head_position(landmarks, input_size):
    left_iris_pos, left_iris_size = extract_landmarks_position_and_size(landmarks, LANDMARKS_LEFT_IRIS, input_size)
    right_iris_pos, right_iris_size = extract_landmarks_position_and_size(landmarks, LANDMARKS_RIGHT_IRIS, input_size)
    face_pos, face_size = extract_landmarks_position_and_size(landmarks, LANDMARKS_FACE_OUTLINE, input_size)
    iris_delta = abs(right_iris_pos[2] - left_iris_pos[2])
    input_width, input_height = input_size
    left_iris_ratio = left_iris_size[0] / left_iris_size[1]
    if left_iris_ratio > 1:
        left_iris_ratio = 1 / left_iris_ratio
    right_iris_ratio = right_iris_size[0] / right_iris_size[1]
    if right_iris_ratio > 1:
        right_iris_ratio = 1 / right_iris_ratio
    if left_iris_pos[2] < right_iris_pos[2]:
        iris_ratio = left_iris_ratio
    else:
        iris_ratio = right_iris_ratio
    return face_pos, face_size, iris_delta, iris_ratio


class DetectionRectangle:
    def __init__(self, input_size, output_size, smoothness_pos, smoothness_size, target_size_factor=1, digital_zoom=False):
        input_width, input_height = input_size
        output_width, output_height = output_size
        self.input_width = input_width
        self.input_height = input_height
        self.output_width = output_width
        self.output_height = output_height
        self.target_size_factor = target_size_factor
        self.digital_zoom = digital_zoom
        self.center_x = SmoothedValue(input_width / 2, smoothness_pos, 0, input_width)
        self.center_y = SmoothedValue(input_height / 2, smoothness_pos, 0, input_height)
        self.zoom_factor = SmoothedValue(1, smoothness_size, 0 if digital_zoom else 1, min(input_width / output_width, input_height / output_height))

    def update(self, pos, size):
        center_x, center_y, *_ = pos
        self.center_x.update(center_x)
        self.center_y.update(center_y)
        width, height, *_ = size
        width = width * self.target_size_factor
        height = height * self.target_size_factor
        zoom_factor = min(width / self.output_width, height / self.output_height)
        self.zoom_factor.update(zoom_factor)


    def reset(self, pos, size):
        center_x, center_y = pos
        width, height = size
        self.center_x.reset(center_x)
        self.center_y.reset(center_y)
        self.zoom_factor.reset(1)

    def crop_image(self, image):
        width = int(self.output_width * self.zoom_factor.value)
        height = int(self.output_height * self.zoom_factor.value)
        offset_x = min(max(int(self.center_x.value - width / 2), 0), self.input_width - width)
        offset_y = min(max(int(self.center_y.value - height / 2), 0), self.input_height - height)
        image = image[offset_y:offset_y + height,offset_x:offset_x + width] 
        return cv2.resize(image, (self.output_width, self.output_height))

    def __repr__(self):
        return f"<DetectionRect({self.center_x.value}, {self.center_y.value}, {self.zoom_factor.value})>"


class InputDevice:
    def __init__(self, device_name, capture, input_size, output_size, fps, face_detection, smoothness_pos, smoothness_size, target_size_factor, digital_zoom):
        self.device_name = device_name
        self.capture = capture
        self.input_size = input_size
        self.fps = fps
        self.face_detection = face_detection
        self.detection_rect = DetectionRectangle(input_size, output_size, smoothness_pos, smoothness_size, target_size_factor, digital_zoom)

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
    def create(device_name, max_resolution, output_size, fps, smoothness_pos, smoothness_size, target_size_factor, digital_zoom):
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
            face_detection = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            return InputDevice(device_name, capture, input_size, output_size, fps, face_detection, smoothness_pos, smoothness_size, target_size_factor, digital_zoom)
        raise ValueError(f"Cannot find resolution for input device {device_name}.")

    @contextlib.contextmanager
    def use(self):
        self.setup()
        with self.face_detection:
            yield
        self.release()

    def get_max_zoom_factor(self, output_width, output_height):
        input_width, input_height = self.input_size
        return min(input_width / output_width, input_height / output_height)

    def capture_image(self):
        success, image = self.capture.read()
        if not success:
            raise ValueError(f"Cannot read image from device {self.capture}")

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detection_results = self.face_detection.process(image)
        image.flags.writeable = True
        good_detection = False
        if detection_results.multi_face_landmarks is not None:
            good_detection = True
            face_landmarks = detection_results.multi_face_landmarks[0]
            face_pos, face_size, iris_delta, iris_size_ratio = extract_head_position(face_landmarks, self.input_size)
            self.detection_rect.update(face_pos, face_size)
            self.iris_delta = iris_delta
            self.iris_size_ratio = iris_size_ratio
            mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
            mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh_connections.FACEMESH_LEFT_EYE, landmark_drawing_spec=None, connection_drawing_spec=drawing_styles._FACEMESH_CONTOURS_CONNECTION_STYLE[mp_face_mesh_connections.FACEMESH_LEFT_EYE])
            mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh_connections.FACEMESH_RIGHT_EYE, landmark_drawing_spec=None, connection_drawing_spec=drawing_styles._FACEMESH_CONTOURS_CONNECTION_STYLE[mp_face_mesh_connections.FACEMESH_RIGHT_EYE])
            mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh_connections.FACEMESH_FACE_OVAL, landmark_drawing_spec=None, connection_drawing_spec=drawing_styles._FACEMESH_CONTOURS_CONNECTION_STYLE[mp_face_mesh_connections.FACEMESH_FACE_OVAL])
            mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh_connections.FACEMESH_LIPS, landmark_drawing_spec=None, connection_drawing_spec=drawing_styles._FACEMESH_CONTOURS_CONNECTION_STYLE[mp_face_mesh_connections.FACEMESH_LIPS])

        self.image = self.detection_rect.crop_image(image)
        self.good_detection = good_detection

    def __repr__(self):
        return f"InputDevice(device_name={self.device_name!r}, capture={self.capture!r}, input_size={self.input_size!r}, fps={self.fps!r})"

    def __str__(self):
        return self.device_name


class SmoothedValue:
    def __init__(self, value, smoothness, min_value, max_value):
        self.value = value
        self.smoothness = smoothness
        self.min_value = min_value
        self.max_value = max_value

    def clip(self, value):
        return min(max(value, self.min_value), self.max_value)

    def reset(self, value):
        self.value = self.clip(value)

    def update(self, new_value):
        self.value = self.clip(self.value * self.smoothness + new_value * (1 - self.smoothness))


def check_looking_at(input_device, iris_size_threshold=0.99):
    if not input_device.good_detection:
        return False
    if input_device.iris_size_ratio > iris_size_threshold:
        print(input_device, input_device.iris_size_ratio)
    return input_device.iris_size_ratio > iris_size_threshold
    return input_device.iris_delta < 0.02 #or input_device.iris_size_ratio > iris_size_threshold


class InputDevices:
    def __init__(self, input_devices):
        self.input_devices = input_devices

    @contextlib.contextmanager
    def use(self):
        with contextlib.ExitStack() as stack:
            for device in self.input_devices:
                stack.enter_context(device.use())
            yield

    def get_max_zoom_factor(self, output_width, output_height):
        return min(input_device.get_max_zoom_factor(output_width, output_height) for input_device in self.input_devices)

    def captures_are_opened(self):
        return all(input_device.capture.isOpened() for input_device in self.input_devices)

    def capture_images(self, end_condition):
        current_input = None
        previous_input = None
        frames_on_input = 0
        while not end_condition():
            frames_on_input += 1
            if current_input is not None:
                previous_input = current_input
            for input_device in self.input_devices:
                input_device.capture_image()
            if frames_on_input > 25 and (current_input is None or not check_looking_at(current_input)) and any(check_looking_at(input_device) for input_device in self.input_devices):
                for input_device in self.input_devices:
                    if check_looking_at(input_device):
                        print(f"Looking at {input_device}")
                        current_input = input_device
                        frames_on_input = 1
                        yield input_device.image
                        break
            elif current_input is not None and current_input.good_detection:
                yield current_input.image
            elif any(input_device.good_detection for input_device in self.input_devices):
                for input_device in self.input_devices:
                    if input_device.good_detection:
                        print(f"Switching to {input_device}")
                        current_input = input_device
                        frames_on_input = 1
                        yield current_input.image
                        break
            elif previous_input is not None:
                if current_input is not None:
                    print("Lost face")
                    current_input = None
                    for input_device in self.input_devices:
                        input_device.capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                        input_device.capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                yield previous_input.image
            else:
                print("Defaulting to first camera")
                yield self.input_devices[0].image
        

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
    parser.add_argument("--digital-zoom", action="store_true", help="Do not limit zoom to input video resolution.")


    args = parser.parse_args()

    input_devices = InputDevices([InputDevice.create(input_device_name, args.max_resolution, args.output_size, args.fps, args.smoothness, args.zoom_smoothness, args.zoom_target, args.digital_zoom) for input_device_name in args.inputs])

    output_width, output_height = args.output_size

    min_zoom_factor = 1
    max_zoom_factor = input_devices.get_max_zoom_factor(output_width, output_height)

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

            with input_devices.use():
                def _end_condition():
                    return not input_devices.captures_are_opened() or readers.value == 0
                for image in input_devices.capture_images(_end_condition):
                    camera.send(image)
                    camera.sleep_until_next_frame()

            camera.send(np.zeros((output_height, output_width, 3), dtype=np.uint8))

            print("No more readers, pausing output.")

        mp_value.value = -100
        inotify_process.join()

if __name__ == "__main__":
    main()
