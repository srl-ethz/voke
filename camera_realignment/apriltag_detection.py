## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

#############################################################
##              Open CV and Numpy integration              ##
#############################################################

import pyrealsense2 as rs
import math
import numpy as np
import cv2

import yaml
from dt_apriltags import Detector

import argparse


parser = argparse.ArgumentParser("apriltag")

parser.add_argument(
    "--data", type=str, default="./export/", help="location of the camera data"
)
parser.add_argument("--camera_id", type=int, default=0, help="id of the camera")
parser.add_argument(
    "--record",
    action="store_true",
    default=False,
    help="if true, record current camera status",
)

args = parser.parse_args()


def is_rotation_matrix(R):
    Rt = np.transpose(R)
    RtR = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - RtR)
    return n < 1e-6


def rotation_to_euler(R):
    assert is_rotation_matrix(R)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def check_tag_id(old_tag_info, tags):
    tag_id = None
    for i in range(len(tags)):
        temp_tag_id = tags[i].tag_id
        if temp_tag_id in old_tag_info:
            tag_id = temp_tag_id
            break
    return tag_id


def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == "RGB Camera":
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    at_detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    with open("./test_info.yaml", "r") as stream:
        parameters = yaml.load(stream, Loader=yaml.Loader)

    cameraMatrix = np.array(parameters["K"]).reshape((3, 3))
    camera_params = (
        cameraMatrix[0, 0],
        cameraMatrix[1, 1],
        cameraMatrix[0, 2],
        cameraMatrix[1, 2],
    )

    # Start streaming
    pipeline.start(config)

    record = args.record
    reposition = not record

    export_path = args.data + "camera" + str(args.camera_id)

    if reposition:
        ref_image = cv2.imread(export_path + ".png")
        with open(export_path + ".yaml", "r") as file:
            old_tags = yaml.load(file, Loader=yaml.Loader)
        if len(old_tags) == 0:
            print("There is no previous camera position information.")
            exit(0)
        old_tag_info = {}
        for i in range(len(old_tags)):
            old_tag_info[old_tags[i].tag_id] = {
                "pose_t": old_tags[i].pose_t,
                "pose_R": old_tags[i].pose_R,
                "pose_angle": rotation_to_euler(old_tags[i].pose_R),
            }

    count = 0
    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            count += 1
            if count % 20 == 0:
                img = np.dot(color_image, np.array([0.2989, 0.5870, 0.1140]))
                img = img.astype(np.float64) / np.max(
                    img
                )  # normalize the data to 0 - 1
                img = 255 * img  # Now scale by 255
                img = img.astype(np.uint8)
                tags = at_detector.detect(
                    img, True, camera_params, parameters["tag_size"]
                )
                if len(tags) > 0:
                    if record:
                        cv2.imwrite(export_path + ".png", color_image)
                        with open(export_path + ".yaml", "w") as file:
                            documents = yaml.dump(tags, file)
                        # record = False

                    if reposition:
                        tag_id = check_tag_id(old_tag_info, tags)
                        if tag_id is not None:
                            pose_t_diff = (
                                tags[0].pose_t - old_tag_info[tag_id]["pose_t"]
                            )
                            pose_x_diff = pose_t_diff[0][0]
                            pose_y_diff = pose_t_diff[1][0]
                            pose_z_diff = pose_t_diff[2][0]

                            pose_angle_diff = (
                                rotation_to_euler(tags[0].pose_R)
                                - old_tag_info[tag_id]["pose_angle"]
                            )
                            pose_alpha_diff = pose_angle_diff[0]
                            pose_beta_diff = pose_angle_diff[1]
                            pose_gamma_diff = pose_angle_diff[2]

                            if pose_x_diff < 0:
                                print("Move LEFT  for " + str(abs(pose_x_diff)))
                            if pose_x_diff > 0:
                                print("Move RIGHT for " + str(abs(pose_x_diff)))
                            if pose_y_diff < 0:
                                print("Move UP    for " + str(abs(pose_y_diff)))
                            if pose_y_diff > 0:
                                print("Move DOWN  for " + str(abs(pose_y_diff)))
                            if pose_z_diff < 0:
                                print("Move BACK  for " + str(abs(pose_z_diff)))
                            if pose_z_diff > 0:
                                print("Move FRONT for " + str(abs(pose_z_diff)))

                            print("")

                            if pose_alpha_diff < 0:
                                print("Rotate FRONT for " + str(abs(pose_alpha_diff)))
                            if pose_alpha_diff > 0:
                                print("Rotate BACK  for " + str(abs(pose_alpha_diff)))
                            if pose_beta_diff < 0:
                                print("Rotate ANTIC for " + str(abs(pose_beta_diff)))
                            if pose_beta_diff > 0:
                                print("Rotate CLOCK for " + str(abs(pose_beta_diff)))
                            if pose_gamma_diff < 0:
                                print("Rotate LEFT  for " + str(abs(pose_gamma_diff)))
                            if pose_gamma_diff > 0:
                                print("Rotate RIGHT for " + str(abs(pose_gamma_diff)))

                            print("")

                            # print(tags[0].pose_R - old_tag_info[tag_id]["pose_R"])

            if reposition:
                color_image = cv2.addWeighted(color_image, 1, ref_image, 0.5, 0)

            # Show images
            cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("RealSense", color_image)
            cv2.waitKey(1)

    finally:

        # Stop streaming
        pipeline.stop()


if __name__ == "__main__":
    main()
