import argparse
import csv
import io
import os
from typing import List

import cv2
import h5py
import numpy as np

from projectaria_tools.core import mps
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

from projectaria_tools.core import data_provider, image
from PIL import Image, ImageDraw

# from projectaria_tools.core.calibration import distort_image_and_calibration
from projectaria_tools.core import calibration

# from projectaria_tools.core.calibration import (
#     rotate_upright_image_and_calibration,
#     undistort_image_and_calibration
# )
# from projectaria_tools.core.mps.utils import (
#     get_nearest_eye_gaze
# )
# from projectaria_tools.core.mps.utils import get_nearest_pose
from PIL import Image
import json

from tqdm import tqdm


def str2table(v):
    return v.split(',')



def bisection_timestamp_search(timed_data, query_timestamp_ns: int) -> int:
    """
    Binary search helper function, assuming that timed_data is sorted by the field names 'tracking_timestamp'
    Returns index of the element closest to the query timestamp else returns None if not found (out of time range)
    """
    # Deal with border case
    if timed_data and len(timed_data) > 1:
        first_timestamp = timed_data[0].tracking_timestamp.total_seconds() * 1e9
        last_timestamp = timed_data[-1].tracking_timestamp.total_seconds() * 1e9
        if query_timestamp_ns <= first_timestamp:
            return None
        elif query_timestamp_ns >= last_timestamp:
            return None
    # If this is safe we perform the Bisection search
    start = 0
    end = len(timed_data) - 1
    while start < end:
        mid = (start + end) // 2
        mid_timestamp = timed_data[mid].tracking_timestamp.total_seconds() * 1e9
        if mid_timestamp == query_timestamp_ns:
            return mid
        if mid_timestamp < query_timestamp_ns:
            start = mid + 1
        else:
            end = mid - 1
    return start

def get_nearest_eye_gaze(eye_gazes, query_timestamp_ns):
    """
    Helper function to get nearest eye gaze for a timestamp (ns)
    Return the closest or equal timestamp eye_gaze information that can be found, returns None if not found (out of time range)
    """
    bisection_index = bisection_timestamp_search(eye_gazes, query_timestamp_ns)
    # print(eye_gazes, bisection_index, len(eye_gazes))
    if bisection_index is None:
        return None
    return eye_gazes[bisection_index]


def get_nearest_pose(
    mps_trajectory, query_timestamp_ns
):
    """
    Helper function to get nearest pose for a timestamp (ns)
    Return the closest or equal timestamp pose information that can be found, returns None if not found (out of time range)
    """
    bisection_index = bisection_timestamp_search(mps_trajectory, query_timestamp_ns)
    if bisection_index is None:
        return None
    return mps_trajectory[bisection_index]


def undistort_image_and_calibration(
    input_image: np.ndarray,
    input_calib: calibration.CameraCalibration,
) -> [np.ndarray, calibration.CameraCalibration]:
    """
    Return the undistorted image and the updated camera calibration.
    """
    input_calib_width = input_calib.get_image_size()[0]
    input_calib_height = input_calib.get_image_size()[1]
    input_calib_focal = input_calib.get_focal_lengths()[0]
    if (
        # numpy array report matrix shape as (height, width)
        input_image.shape[0] != input_calib_height
        or input_image.shape[1] != input_calib_width
    ):
        raise ValueError(
            f"Input image shape {input_image.shape} does not match calibration {input_calib.get_image_size()}"
        )

    # Undistort the image
    pinhole = calibration.get_linear_camera_calibration(
        int(input_calib_width),
        int(input_calib_height),
        input_calib_focal,
        "pinhole",
        input_calib.get_transform_device_camera(),
    )
    updated_calib = pinhole
    output_image = calibration.distort_by_calibration(
        input_image, updated_calib, input_calib
    )

    return output_image, updated_calib


def rotate_upright_image_and_calibration(
    input_image: np.ndarray,
    input_calib: calibration.CameraCalibration,
) -> [np.ndarray, calibration.CameraCalibration]:
    """
    Return the rotated upright image and update both intrinsics and extrinsics of the camera calibration
    NOTE: This function only supports pinhole and fisheye624 camera model.
    """
    output_image = np.rot90(input_image, k=3)
    updated_calib = calibration.rotate_camera_calib_cw90deg(input_calib)

    return output_image, updated_calib

parser = argparse.ArgumentParser()
parser.add_argument('--data_root',default='/home/fias/postdoc/datasets/nymeria/', type=str)
parser.add_argument('--id_files', default=[] , type=str2table)#"Nymeria_v0.0_20230607_s0_james_johnson_act0_e72nhq"
parser.add_argument('--id_exceptions', default=[] , type=str2table)#"Nymeria_v0.0_20230607_s0_james_johnson_act0_e72nhq"
parser.add_argument('--fps', default=5 , type=int)
parser.add_argument('--resolution', default=512 , type=int)
parser.add_argument('--mode', default="a" , type=str)
args = parser.parse_args()

# save ="nymeria"

datah5_name = os.path.join(args.data_root, f"data_fps{args.fps}_res{args.resolution}.h5")
# if not os.path.exists(datah5_name):
datah5 = h5py.File(datah5_name, args.mode)
datah5_grp = datah5.create_dataset("data", shape=(6167461,), dtype=h5py.vlen_dtype(np.dtype('uint8')))

csv_file = csv.writer(open(os.path.join(args.data_root, f"egodata_fps{args.fps}_res{args.resolution}.csv"),args.mode))

if args.mode == "w":
    csv_file.writerow(["file_id", "device_time_ns", "index", "gaze_x", "gaze_y", "quatw", "quatx", "quaty", "quatz", "tx", "ty", "tz","new_index","gaze_depth"])

if args.id_files:
    list_files = args.id_files
else:
    json_list_sequence = json.load(open(os.path.join(args.data_root, "Nymeria_download_urls.json" ),"r"))
    list_files = ["Nymeria_v0.0_"+k for k in json_list_sequence["sequences"].keys()]

args.id_exceptions = ["Nymeria_v0.0_"+k for k in args.id_exceptions]
os.makedirs(os.path.join(args.data_root, "tests"), exist_ok=True)

all_index = 0
for id_file in tqdm(list_files):
    if id_file in datah5 and args.mode == "a" and id_file not in args.id_exceptions:
        continue
    # Initialize all recording data
    try:
        vrs_camera = os.path.join(args.data_root, f"{id_file}_recording_head_data_data.vrs")
        provider_vrs = data_provider.create_vrs_data_provider(vrs_camera)
        stream_id = provider_vrs.get_stream_id_from_label("camera-rgb")
    except Exception as e:
        print("vrs", id_file)
        continue

    device_calibration = provider_vrs.get_device_calibration()
    rgb_camera_calibration_init = device_calibration.get_camera_calib("camera-rgb")

    T_device_CPF = device_calibration.get_transform_device_cpf()

    try:
        gaze_path = os.path.join(args.data_root,f"{id_file}_recording_head/recording_head/mps/eye_gaze/general_eye_gaze.csv")
        gaze_cpf = mps.read_eyegaze(gaze_path)
        closed_loop_path = os.path.join(args.data_root,f"{id_file}_recording_head/recording_head/mps/slam/closed_loop_trajectory.csv")
        closed_loop_traj = mps.read_closed_loop_trajectory(closed_loop_path)
    except Exception as e:
        print("gaze", id_file)
        continue

    # gaze_point_cpf = mps.get_eyegaze_point_at_depth(gaze_cpf[1].yaw, gaze_cpf[1].pitch, depth_m)
    total_size = provider_vrs.get_num_data(stream_id)

    device_time_ns = provider_vrs.get_first_time_ns(stream_id, TimeDomain.DEVICE_TIME)
    device_time_ns_end = provider_vrs.get_last_time_ns(stream_id, TimeDomain.DEVICE_TIME)

    cpt_false = 0
    j = 0



    while device_time_ns <= device_time_ns_end:
        data = provider_vrs.get_sensor_data_by_time_ns(stream_id, device_time_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)
        eye_gaze = get_nearest_eye_gaze(gaze_cpf, int(device_time_ns))

        pose_info = get_nearest_pose(closed_loop_traj, int(device_time_ns))
        # gps_data = provider_vrs.get_gps_data_by_time_ns(gps_id, int(device_time_ns), TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)
        device_time_ns += int(1e9 / args.fps)

        if eye_gaze is None or pose_info is None:
            print("continue", id_file, device_time_ns)
            cpt_false += 1
            continue

        try:
            img, record = data.image_data_and_record()
        except:
            print("image type", id_file, device_time_ns)
            continue
        img = img.to_numpy_array()

        try:
            img = image.debayer(img)
        except:
            img = img

        img, rgb_camera_calibration = undistort_image_and_calibration(img, rgb_camera_calibration_init)
        img, rgb_camera_calibration = rotate_upright_image_and_calibration(img, rgb_camera_calibration)
        gaze_vector_in_cpf = mps.get_eyegaze_point_at_depth(eye_gaze.yaw, eye_gaze.pitch, eye_gaze.depth)
        gaze_center_in_camera = (
                rgb_camera_calibration.get_transform_device_camera().inverse()
                @ T_device_CPF
                @ gaze_vector_in_cpf
        )
        gaze_projection = rgb_camera_calibration.project(gaze_center_in_camera)
        if gaze_projection is None:
            print("gaze continue", device_time_ns)
            cpt_false += 1
            continue


        # if j == 0:
        #     if id_file in args.id_exceptions:
        #         datah5_grp = datah5.get(id_file)
        #
        #     else:
        #         datah5_grp = datah5.create_dataset(id_file, shape=(total_size - cpt_false,), dtype=h5py.vlen_dtype(np.dtype('uint8')))

        w, h = img.shape[0], img.shape[1]
        r = min(args.resolution/w, args.resolution/h, 1)



        img = cv2.resize(img, dsize=(int(r*w), int(r*h)), interpolation=cv2.INTER_CUBIC)
        x, y = r*gaze_projection[0], r*gaze_projection[1]
        data = [id_file, device_time_ns, j, x ,y ] + list(pose_info.transform_world_device.to_quat_and_translation()[0]) + [all_index, eye_gaze.depth]
        csv_file.writerow(data)


        def encode_np_image(np_img: np.ndarray, format="JPEG") -> bytes:
            """
            Encode a NumPy image (HWC, uint8) into bytes (JPEG or PNG).
            """
            pil_img = Image.fromarray(np_img.astype(np.uint8))  # assumes RGB, dtype=uint8
            buffer = io.BytesIO()
            pil_img.save(buffer, format=format)
            return buffer.getvalue()  # returns raw bytes


        adj_resolution = int(args.resolution * 0.9)
        # r = min(1408/w, 1408/h, 1)
        center = img.shape[0] // 2, img.shape[1] // 2
        x = center[1] - adj_resolution // 2
        y = center[0] - adj_resolution // 2
        crop_img = img[y:y + adj_resolution, x:x + adj_resolution]

        datah5_grp[all_index] = np.frombuffer(encode_np_image(crop_img, "JPEG"), dtype=np.uint8)
        all_index += 1

        # if j < 30:
        #     Image.fromarray(img).save(os.path.join(os.path.join(args.data_root, "tests", f"{j}.png")))




        # r = 5
        # img = Image.fromarray(img)
        # draw = ImageDraw.Draw(img)
        # draw.ellipse((x-r, y- r, x + r, y + r), fill=(255,0,0,0) )
        # img.save(f"/home/fias/postdoc/gym_results/test_images/{save}/%3d.jpeg" % j)


        j+=1
    # print(id_file, num_file)
        # if j%100 == 0:
        #     print(j)
        # if j > 100:
        #     break

datah5.close()