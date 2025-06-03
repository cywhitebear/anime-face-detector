import argparse
import pathlib

import cv2
import gradio as gr
from gradio.components import Image, Slider
import numpy as np
import PIL.Image
import torch

import anime_face_detector


def detect_with_angles(img, face_score_threshold: float, landmark_score_threshold: float) -> PIL.Image.Image:
    """
    Detect faces, draw bounding boxes, landmarks, and calculate 3D angles (pitch, yaw, roll).

    Args:
        img (str): Path to the input image.
        face_score_threshold (float): Threshold for face detection confidence.
        landmark_score_threshold (float): Threshold for landmark detection confidence.

    Returns:
        PIL.Image.Image: Processed image with bounding boxes, landmarks, and angles.
    """
    global detector  # Use the global detector object

    image = cv2.imread(img)
    if image is None:
        raise ValueError("Failed to read the image. Please check the file path or format.")

    preds = detector(image)

    res = image.copy()
    for pred in preds:
        box = pred['bbox']
        box, score = box[:4], box[4]
        if score < face_score_threshold:
            continue
        box = np.round(box).astype(int)

        # Draw bounding box
        lt = max(2, int(3 * (box[2:] - box[:2]).max() / 256))
        cv2.rectangle(res, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), lt)

        pred_pts = pred['keypoints']
        landmarks = []
        for i, (*pt, landmark_score) in enumerate(pred_pts):
            if landmark_score >= landmark_score_threshold:
                pt = np.round(pt).astype(int)
                landmarks.append(pt)
                # Paint the 6 PnP landmarks in blue
                if i in [23, 2, 22, 14, 26, 24]:  # Specific landmarks for PnP
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 255)
                cv2.circle(res, tuple(pt), lt, color, cv2.FILLED)
                # Uncomment the next line to label landmarks
                cv2.putText(res, str(i), tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Calculate 3D face angles using PnP
        if len(landmarks) >= 4:  # Ensure we have at least 4 landmarks for PnP
            try:
                pitch, yaw, roll = calculate_3d_angles(landmarks, res.shape[1], res.shape[0])

                if 23 < len(landmarks):  # Check if the landmark index 23 exists
                    draw_3d_axes(res, landmarks[23], pitch, yaw, roll)
                else:
                    draw_3d_axes(res, landmarks[0], pitch, yaw, roll)

                text_bg_color = (0, 0, 0, 128)
                text_color = (255, 255, 255)
                font_scale = 0.5
                thickness = 1

                draw_text_with_bg(res, f"Pitch: {pitch:.2f}", (box[0], box[1] - 60), text_color, text_bg_color, font_scale, thickness)
                draw_text_with_bg(res, f"Yaw: {yaw:.2f}", (box[0], box[1] - 40), text_color, text_bg_color, font_scale, thickness)
                draw_text_with_bg(res, f"Roll: {roll:.2f}", (box[0], box[1] - 20), text_color, text_bg_color, font_scale, thickness)
            except ValueError as e:
                print(f"PnP calculation skipped: {e}")

    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    image_pil = PIL.Image.fromarray(res)
    return image_pil


def draw_3d_axes(image, origin, pitch, yaw, roll, axis_length=50):
    """
    Draw 3D axes (X, Y, Z) on the image to represent pitch, yaw, and roll.

    Args:
        image (np.ndarray): The image to draw on.
        origin (tuple): The origin point (x, y) of the axes.
        pitch (float): Pitch angle in degrees.
        yaw (float): Yaw angle in degrees.
        roll (float): Roll angle in degrees.
        axis_length (int): Length of the axes.
    """
    # Convert angles to radians
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    roll = np.radians(roll)

    # Rotation matrix for pitch, yaw, roll
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch)],
                    [0, np.sin(pitch), np.cos(pitch)]])
    R_y = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                    [0, 1, 0],
                    [-np.sin(yaw), 0, np.cos(yaw)]])
    R_z = np.array([[np.cos(roll), -np.sin(roll), 0],
                    [np.sin(roll), np.cos(roll), 0],
                    [0, 0, 1]])

    # Combined rotation matrix
    R = R_z @ R_y @ R_x

    # Define the 3D axes in the local coordinate system
    axes = np.array([
        [axis_length, 0, 0],  # X-axis (red)
        [0, axis_length, 0],  # Y-axis (green)
        [0, 0, -axis_length]  # Z-axis (blue)
    ])

    # Rotate the axes
    rotated_axes = axes @ R.T

    # Project the 3D axes onto the 2D image plane
    for i, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):  # BGR: Blue, Green, Red
        end_point = (int(origin[0] + rotated_axes[i, 0]), int(origin[1] - rotated_axes[i, 1]))
        cv2.line(image, origin, end_point, color, 2)


def draw_text_with_bg(image, text, position, text_color, bg_color, font_scale, thickness):
    """
    Draw text with a semi-transparent background.

    Args:
        image (np.ndarray): The image to draw on.
        text (str): The text to draw.
        position (tuple): The bottom-left corner of the text.
        text_color (tuple): The color of the text (B, G, R).
        bg_color (tuple): The color of the background (B, G, R, A).
        font_scale (float): Font scale.
        thickness (int): Thickness of the text.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x, text_y = position
    bg_x1, bg_y1 = text_x, text_y - text_size[1] - 5
    bg_x2, bg_y2 = text_x + text_size[0] + 5, text_y + 5

    # Draw semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color[:3], -1)
    alpha = bg_color[3] / 255.0
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Draw text
    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness)


def calculate_3d_angles(landmarks: list, image_width: int, image_height: int) -> tuple:
    """
    Calculate the 3D angles (pitch, yaw, roll) of the face using the PnP algorithm.

    Args:
        landmarks (list): List of 2D facial landmarks.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        tuple: (pitch, yaw, roll) in degrees.
    """
    # Define a 3D model of facial landmarks
    model_points = np.array([
        (0.0, 0.0, 0.0),           # Nose tip
        (0.0, -55.0, -29.33),      # Chin
        # (-62.33, 29.33, -62.33),   # Left eye corner
        # (62.33, 29.33, -62.33),    # Right eye corner
        (-51.33, 14.67, -55.0),     # Left eye outer bottom corner
        (51.33, 14.67, -55.0),      # Right eye outer bottom corner
        # (-40.33, 14.67, -47.67),    # Left lower eye lid
        # (40.33, 14.67, -47.67),     # Right lower eye lid
        # (-29.33, 14.67, -55.0),     # Left eye inner bottom corner
        # (29.33, 14.67, -55.0),      # Right eye inner bottom corner
        (-18.33, -27.5, -25.67),    # Left mouth corner
        (18.33, -27.5, -25.67)      # Right mouth corner
    ], dtype=np.float32)

    # Required landmark indices
    required_indices = [23, 2, 22, 14, 26, 24]

    # Filter out missing landmarks
    valid_model_points = []
    valid_image_points = []
    for idx, model_point in zip(required_indices, model_points):
        if idx < len(landmarks):  # Check if the landmark exists
            valid_model_points.append(model_point)
            valid_image_points.append(landmarks[idx])

    # Ensure we have at least 4 points for PnP
    if len(valid_model_points) < 4:
        raise ValueError("Not enough landmarks detected for PnP calculation.")

    # Convert to numpy arrays
    valid_model_points = np.array(valid_model_points, dtype=np.float32)
    valid_image_points = np.array(valid_image_points, dtype=np.float32)

    # Camera matrix (intrinsic parameters)
    focal_length = image_width
    center = (image_width / 2, image_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    # Distortion coefficients (assume no distortion)
    dist_coeffs = np.zeros((4, 1))

    # Solve PnP to get rotation and translation vectors
    success, rotation_vector, translation_vector = cv2.solvePnP(
        valid_model_points, valid_image_points, camera_matrix, dist_coeffs
    )

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Extract pitch, yaw, roll from the rotation matrix
    pitch = np.arctan2(-rotation_matrix[2][1], rotation_matrix[2][2])
    yaw = np.arctan2(rotation_matrix[2][0], np.sqrt(rotation_matrix[2][1]**2 + rotation_matrix[2][2]**2))
    roll = np.arctan2(-rotation_matrix[1][0], rotation_matrix[0][0])

    # Convert radians to degrees
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    roll = np.degrees(roll)

    return pitch, yaw, roll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector',
                        type=str,
                        default='yolov3',
                        choices=['yolov3', 'faster-rcnn'])
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        choices=['cuda:0', 'cpu'])
    parser.add_argument('--face-score-threshold', type=float, default=0.5)
    parser.add_argument('--landmark-score-threshold', type=float, default=0.3)
    parser.add_argument('--score-slider-step', type=float, default=0.05)
    parser.add_argument('--port', type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()

    sample_path = pathlib.Path('input.jpg')
    if not sample_path.exists():
        torch.hub.download_url_to_file(
            'https://raw.githubusercontent.com/hysts/anime-face-detector/main/assets/input.jpg',
            sample_path.as_posix())

    # Define the detector as a global variable
    global detector
    detector = anime_face_detector.create_detector(args.detector, device=args.device)

    title = 'hysts/anime-face-detector with 3D Angles'
    description = 'Demo for hysts/anime-face-detector with 3D face angle calculation. Upload your image or use the examples below.'
    article = "<a href='https://github.com/hysts/anime-face-detector'>GitHub Repo</a>"

    gr.Interface(
        detect_with_angles,  # Pass the updated detect function
        [
            Image(type='filepath', label='Input'),
            Slider(0,
                   1,
                   step=args.score_slider_step,
                   default=args.face_score_threshold,
                   label='Face Score Threshold'),
            Slider(0,
                   1,
                   step=args.score_slider_step,
                   default=args.landmark_score_threshold,
                   label='Landmark Score Threshold'),
        ],
        gr.outputs.Image(type='pil', label='Output'),
        server_port=args.port,
        title=title,
        description=description,
        article=article,
        examples=[
            [
                sample_path.as_posix(),
                args.face_score_threshold,
                args.landmark_score_threshold,
            ],
        ],
        enable_queue=True,
        live=args.live,
    ).launch(debug=args.debug, share=args.share)


if __name__ == '__main__':
    main()