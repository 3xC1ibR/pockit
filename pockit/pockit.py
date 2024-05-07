import os
import cv2
import mediapipe as mp
from .constants import deeplab

from .utils import file_exists
import time
import numpy as np
from tensorflow.keras.models import load_model
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.__version__)

from tensorflow.keras.utils import CustomObjectScope

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
    print('Loading model...')
    print('Model:', deeplab)
    model = tf.keras.models.load_model(deeplab)

    model_input_shape = model.input_shape  # This will give you a tuple like (None, height, width, channels)
    print(f"The model expects input shape: {model_input_shape}")


def load(video_path):
    def preprocess_image(frame, model_input_size):
        """
        Preprocess the input frame for segmentation model.

        Parameters:
        - frame: The input image frame (numpy array).
        - model_input_size: The required input size of the model (tuple of height and width).

        Returns:
        - preprocessed_frame: The preprocessed image frame ready for model input.
        """
        # Resize the frame to the required input dimensions
        print('frame:', frame.shape)
        resized_frame = cv2.resize(frame,
                                   (model_input_size[1], model_input_size[0]))  # cv2.resize expects (width, height)

        # Convert color space from BGR to RGB
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Normalize pixel values to be between 0 and 1
        normalized_frame = rgb_frame / 255.0

        # Add a batch dimension because deep learning models expect a batch of images as input
        preprocessed_frame = np.expand_dims(normalized_frame, axis=0)

        return preprocessed_frame

    def get_person_segmentation(preprocessed_frame, model):
        """
        Generate a segmentation mask for the person in the preprocessed frame.

        Parameters:
        - preprocessed_frame: A numpy array representing the preprocessed video frame for model input.
        - model: The loaded DeepLab model or similar segmentation model.

        Returns:
        - segmentation_mask: A binary mask indicating the location of the person within the frame.
        """
        # Predict the segmentation mask
        model_predictions = model(preprocessed_frame)

        # Assuming the model output is in 'model_predictions', and it returns a tensor
        # The output shape is expected to be (1, height, width, channels) where 'channels' would typically
        # be the number of classes. For a person segmentation model, there might be a dedicated channel
        # for 'person' class, or you may need to select it based on your model's documentation.

        # Convert model output to a binary mask (1 for person, 0 for background)
        # This step highly depends on how your model structures its output.
        # Below is a generic approach assuming a single channel output that needs thresholding.

        # Extract segmentation output and remove batch dimension
        segmentation_output = model_predictions[0].numpy() if isinstance(model_predictions, tf.Tensor) else \
            model_predictions[0]

        # Determine the 'person' class channel index based on your model's documentation
        # For illustration, we're assuming it's the first channel
        person_channel_index = 0  # Adjust based on your model

        # Threshold the segmentation output to generate a binary mask
        # Adjust the threshold as needed based on your model's output range
        threshold = 0.5
        segmentation_mask = segmentation_output[:, :, person_channel_index] > threshold

        # Convert to uint8 for consistency with image data types
        segmentation_mask = segmentation_mask.astype(np.uint8) * 255

        return segmentation_mask

    file_exists(video_path)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))
    model_input_size = (model_input_shape[1], model_input_shape[2])
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        preprocessed_frame = preprocess_image(frame, model_input_size)
        # show the frame
        cv2.imshow('Frame', frame)
        segmentation_mask = get_person_segmentation(preprocessed_frame, model)

        # # Use segmentation mask to isolate the person
        isolated_person = frame * segmentation_mask[:, :, None]

        out.write(isolated_person)

    cap.release()
    out.release()


def load4(video_path):
    file_exists(video_path)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    # Initialize video capture and output
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        exit()

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # Setup video writer
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))

    while cap.isOpened():
        success, frame = cap.read()
        print('Frame:', cap.get(cv2.CAP_PROP_POS_FRAMES) - 1, '/', cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        if not success:
            break

        # Process the video frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Create a blank frame (change np.zeros to np.ones and multiply by 255 for white background)
        blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        # Check if pose landmarks are detected and draw only the pose of the person of interest
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(blank_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # show the frame

        cv2.imshow('Frame', blank_frame)
        # Write the frame with the isolated person to the output video
        out.write(blank_frame)

    # Release resources
    cap.release()
    out.release()


def load3(video_path):
    """
    extract first person from video and save it as a new video
    :param video_path:
    :return:
    """

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    frame_count = 0

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        cap.release()
        exit(1)

    selected_person_index = None  # Index of the selected person (for simplicity, not used here)
    # Prepare video writer
    frame_height, frame_width = frame.shape[:2]
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))

    while cap.isOpened():
        print('Frame:', frame_count)
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect pose
        results = pose.process(frame_rgb)

        # Draw the pose annotations on the frame
        if results.pose_landmarks:
            # For the first person detected (simplification for this example)
            # Additional logic to select a specific person can go here
            # For now, we proceed with the detected pose as is

            # Creating a mask for the selected person
            mask = np.zeros(frame.shape, dtype=np.uint8)
            mp_drawing.draw_landmarks(mask, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                                   thickness=2, circle_radius=1),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                                     thickness=2, circle_radius=2))
            # Convert mask to grayscale and threshold it
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask_thresh = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)

            # Use the mask to isolate the person
            isolated_person = cv2.bitwise_and(frame, frame, mask=mask_thresh)
            cv2.imshow('Isolated Person', isolated_person)

            if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
                break
            out.write(isolated_person)

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def load2(video_path):
    file_exists(video_path)
    print('Loading video...')
    print('Video:', video_path)
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    success, frame = cap.read()
    if not success:
        print("Failed to read video")
        cap.release()
        exit(1)

    # Manually select the bounding box around the person
    bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    # Initialize pose estimation
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Extract person ROI (Region of Interest) using the bbox coordinates
        # Ensure bbox has valid coordinates before attempting to extract ROI
        if bbox and all(bbox):
            x, y, w, h = bbox
            person_roi = frame[y:y + h, x:x + w]

            # Convert the ROI to RGB for MediaPipe
            person_roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)

            # Perform pose estimation on the ROI
            results = pose.process(person_roi_rgb)

            # Draw the pose annotation on the ROI
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    person_roi, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Display the resulting frame with ROI
        cv2.imshow('MediaPipe Pose', frame)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
            break


output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)


def load1(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    frame_num = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Finished processing video.")
            break

        # Convert the BGR image to RGB.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and find poses.
        results = pose.process(image_rgb)

        # Draw pose landmarks.
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Save the image with drawn landmarks
        cv2.imwrite(os.path.join(output_dir, f'frame_{frame_num:04d}.png'), image)
        frame_num += 1

    cap.release()


def get_duration(cap):
    return cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)


def get_info(video_path):
    file_exists(video_path)
    cap = cv2.VideoCapture(video_path)
    get_duration(cap)
    return {
        'frame_width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        'frame_height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        'frame_count': cap.get(cv2.CAP_PROP_FRAME_COUNT),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'duration': get_duration(cap)
    }
