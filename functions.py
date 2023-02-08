import os
import cv2
from PIL import Image
from datetime import timedelta
import io
import numpy as np
import pandas as pd

from decord import VideoReader, cpu
from moviepy.video.io.VideoFileClip import VideoFileClip

import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet152, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras import metrics

import mediapipe as mp
import streamlit as st

mp_face_detection = mp.solutions.face_detection

#farcascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# videomae
image_processor = VideoMAEImageProcessor.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics")

videoMAE_model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics")

ResNet152_model = ResNet152(include_top=False, weights='imagenet', pooling='avg')

# Paths
snipped_files_path = r"Snipped_clips"
Uploaded_images_path = r"Uploaded_images"
frame_face_path = r"Generated data/frame_face.jpg"
original_img_path = r'Generated data/image.jpg'
flipped_img_path = r'Generated data/flipped_image.jpg'
# Get the input image encodings

# Checks the face in an image
def check_for_face(image_path):
    # Read the image from the file path
    image = cv2.imread(image_path)
    # Use the multiprocessing-based-face-detection library to detect faces
    with mp_face_detection.FaceDetection(
        # Set the model to use for face detection to model number 1
        model_selection=1, 
        # Set the minimum detection confidence to 0.5
        min_detection_confidence=0.5) as face_detection:
        # Run the face detection process
        results = face_detection.process(
            # Convert the image from BGR to RGB color space
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Return the results of the face detection and the original image
        return results, image

# Get the bounding boxes of the faces
def bounding_boxes(results):
    bboxes = []
    for detection in results.detections:
        xmin = detection.location_data.relative_bounding_box.xmin
        ymin = detection.location_data.relative_bounding_box.ymin
        width = detection.location_data.relative_bounding_box.width
        height = detection.location_data.relative_bounding_box.height
        bbox = (xmin, ymin, width, height)
        bboxes.append(bbox)
    return bboxes
    
# Generates the image embedding
# The function returns the image embedding by using a pre-trained model.
def return_image_embedding(model,img_path):
    # Load the image with target size of (224,224)
    img = image.load_img(img_path, target_size=(224, 224))
    # Convert the image to array
    x = image.img_to_array(img)
    # Expand the dimensions of the image to include the batch size
    x = np.expand_dims(x, axis=0)
    # Preprocess the image for the model input
    x = preprocess_input(x)
    # Predict the embedding for the image
    preds = model.predict(x)
    # Create a DataFrame for the embedding and return it
    curr_df = pd.DataFrame(preds[0]).T
    return curr_df

def input_image_encodings(bboxes, image):
    """
    Calculates the image embeddings for original and flipped image.

    Parameters:
        bboxes (list): list of bounding boxes for each face detected
        image (np.array): numpy array of the image

    Returns:
        embeddings (list): list of image embeddings (original and flipped)
    """
    embeddings = []
    for box in bboxes:
        x, y, w, h = box
        x, y, w, h = [int(val * image.shape[1]) for val in [x, y, w, h]]
        face = image[y:h, x:w]
        flipped = cv2.flip(face, 1)
        cv2.imwrite(original_img_path, face)
        cv2.imwrite(flipped_img_path, flipped)
        image = Image.open(original_img_path)
        st.image(image, caption='Sunrise by the mountains')
        original_embedding = return_image_embedding(ResNet152_model, original_img_path).values[0]
        flipped_embedding = return_image_embedding(ResNet152_model, flipped_img_path).values[0]
        embeddings.append([original_embedding, flipped_embedding])
    return embeddings


# Extracts the image from the frame
def cropped_face(frame, result):
    # x1, y1, width, height = result['box']
    # Unpack the result tuple into x1, y1, width, height
    x1, y1, width, height = result
    # Get the absolute values of x1 and y1
    x1, y1 = abs(x1), abs(y1)
    # Calculate the x2 and y2 coordinates
    x2, y2 = x1 + width, y1 + height
    # Convert the frame from BGR to RGB
    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Extract the face from the frame using numpy slicing
    face = frame[y1:y2, x1:x2]
    # Create a PIL Image object from the extracted face
    image = Image.fromarray(face)
    # Convert the PIL Image object to a numpy array
    image = np.array(image)
    # Save the extracted image to file using cv2
    cv2.imwrite(frame_face_path, image)
    #return image

def cosine_similarity(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

# Checks if the person is present in the frame (image) or not
def check_for_person(cap,fps,inputImg_embeddings):
    frameFace_embedding = return_image_embedding(ResNet152_model,frame_face_path).values[0]
    # Get the length of face encodings found in the unknown image
    length = len(frameFace_embedding)
    # Initialize the seconds variable
    seconds = 0
    # Initialize the person_present variable
    person_present = False
    if length > 0:
        # Create an empty list to store the comparison results
        results = []
        # Loop through the list of known encodings
        for embed in inputImg_embeddings:
            # Compare the unknown encoding to the current known encoding
            result = cosine_similarity(embed, frameFace_embedding)
            #result = face_recognition.compare_faces([encod], uknown_encoding)
            # Append the result of the comparison to the results list
            results.append(result)
        # Check if all the comparison results are True
        if len(results)>1:
            for val in results:
                if val > 0.7:
                    # Get the timestamp of the current frame
                    timestamp = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    # Calculate the seconds from the timestamp and fps
                    seconds = timestamp/int(fps)
                    # Set the person_present variable to True
                    person_present = True
        else:
            # If the person is not present, set seconds to 0 and person_present to False
            seconds = 0
            person_present = False
    # Return the seconds and the person_present
    return seconds, person_present

# Gets the entire length of a video
def get_video_length(filepath):
    # Open the video file using the VideoFileClip class from moviepy library
    clip = VideoFileClip(filepath)
    # Get the duration of the video
    return clip.duration

# Modify the seconds
def modify_seconds(seconds_list):
    # Create an empty list to store the modified values
    new_list = []
    # Iterate over the seconds_list
    for i, value in enumerate(seconds_list):
        # If it is not the first element
        if i > 0:
            # Append the previous second
            new_list.append(seconds_list[i]-1)
        # Append the current second
        new_list.append(value)
        # If it is not the last element
        if i < len(seconds_list)-1:
            # Append the next second
            new_list.append(seconds_list[i]+1)
    # Return the modified list
    return new_list

# Get the list of seconds (timestamps) at where the person is present in the frame
def get_seconds(filepath, encoddings):
    # Create an empty list to store the seconds
    list_of_seconds = []
    # Open the video file using the cv2 library
    cap = cv2.VideoCapture(filepath)
    # Get the fps of the video
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    # Get the length of the video in seconds
    length_in_seconds = int(get_video_length(filepath))
    while True:
        # Get the current frame
        ret, frame = cap.read()
        # Get the timestamp of the current frame
        timestamp = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if ret is True and timestamp % int(fps) == 0:
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces in the frame
            results = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(results) > 0:
                # Iterate through the detected faces
                for result in results:
                    # Extract the image of the face from the frame
                    cropped_face(frame, result)
                    #extract_image_from_Frame(frame, result)
                    # Check if the face belongs to a person in the known encodings
                    seconds, person_present = check_for_person(cap,fps,encoddings)
                    # If the person is present, append the timestamp to the list
                    if person_present is True:
                        list_of_seconds.append(round(seconds))
        # Check if the video has ended
        if timestamp/int(fps) >= length_in_seconds:
            break
    # Close all windows
    try:
        cv2.destroyAllWindows()
    except:
        pass
    print(list_of_seconds)
    # Modify the seconds
    list_of_seconds = modify_seconds(list_of_seconds)
    print(list_of_seconds)
    # Remove duplicates and sort the list
    list_of_seconds = sorted(list(set(list_of_seconds)))
    print(list_of_seconds)
    # Return the list of seconds
    return list_of_seconds

# Groups the time stamps to a clip length 
def group_the_seconds(list_of_seconds):
    # Initialize an empty list to store the grouped numbers
    grouped_list_of_seconds = []
    # Initialize a temporary list to store numbers that have a distance of 1 or 2
    temp = [list_of_seconds[0]]
    # Iterate through the list_of_seconds list
    for i in range(1, len(list_of_seconds)):
        # Calculate the absolute difference between the current and previous number
        diff = abs(list_of_seconds[i] - list_of_seconds[i-1])
        # Check if the difference is equal to 1 or 2
        if diff == 1 or diff == 2:
            # If the difference is equal to 1 or 2, append the current number to the temporary list
            temp.append(list_of_seconds[i])
        else:
            # If the difference is not equal to 1 or 2, check the length of the temporary list
            if len(temp) > 1:
                # If the length of the temporary list is greater than 1, append it to the final list
                grouped_list_of_seconds.append(temp)
            # Clear the temporary list
            temp = [list_of_seconds[i]]
    # After the loop, check the length of the temporary list
    if len(temp) > 1:
        # If the length of the temporary list is greater than 1, append it to the final list
        grouped_list_of_seconds.append(temp)
    return grouped_list_of_seconds

# Extract these clips from the video
def extract_clips(video_path, grouped_list_of_seconds):
    if not os.path.exists(snipped_files_path):
        os.makedirs(snipped_files_path)
    list_of_extractedFilePaths = []
    for i, group in enumerate(grouped_list_of_seconds):
        if len(group) <= 3:
            min_time = min(group)-1
            max_time = max(group)+1
        else:
            min_time = min(group)
            max_time = max(group)
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        # Calculate the starting and ending frame numbers for the desired segment
        # 7 seconds * frames per second
        start_frame = int(min_time * cap.get(cv2.CAP_PROP_FPS))
        # 9 seconds * frames per second
        end_frame = int(max_time * cap.get(cv2.CAP_PROP_FPS))
        # Define the codec and create a video writer object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for mp4 files
        file_name = "splitted_video_"+str(i)+".mp4"
        file_path = os.path.join(snipped_files_path, file_name)
        out = cv2.VideoWriter(file_path, fourcc, cap.get(
            cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))), isColor=True)
        # Loop through the frames, starting at the desired start frame
        for i in range(start_frame):
            cap.grab()
        # Loop through the desired segment of frames
        for i in range(start_frame, end_frame):
            # Read the next frame
            ret, frame = cap.read()
            if ret:
                # Write the frame to the output video file
                out.write(frame)
            else:
                break
        # Release the resources
        cap.release()
        out.release()
        list_of_extractedFilePaths.append(file_path)
    return list_of_extractedFilePaths

# Gets the time stamps to display during the result
def get_timeStamps(grouped_list_of_seconds):
    time_stamps = []
    for i, group in enumerate(grouped_list_of_seconds):
        if len(group) <= 3:
            min_time = min(group)-1
            max_time = max(group)+1
        else:
            min_time = min(group)
            max_time = max(group)
        timestamp = str(timedelta(seconds=min_time)) +' - '+ str(timedelta(seconds=max_time))
        time_stamps.append(timestamp)
    return time_stamps

# --- got this code from ()-----
def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    # convert clip_len from seconds to frames
    converted_len = int(clip_len * frame_sample_rate)
    # select a random end index of the segment
    end_idx = np.random.randint(converted_len, seg_len)
    # calculate the start index
    start_idx = end_idx - converted_len
    # create an array of indices of the frames that will be used in the clip
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    # ensure that the indices are within the valid range of the segment
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    # return the indices
    return indices

# extract the labels for each clip using the ML model
def get_label(snipped_filepath):
    # Create a VideoReader object
    videoreader = VideoReader(snipped_filepath, num_threads=3, ctx=cpu(0))
    # Seek to the beginning of the video
    videoreader.seek(0)
    # Sample the frame indices
    indices = sample_frame_indices(clip_len=16, frame_sample_rate=4, seg_len=len(videoreader))
    # Get the video frames
    video = videoreader.get_batch(indices).asnumpy()
    # Preprocess the video frames
    inputs = image_processor(list(video), return_tensors="pt")
    # Use the model to predict the label
    with torch.no_grad():
        outputs = videoMAE_model(**inputs)
        logits = outputs.logits
    # model predicts one of the 400 Kinetics-400 classes
    predicted_label = logits.argmax(-1).item()
    # Get the label string
    label = videoMAE_model.config.id2label[predicted_label]
    return label
#--------------------------------

# Gathers all the labels for the clips into a list
def gather_all_labels(snipped_folder_path):
    labels = []
    # Iterate through all the files in the folder
    for file_name in os.listdir(snipped_folder_path):
        # Get the full filepath
        snipped_filepath = os.path.join(snipped_folder_path, file_name)
        # Check if the file is a valid file
        if os.path.isfile(snipped_filepath):
            # Get the label of the video
            label = get_label(snipped_filepath)
            # Append the label to the list
            labels.append(label)
    return labels

# Deletes the file
def delete_files(folder_path):
    # Iterate through all the files in the folder
    for file_name in os.listdir(folder_path):
        # Get the full filepath
        file_path = os.path.join(folder_path, file_name)
        # Attempt to delete the file
        try:
            # Check if the file is a valid file
            if os.path.isfile(file_path):
                # Remove the file
                os.remove(file_path)
        except Exception as e:
            # Catch any errors that may occur and continue
            pass

def deleteFile(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)
        
# Deletes the required files
def delete():
    # Delete all files in the snipped_files folder
    delete_files(snipped_files_path)
    # delete all uploaded images
    delete_files(Uploaded_images_path)
    #deleteFiles
    deleteFile(original_img_path)
    deleteFile(frame_face_path)
    deleteFile(flipped_img_path)

# Cotains all the functions that needs to be executed in an order.
def app(input_image_path,video_path,snipped_folder_path):
    output = "none"
    labels = []
    time_stamps = []
    list_of_extractedFilePaths = []
    results, image = check_for_face(input_image_path)
    if results.detections:
        bboxes = bounding_boxes(results)
        embeddings = input_image_encodings(bboxes,image)
        list_of_seconds = get_seconds(video_path, embeddings)
        print(list_of_seconds)
        if len(list_of_seconds)>0:
            grouped_list_of_seconds = group_the_seconds(list_of_seconds)
            print(grouped_list_of_seconds)
            time_stamps = get_timeStamps(grouped_list_of_seconds)
            list_of_extractedFilePaths = extract_clips(video_path, grouped_list_of_seconds)
            labels = gather_all_labels(snipped_folder_path)
        else:
            output = "Person is not present in the video"
    else:
        output = 'Upload a correct image of the person face'
    delete()
    return output, time_stamps, list_of_extractedFilePaths, labels

