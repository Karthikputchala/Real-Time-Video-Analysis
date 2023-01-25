import os
import cv2
import face_recognition

import numpy as np
from mtcnn import MTCNN
from decord import VideoReader, cpu
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip

import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from datetime import timedelta

#farcascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# detector
detector = MTCNN()
# videomae
image_processor = VideoMAEImageProcessor.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics")
model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics")

# Get the input image encodings
def input_image_encodings(input_image):
    # Load image file using the face_recognition library
    image = face_recognition.load_image_file(input_image)
    # Get the length of the face encodings found in the image
    encodings_length = len(face_recognition.face_encodings(image))
    if encodings_length>0:
        # Extract the first face encoding found in the image
        input_encoding = face_recognition.face_encodings(image)[0]
        # Read the image file using the cv2 library
        image = cv2.imread(input_image)
        # Flip the image horizontally
        flipped_image = cv2.flip(image, 1)
        # Flip the image horizontally using numpy
        flipped_image = np.fliplr(flipped_image)
        # Save the flipped image
        path = r"C:\Users\personal\Projects\findme\Data\Extracted_image\flipped_image.jpg"
        cv2.imwrite(path, flipped_image)
        # Load the flipped image using the face_recognition library
        flipped_image_recog = face_recognition.load_image_file(path)
        # Extract the face encoding from the flipped image
        flipped_encoding = face_recognition.face_encodings(flipped_image_recog)[0]
        # Set the wrong_image flag to False
        wrong_image = False
        # Create a list of encodings
        encodings = [input_encoding,flipped_encoding]
    else:
        # Set the wrong_image flag to True
        wrong_image = True
        # Create an empty list of encodings
        encodings = []
    # Return the wrong_image flag and the list of encodings
    return wrong_image, encodings

# Extracts the image from the frame
def extract_image_from_Frame(frame, result):
    # x1, y1, width, height = result['box']
    # Unpack the result tuple into x1, y1, width, height
    x1, y1, width, height = result
    # Get the absolute values of x1 and y1
    x1, y1 = abs(x1), abs(y1)
    # Calculate the x2 and y2 coordinates
    x2, y2 = x1 + width, y1 + height
    # Convert the frame from BGR to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Extract the face from the frame using numpy slicing
    face = img[y1:y2, x1:x2]
    # Create a PIL Image object from the extracted face
    image = Image.fromarray(face)
    # Convert the PIL Image object to a numpy array
    image = np.array(image)
    # Save the extracted image to file using cv2
    cv2.imwrite(r"C:\Users\personal\Projects\findme\Data\Extracted_image\extracted_image.jpg", image)

# Checks if the person is present in the frame (image) or not
def check_for_person(cap,fps,encoddings):
    # Load the unknown image from file
    unknown_image = face_recognition.load_image_file(
        r"C:\Users\personal\Projects\findme\Data\Extracted_image\extracted_image.jpg")
    # Get the length of face encodings found in the unknown image
    length = len(face_recognition.face_encodings(unknown_image))
    # Initialize the seconds variable
    seconds = 0
    # Initialize the person_present variable
    person_present = False
    if length > 0:
        # Extract the first face encoding found in the unknown image
        uknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        # Create an empty list to store the comparison results
        results = []
        # Loop through the list of known encodings
        for encod in encoddings:
            # Compare the unknown encoding to the current known encoding
            result = face_recognition.compare_faces([encod], uknown_encoding)
            # Append the result of the comparison to the results list
            results.append(result[0])
        # Check if all the comparison results are True
        if len(results)>1 and all(results):
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
                    extract_image_from_Frame(frame, result)
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
    # Modify the seconds
    list_of_seconds = modify_seconds(list_of_seconds)
    # Remove duplicates and sort the list
    list_of_seconds = sorted(list(set(list_of_seconds)))
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
    snipped_files_path = r"C:\Users\personal\Projects\findme\Data\snipped_clips"
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
        outputs = model(**inputs)
        logits = outputs.logits
    # model predicts one of the 400 Kinetics-400 classes
    predicted_label = logits.argmax(-1).item()
    # Get the label string
    label = model.config.id2label[predicted_label]
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

# Deletes the required files
def delete():
    # Delete all files in the snipped_files folder
    delete_files(r"C:\Users\personal\Projects\findme\Data\snipped_clips")
    # Delete all files in the Extracted images folder
    delete_files(r"C:\Users\personal\Projects\findme\Data\Extracted_image")
    # delete all uploaded images
    delete_files(r"C:\Users\personal\Projects\findme\Data\Uploaded_images")

# Cotains all the functions that needs to be executed in an order.
def main(input_image_path,video_path,snipped_folder_path):
    output = "none"
    labels = []
    time_stamps = []
    list_of_extractedFilePaths = []
    wrong_image, encodings = input_image_encodings(input_image_path)
    if wrong_image is False:
        list_of_seconds = get_seconds(video_path, encodings)
        if len(list_of_seconds)>0:
            grouped_list_of_seconds = group_the_seconds(list_of_seconds)
            time_stamps = get_timeStamps(grouped_list_of_seconds)
            list_of_extractedFilePaths = extract_clips(video_path, grouped_list_of_seconds)
            labels = gather_all_labels(snipped_folder_path)
        else:
            output = "Person is not present in the video"
    else:
        output = 'Upload a correct image of the person face'
    delete()
    return output, time_stamps, list_of_extractedFilePaths, labels


