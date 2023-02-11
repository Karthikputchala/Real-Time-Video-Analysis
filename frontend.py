import streamlit as st
from PIL import Image
import cv2
import pandas as pd
from io import StringIO
import numpy as np
import os
from functions import delete

Uploaded_images_path = r"Uploaded_images"
# Has all the introduction about the Project
def about():
    st.set_page_config(initial_sidebar_state = "expanded")
    Github = 'https://www.fao.org/faostat/en/#data/GB/visualize'
    view_raw_data = "https://raw.githubusercontent.com/Karthikputchala/FAOSTAT-Burning-Crop-Residues/main/data/Emissions_Agriculture_Burning_crop_residues_E_All_Data_(Normalized).csv"
    st.header(':black[Real-time Person Detection and Action Recognition in Video Surveillance]')
    st.write('About: This project allows for the identification of individuals in video footage and the detection of\
         the scene happening during their presence in real-time. This technology can be implemented in a variety of settings, such as security systems,\
        retail stores, and public spaces. It utilizes advanced machine learning algorithms to analyze video footage and\
             identify individuals based on their unique characteristics, such as their facial features.'+ " [(Github)](%s)" % Github)
    st.write("Usage: The system allows for the user to either select or upload an image, and then select the video option to test for\
         real-time person identification and their respective scene detection.")

# Resizes the image
def image_resize(image):
    small_size = (225, 225) # width, height
    small_image = image.resize(small_size)
    return small_image

# Reads the uploaded image
def read_image():
    # Check if the Uploaded_images_path directory exists
    if not os.path.exists(Uploaded_images_path):
        # If the directory doesn't exist, create it
        os.makedirs(Uploaded_images_path)
    # Allow the user to upload a file using a file uploader from the Streamlit library
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    # Initialize a flag to track if an image was uploaded
    uploaded = False
    # If a file was uploaded
    if uploaded_file is not None:
        # Open the image using the Pillow library
        image = Image.open(uploaded_file)
        # Convert the image to a numpy array
        image_array = np.array(image)
        # Save the image as a jpeg to the Uploaded_images_path directory
        cv2.imwrite(r"Uploaded_images/uploaded_image.jpg", image_array)
        # Display the image using the Streamlit library
        st.image(image_array)
        # Set the uploaded flag to True
        uploaded = True
    # Return the value of the uploaded flag
    return uploaded


# Gets the Video and image location
def input_selections():
    # Create two columns in the Streamlit app using the `st.columns` function
    col1, col2 = st.columns(2)
    # Define a list of video options to display in a selectbox
    video_options = ['Tom Holland']
    # Define a list of image options to display in a selectbox
    image_options = ['Tom Holland','Upload an Image']

    with col1:
        # Allow the user to select a video from the video_options list using a selectbox
        video_choice = st.selectbox('Select a video to try out', video_options)
        # Get the location of the selected video
        video_loc = f"Videos/{video_choice}.mp4"
        # Open the video file and read its contents into memory
        with open(video_loc, 'rb') as video:
            video_bytes = video.read()
        # Display the video in the Streamlit app using the `st.video` function
        st.video(video_bytes)

    with col2:
        # Allow the user to select an image from the image_options list using a selectbox
        image_choice = st.selectbox('Select an image to try out', image_options)
        # If the user selects the "Upload an Image" option
        if image_choice == 'Upload an Image':
            # Initialize an empty location for the uploaded image
            img_loc = ''
            # Call the `read_image` function to allow the user to upload an image
            uploaded = read_image()
            # If an image was uploaded
            if uploaded:
                # Set the location of the uploaded image
                img_loc = r"Uploaded_images/uploaded_image.jpg"
        # If the user selects one of the predefined images
        else:
            # Get the location of the selected image
            img_loc = f"Images/{image_choice}.jpg"
            # Open the image using the Pillow library
            with Image.open(img_loc) as image:
                # Call the `image_resize` function on the image
                small_image = image_resize(image)
            # Display the resized image in the Streamlit app using the `st.image` function
            st.image(small_image)

    # Return the locations of the selected video and image
    return video_loc, img_loc


# Displays the results to the User.
def display_result(output, time_stamps, labels):
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if output == 'none':
        col1, col2 = st.columns(2)
        # headings
        with col1:
            st.subheader('Identified at (Time stamp)')
        with col2:
            st.subheader('Action recognized')
        # Details
        for i , timeStamp in enumerate(time_stamps):
            with st.container():
                with col1:
                    st.write(timeStamp)
                with col2:
                    st.write(labels[i])
    else:
        st.write(output)
        
