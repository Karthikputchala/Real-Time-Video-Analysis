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

# Function to upload and read the image
def read_image():
    if not os.path.exists(Uploaded_images_path):
        os.makedirs(Uploaded_images_path)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    uploaded = False
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        cv2.imwrite(r"Uploaded_images/uploaded_image.jpg", image_array)
        st.image(image_array)
        uploaded = True
    return uploaded

# Gets the image location


def input_selections():
    col1, col2 = st.columns(2)
    video_options = ['GOT', 'PM Modi G20']
    image_options = ['Jhon snow', 'PM Modi','Upload an Image']

    with col1:
        video_choice = st.selectbox('Select a video to try out', video_options)
        video_loc = f"Videos/{video_choice}.mp4"
        print(video_loc)
        with open(video_loc, 'rb') as video:
            video_bytes = video.read()
        st.video(video_bytes)

    with col2:
        image_choice = st.selectbox('Select an image to try out', image_options)
        if image_choice == 'Upload an Image':
            img_loc = ''
            uploaded = read_image()
            if uploaded:
                img_loc = r"Uploaded_images/uploaded_image.jpg"
        else:
            img_loc = f"Images/{image_choice}.jpg"
            with Image.open(img_loc) as image:
                small_image = image_resize(image)
            st.image(small_image)

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
        
