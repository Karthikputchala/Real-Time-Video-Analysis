import streamlit as st
from PIL import Image
import cv2
import pandas as pd
from io import StringIO
import numpy as np

from functions import delete

# Has all the introduction about the Project
def about():
    st.set_page_config(layout="wide",initial_sidebar_state = "expanded")
    Github = 'https://www.fao.org/faostat/en/#data/GB/visualize'
    view_raw_data = "https://raw.githubusercontent.com/Karthikputchala/FAOSTAT-Burning-Crop-Residues/main/data/Emissions_Agriculture_Burning_crop_residues_E_All_Data_(Normalized).csv"
    st.header(':black[Real-time Person Detection and Action Recognition in Video Surveillance]')
    st.write('About: This project allows for the identification of individuals in video footage and the detection of\
         the scene happening during their presence in real-time. This technology can be implemented in a variety of settings, such as security systems,\
        retail stores, and public spaces. It utilizes advanced machine learning algorithms to analyze video footage and\
             identify individuals based on their unique characteristics, such as facial features and body\
                 movements.'+ " [(Github)](%s)" % Github)
    st.write("Usage: The system allows for the user to either select or upload an image, and then select the video option to test for\
         real-time person identification and scene detection.")

# Resizes the image
def image_resize(image):
    small_size = (225, 225) # width, height
    small_image = image.resize(small_size)
    return small_image

# Function to upload and read the image
def read_image():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    uploaded = False
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        small_image = image_resize(image)
        image_array = np.array(image)
        cv2.imwrite(r"C:\Users\personal\Projects\findme\Data\Uploaded_images\uploaded_image.jpg", image_array)
        print("saved_image")
        st.image(image_array)
        uploaded = True
    return uploaded

# Gets the image location
def image_location(none, uploaded, img_code):
    img_loc = ""
    if none is True and uploaded is True:
        img_loc = r"C:\Users\personal\Projects\findme\Data\Uploaded_images\uploaded_image.jpg"
    elif none is True and uploaded is False:
        img_loc = ""
    elif none is False and uploaded is False:
        if img_code == "one":
            img_loc = r'C:\Users\personal\Projects\findme\Data\Images\image-1.jpg'
        elif img_code == "two":
            img_loc = r'C:\Users\personal\Projects\findme\Data\Images\image-2.jpg'
        elif img_code == "three":
            img_loc = r'C:\Users\personal\Projects\findme\Data\Images\image-3.jpg'
    return img_loc

# Takes the user input selections
def input():
    col1, col2 = st.columns(2)
    with col1:
        video_option = st.selectbox(
            'Select a video to try out',
            ('Video-1', 'Video-2', 'Video-3'))
        if video_option == "Video-1":
            video_loc = r'C:\Users\personal\Projects\findme\Data\Videos\video-1.mp4'
            video = open(r'C:\Users\personal\Projects\findme\Data\Videos\video-1.mp4', 'rb')
            video_bytes = video.read()
            st.video(video_bytes)
        elif video_option == "Video-2":
            video_loc = r'C:\Users\personal\Projects\findme\Data\Videos\video-1.mp4'
            video = open(r'C:\Users\personal\Projects\findme\Data\Videos\video-1.mp4', 'rb')
            video_bytes = video.read()
            st.video(video_bytes)
        else:
            video_loc = r'C:\Users\personal\Projects\findme\Data\Videos\video-1.mp4'
            video = open(r'C:\Users\personal\Projects\findme\Data\Videos\video-1.mp4', 'rb')
            video_bytes = video.read()
            st.video(video_bytes)
    with col2:
        Image_option = st.selectbox(
            'Select an image to try out',
            ('Image-1', 'Image-2', 'Image-3','Upload an Image'))
        if Image_option == "Image-1":
            delete()
            image = Image.open(r'C:\Users\personal\Projects\findme\Data\Images\image-1.jpg')
            small_image = image_resize(image)
            st.image(small_image)
            img_code = "one"
            none = False
            uploaded = False
        elif Image_option == "Image-2":
            delete()
            image = Image.open(r'C:\Users\personal\Projects\findme\Data\Images\image-2.jpg')
            small_image = image_resize(image)
            st.image(small_image)
            img_code = "two"
            none = False
            uploaded = False
        elif Image_option == "Image-3":
            delete()
            image = Image.open(r'C:\Users\personal\Projects\findme\Data\Images\image-3.jpg')
            small_image = image_resize(image)
            st.image(small_image)
            img_code = "three"
            none = False
            uploaded = False
        else:
            none = True
            img_code = 'not_selected'
            uploaded = read_image()

    img_loc = image_location(none, uploaded, img_code)

    return video_loc, img_loc

# Displays the results to the User.
def display_result(output, time_stamps, labels):
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if output == 'none':
        col1, col2 = st.columns(2)
        # headings
        with col1:
            st.subheader('Time stamp')
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
    pass
        
