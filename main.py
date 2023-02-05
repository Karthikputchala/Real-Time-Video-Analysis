from frontend import about, display_result,input_selections
from functions import app
import streamlit as st
import os

snipped_files_path = r"Snipped_clips"
if not os.path.exists(snipped_files_path):
    os.makedirs(snipped_files_path)
Uploaded_images_path = r"Uploaded_images"
if not os.path.exists(Uploaded_images_path):
    os.makedirs(Uploaded_images_path)

# Introdution of the project
about()
# Take the User input
video_loc, image_loc= input_selections()
# If the "Generate button is pressed"
if st.button('Generate'):
    # Runs the main function
    output, time_stamps, list_of_extractedFilePaths, labels = app(image_loc,video_loc,snipped_files_path)
    # Displays the results to the user
    display_result(output, time_stamps, labels)


