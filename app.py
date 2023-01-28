from frontend import about, input, display_result
from functions import main
import streamlit as st

snipped_folder_path = r"C:\Users\personal\Projects\findme\Data\snipped_clips"

def run():
    # Introdution of the project
    about()
    # Take the User input
    video_loc, image_loc= input()
    # If the "Generate button is pressed"
    if st.button('Generate'):
        # Runs the main function
        output, time_stamps, list_of_extractedFilePaths, labels = main(image_loc,video_loc,snipped_folder_path)
        # Displays the results to the user
        display_result(output, time_stamps, labels)

run()
