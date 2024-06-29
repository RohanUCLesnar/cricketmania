# import streamlit as st 
# st.set_page_config(
#     page_title = "Analytics Dashboard",
#     page_icon = "🏏"
# )
# st.title("Main Page")
# st.sidebar.success("Select a page from above.")

import streamlit as st
import base64
from pathlib import Path
import os 
# Set page configuration
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="🏏",
    layout="wide"
)

# Function to add background image from local file
def add_bg_from_local(image_path, position_x, position_y, size_width, size_height):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url(data:image/png;base64,{encoded_image});
             background-size: {size_width} {size_height};
             background-position: {position_x} {position_y};
             background-repeat: no-repeat;
             background-attachment: fixed;
             height: 100vh;
             width: 100vw;
             overflow: hidden;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
# Image path 
image_path = os.path.join(os.getcwd(), "CRICKET MANIA - IPL.png")
# Set the exact coordinates for positioning
position_x = "220px"  # X coordinate
position_y = "60px"  # Y coordinate
# Set the size of the background image
size_width = "90%"  # Width as a percentage or specific value like "500px"
size_height = "100%"  # Height as a percentage or specific value like "500px"

add_bg_from_local(image_path, position_x, position_y, size_width, size_height)

# Main page content
#st.title("Main Page")
st.sidebar.success("Select a page from above.")

#streamlit run "/Users/bishalghosh/Desktop/Trimester 3/MIS41420-Sports/Project1/1_📊_Welcome_to_Cricket_Mania.py"
