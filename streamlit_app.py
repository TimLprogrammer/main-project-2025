"""
Weather Forecast & Model Management App
Main entry point for Streamlit Cloud
"""

import os
import sys

# Add the streamlit directory to the path so we can import the main module
streamlit_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit")
sys.path.append(streamlit_dir)

# Import the main module
import main

# The main module will be executed automatically when imported
