# ISP Pipeline 


### Launch Instructions

1. **Install Dependencies:** Ensure all required packages from `requirements.txt` are installed.

2. **Start the Application:** Run the application with the command:
   
   python app.py

3. **View the Project:** Open your web browser and navigate to http://127.0.0.1:5000/ to view the project.

# Image Processing Pipeline Documentation

This document provides a comprehensive guide on using the Image Processing Pipeline script, which is designed for processing raw images through various stages including linearization, white balancing, demosaicing, color space correction, gamma encoding, and compression.

## Overview

The script processes a raw `.tiff` image file through several steps to enhance its quality and prepare it for viewing or further processing. The steps include:

1. **Linearization**: Converts the image into a double-precision array and applies linearization based on predefined black and white points.
2. **Bayer Pattern Identification**: Identifies the Bayer pattern of the image to guide the demosaicing process.
3. **White Balancing**: Applies different white balancing techniques including grey world, camera presets, and manual white balancing.
4. **Demosaicing**: Converts the Bayer patterned image into a full-color image.
5. **Color Space Correction**: Applies color space correction to adjust the colors based on camera-specific matrices.
6. **Brightness Adjustment and Gamma Encoding**: Adjusts the brightness and applies gamma encoding to prepare the image for display.
7. **Compression**: Compresses the final image and saves it in different formats.

## Manual White Balancing

Once run, the code will prompt the user to select a space on the original image for white balancing to occur. 

Specifications:
1. Pick the most white square of the image.
2. Try and pick the largest area possible for the best results.
3. May need to try multiple areas of the image for best results.


## Prerequisites

Before running the script, ensure you have the following dependencies installed:

- Python 3.6 or higher
- NumPy
- SciPy
- scikit-image
- matplotlib
- Pillow
- colour_demosaicing
- Flask

You can install these dependencies using pip:

```bash
pip install -r requirements.txt


