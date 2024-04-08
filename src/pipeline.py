# Imports
from skimage import io
import numpy as np
from scipy.interpolate import RegularGridInterpolator



#####################
# CODE #
####################

# Assining tiff_image to a variable
tiff_image = io.imread('data/baby.tiff')

# Properties
# Calculating the bit depth
image_depth = tiff_image.dtype.itemsize * 8

# Get width and height
width = tiff_image.shape[1]
height = tiff_image.shape[0]

# Print Values
print(f"Width: {width}")
print(f"Height: {height}")
print(f"Bit Depth: {image_depth} bits per pixel")
print("###")

# Converting function into double-precision array
double_precision_array = tiff_image.astype(np.float64)

# Print double-precision array
print(f"Image converted to double-precision array: {double_precision_array.dtype}")
print("###")

### Linearization ###
black = 0
white = 16383

# Apply Linearization
linearized_image = (double_precision_array - black) / (white - black)

# Create bounds by clipping values
linearized_image = np.clip(linearized_image, 0, 1)

# Print values
print(f"Image after being clipped and linearized:\n{linearized_image}")
print("###")

### Bayer Pattern ###

# Creating function to identify bayer pattern
def get_bayer_pattern(image):
    # Getting top-left 2 x 2
    top_left = image[:2, :2]

    # Getting mean values
    mean_values = np.mean(top_left, axis= (0, 1))

    # Analyize all possible bayer patterns and choose the closest match
    bayer_patterns = ['grbg', 'rggb', 'bggr', 'gbrg']
    closest_bayer_patterns = np.argmin(np.abs(np.array(mean_values) - [128, 128, 128]))

    # Return value
    return bayer_patterns[closest_bayer_patterns]

# Get bayer pattern
bayer_pattern = get_bayer_pattern(linearized_image)

# Print Value
print(f"Identified Bayer Pattern: {bayer_pattern}")
print("###")

### White Balancing ###
# Values from dcraw
r_scale = 1.628906
g_scale = 1.000000
b_scale = 1.386719


# Functions for balancing image
def grey_bal(image):
    return image / np.mean(image)

def white_bal(image):
    return image / np.mean(image, axis=(0, 1))

# Function to create campera presets
def camera_presets(image, r_scale, g_scale, b_scale):
    # Balance each individual channel
    r_scale_balanced = image[:, 0] / r_scale

    g_scale_balanced = image[:, 1] / g_scale

    b_scale_balanced = image[:, 2] / b_scale
    
    # Combine each channel into singular image and return
    image_balanced = np.stack([r_scale_balanced, g_scale_balanced, b_scale_balanced], axis= -1)
    return image_balanced

# Using above fuctions to white balance
white_balanced_image = white_bal(linearized_image)
grey_balanced_image = grey_bal(linearized_image)
camera_presets_image = camera_presets(linearized_image, r_scale, g_scale, b_scale)

# Print Values
print("Best white balancing algorithm: PLACE HOLDER{}")
print("###")

### Demosaicing ###

def demosaic_image(image, pattern):
    # Identify pattern based on above bayer pattern
        # grbg, rggb, bggr, gbrg
    if pattern == 'grbg':
       r_index = (0,1)
       g_index = (0,0)
       b_index = (1,1)
    elif pattern == 'rggb':
       r_index = (0,0)
       g_index = (0,1)
       b_index = (1,0)
    elif pattern == 'bggr':
       r_index = (1,1)
       g_index = (1,0)
       b_index = (0,1)
    elif pattern == 'gbrg':
       r_index = (1,0)
       g_index = (1,1)
       b_index = (0,0)
    else:
       raise ValueError("Invalid Pattern")
    

     # Interpolate each channel
    def interpolate_channel(image, index):
        # Extract the channel data based on the Bayer pattern
        channel_data = image[index[0]::2, index[1]::2]

        # Create an interpolator
        y = np.arange(0, channel_data.shape[0])
        x = np.arange(0, channel_data.shape[1])
        interpolator = RegularGridInterpolator((y, x), channel_data, method='linear', bounds_error=False, fill_value=None)

        # Prepare the full grid
        full_x = np.arange(image.shape[1])
        full_y = np.arange(image.shape[0])
        full_xx, full_yy = np.meshgrid(full_x, full_y)

        # Interpolate
        return interpolator((full_yy.flatten(), full_xx.flatten())).reshape(image.shape[0], image.shape[1])

    # Interpolate R, G, and B channels
    r_channel = interpolate_channel(image, r_index)
    g_channel = interpolate_channel(image, g_index)
    b_channel = interpolate_channel(image, b_index)

    # Stack the channels back together
    demosaiced_image = np.stack((r_channel, g_channel, b_channel), axis=-1)

    return demosaiced_image   
       
# Print demosiac images
print(f"Demosaiced White Balanced Image:\n{demosaic_image(white_balanced_image, 'grbg')}")
print('\n')
print(f"Demosaiced Grey Balanced Image:\n{demosaic_image(grey_balanced_image, 'grbg')}")
print('\n')
print(f"Demosaiced Camera Presets Image:\n{demosaic_image(camera_presets_image, 'grbg')}")
print("###")

### Colour Space Correction ###
def colour_space_correction(image, mxyz_to_camera):
    # Transform array to xyz image
    MsRGB_to_XYZ = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])

    M_XYZ_To_Camera = mxyz_to_camera / 10000.0

    M_XYZ_To_Camera = M_XYZ_To_Camera.reshape(3,3)

    # Compute sRGB to camera specific colour
    MsRGB_to_cam = np.dot(M_XYZ_To_Camera, MsRGB_to_XYZ)

    # Normalize matrix
    MsRGB_to_cam /= MsRGB_to_cam.sum(axis=1)[:, np.newaxis]

    # Compute Inverse
    M_cam_to_sRGB = np.linalg.inv(MsRGB_to_cam)

    # Apply colour correction
    corrected_image = np.dot(image.reshape(-1, 3), M_cam_to_sRGB.T).reshape(image.shape)

    # Clip values to appropiate values
    corrected_image = np.clip(corrected_image, 0, 1)

    return corrected_image

# Camera specific matrix found from dcraw source code for specific camera model: Nikon D3400
# {6988, -1384, -714, -5631, 13410, 2447, -1485, 2204, 7318}
camera_matrix = np.array([6988, -1384, -714, -5631, 13410, 2447, -1485, 2204, 7318])
white_balanced_image_sRGB = colour_space_correction(demosaic_image(white_balanced_image, 'grbg'), camera_matrix)
grey_balanced_image_sRGB = colour_space_correction(demosaic_image(grey_balanced_image, 'grbg'), camera_matrix)
camera_balanced_image_sRGB = colour_space_correction(demosaic_image(camera_presets_image, 'grbg'), camera_matrix)




print(f"White balanced image after color space image\n{white_balanced_image_sRGB}")
print('\n')
print(f"Grey balanced image after color space image\n{grey_balanced_image_sRGB}")
print('\n')
print(f"Camera image after color space image\n{camera_balanced_image_sRGB}")
print('\n')



        
