# Imports
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
from skimage import io
from skimage.color import rgb2gray
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
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
    flattened_image = image.reshape(-1, 3)

    for index in range(0, len(flattened_image)):
        if index % 4 == 0:
            flattened_image[index] *= r_scale  # Red channel
        elif index % 4 == 1 or index % 4 == 2:
            flattened_image[index] *= g_scale  # Green channel
        elif index % 4 == 3:
            flattened_image[index] *= b_scale  # Blue channel
    
    image_balanced = flattened_image.reshape(image.shape)
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
    return demosaicing_CFA_Bayer_bilinear(image, pattern)
       

# Print demosiac images
print(f"Demosaiced White Balanced Image:\n{demosaic_image(white_balanced_image, 'rggb')}")
print('\n')
print(f"Demosaiced Grey Balanced Image:\n{demosaic_image(grey_balanced_image, 'rggb')}")
print('\n')
print(f"Demosaiced Camera Presets Image:\n{demosaic_image(camera_presets_image, 'rggb')}")
print("###")


plt.imshow(demosaic_image(white_balanced_image, 'rggb'))
plt.show()



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
white_balanced_image_sRGB = colour_space_correction(demosaic_image(white_balanced_image, 'rggb'), camera_matrix)
grey_balanced_image_sRGB = colour_space_correction(demosaic_image(grey_balanced_image, 'rggb'), camera_matrix)
camera_balanced_image_sRGB = colour_space_correction(demosaic_image(camera_presets_image, 'rggb'), camera_matrix)
# plt.imshow(white_balanced_image_sRGB)
# plt.show()




print(f"White balanced image after color space image\n{white_balanced_image_sRGB}")
print('\n')
print(f"Grey balanced image after color space image\n{grey_balanced_image_sRGB}")
print('\n')
print(f"Camera image after color space image\n{camera_balanced_image_sRGB}")
print('\n')

### Brightness adjustment and gamma encoding ###


# Gamma Encoding
def gamma_encoding(image, mean = 0.25):
    greyscale = rgb2gray(image)
    mean_intensity = np.mean(greyscale)
    scaled_image = image * (mean / mean_intensity)
    clipped_image = np.clip(scaled_image, 0, 1)

    gamma_encoded_image = np.where(clipped_image <= .0031308, 12.92 * clipped_image, (1 + .055) * np.power(clipped_image, 1 / 2.4) - .055)

    gamma_encoded_image = np.dstack( [gamma_encoded_image[:, :, 0],
                                      gamma_encoded_image[:, :, 1],
                                      gamma_encoded_image[:, :, 2]])
    
    return gamma_encoded_image
    
gamma_encoded_white_balanced = gamma_encoding(white_balanced_image_sRGB)
gamma_encoded_grey_balanced = gamma_encoding(grey_balanced_image_sRGB)
gamma_encoded_camera_balanced = gamma_encoding(camera_balanced_image_sRGB)


### Compression ###

def compress_and_save_image(gamma_encoded_imgage, output_directory):

    png_file = os.path.join(output_directory, 'image.png')
    plt.imsave(png_file, gamma_encoded_imgage, cmap='gray')
    png_size = os.path.getsize(png_file)

    jpeg_file_q95 = os.path.join(output_directory, 'image_q95_highQuality.jpeg')
    plt.imsave(jpeg_file_q95, gamma_encoded_imgage, cmap='gray')
    jpeg_size_q95 = os.path.getsize(jpeg_file_q95)

    compression_ratio_q95 = png_size / jpeg_size_q95
    print(f"Compression ratio for JPEG: {compression_ratio_q95:.2f}")

    # Find the lowest JPEG quality setting with indistinguishable quality
    for quality in range(100, 0, -5):
        jpeg_file = os.path.join(output_directory, f'image_q{quality}.jpeg')
        plt.imsave(jpeg_file, gamma_encoded_imgage, cmap='grey')
        jpeg_size = os.path.getsize(jpeg_file)

        if jpeg_size < png_size:
            compression_ratio = png_size / jpeg_size
            print(f"Compression ratio for JPEG with quality={quality}: {compression_ratio:.2f}")
            break


    print("Inspect the compressed images to determine the lowest acceptable quality setting.")


compress_and_save_image(gamma_encoded_white_balanced, 'data/white_balanced_data')
compress_and_save_image(gamma_encoded_grey_balanced, 'data/grey_balanced_data')
compress_and_save_image(gamma_encoded_camera_balanced, 'data/camera_balanced_data')


### Manual White Balancing ###
def manual_white_balancing(image_path, white_patch_coords, brightness_scale = 0.8):
    """
    Perform manual white balancing on an image by normalizing RGB channels
    based on a selected white patch.

    Parameters:
    - image_path: Path to the image file.
    - white_patch_coords: Coordinates of the white patch (x, y, width, height).

    Returns:
    - Displays the original and white-balanced images.
    """
    # Load the image
    image = io.imread(image_path)
    
    # Convert the image to double-precision array
    image = image.astype(np.float64)
    
    # Normalize the image
    image = (image - image.min()) / (image.max() - image.min())
    
    # Extract the white patch
    x, y, width, height = white_patch_coords
    white_patch = image[y:y+height, x:x+width]
    
    # Calculate the mean values of the RGB channels in the white patch
    mean_values = np.mean(white_patch, axis=(0, 1))
    
    # Calculate scaling factors to normalize the RGB channels
    scale_factors = mean_values.max() / mean_values
    
    # Apply the scaling factors to the entire image
    white_balanced_image = (image * scale_factors).clip(0, 1)

    white_balanced_image = white_balanced_image * brightness_scale


    
    # # Display the original and white-balanced images
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(image)
    # ax[0].set_title('Original Image')
    # ax[0].axis('off')
    
    # ax[1].imshow(white_balanced_image)
    # ax[1].set_title('White-Balanced Image')
    # ax[1].axis('off')
    
    # plt.show()

    descriptive_name = os.path.basename(os.path.dirname(image_path))
    
    # Save the original image
    original_image_save_path = os.path.join('data/manual_white_balancing_data', f'{descriptive_name}_original.png')
    plt.imsave(original_image_save_path, image)
    
    # Save the white-balanced image
    white_balanced_image_save_path = os.path.join('data/manual_white_balancing_data', f'{descriptive_name}_white_balanced.png')
    plt.imsave(white_balanced_image_save_path, white_balanced_image)


def get_white_patch_coords(image_path):
    image = io.imread(image_path)
    
    # Convert the image to double-precision array
    image = image.astype(np.float64)
    
    # Normalize the image
    image = (image - image.min()) / (image.max() - image.min())
    
    # Display the image for white patch selection
    plt.imshow(image)
    plt.title('Click on two opposite corners of the white patch')
    plt.axis('on')
    
    # Use ginput to select the white patch
    points = plt.ginput(2)
    plt.close()
    
    # Calculate coordinates and size of the white patch
    x1, y1 = points[0]
    x2, y2 = points[1]
    x, y = min(x1, x2), min(y1, y2)
    width, height = abs(x2 - x1), abs(y2 - y1)
    white_patch_coords = (int(x), int(y), int(width), int(height))
    return white_patch_coords

#white_patch_coords = (3800, 100, 50, 50)  
white_patch_coords = get_white_patch_coords('data/baby.jpeg')
manual_white_balancing('data/white_balanced_data/image.png', white_patch_coords)
manual_white_balancing('data/grey_balanced_data/image.png', white_patch_coords)
manual_white_balancing('data/camera_balanced_data/image.png', white_patch_coords)











        
