from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-image', methods=['POST'])
def process_image():
    data = request.get_json()
    image_data = data['image']
    white_balance_method = data['whiteBalanceMethod']
    
    # Decode the image
    try:
        image_data = base64.b64decode(image_data.split(',')[1])
        image = Image.open(BytesIO(image_data))
        
        # Check if the image is a TIFF
        if image.format != 'TIFF':
            return jsonify({'error': 'Invalid file type. Only .tiff files are allowed'}), 400
        
        image = np.array(image)
    except Exception as e:
        return jsonify({'error': f'Error decoding image: {str(e)}'}), 400
    
    # Apply the image processing pipeline
    try:
        processed_image = apply_pipeline(image, white_balance_method)
        processed_image = np.clip(processed_image, 0, 1)
    except (RuntimeError, ValueError) as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 400
    
    # Encode the processed image to base64
    buffered = BytesIO()
    plt.imsave(buffered, processed_image, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return jsonify({'processedImage': f'data:image/png;base64,{img_str}'})

def apply_pipeline(image, white_balance_method):
    # Apply white balance method to the image
    if white_balance_method == 'white':
        processed_image = white_bal(image)
    elif white_balance_method == 'grey':
        processed_image = grey_bal(image)
    elif white_balance_method == 'camera':
        # Define r_scale, g_scale, b_scale
        r_scale = 1.628906
        g_scale = 1.000000
        b_scale = 1.386719
        processed_image = camera_presets(image, r_scale, g_scale, b_scale)
    else:
        raise ValueError('Invalid white balance method specified.')
    
    # Apply demosaicing (if needed)
    try:
        processed_image = demosaicing_CFA_Bayer_bilinear(processed_image, 'RGGB')
    except RuntimeError as e:
        raise RuntimeError(f'Demosaicing failed: {str(e)}')
    
    return processed_image

def white_bal(image):
    return image / np.mean(image, axis=(0, 1))

def grey_bal(image):
    return image / np.mean(image)

def camera_presets(image, r_scale, g_scale, b_scale):
    # Convert image to float64 for processing
    image = image.astype(np.float64)
    
    # Flatten the image to manipulate the pixels directly
    flattened_image = image.flatten()

    # Apply the scaling factors to each color channel
    for index in range(len(flattened_image)):
        if index % 3 == 0:
            flattened_image[index] *= r_scale  # Red channel
        elif index % 3 == 1:
            flattened_image[index] *= g_scale  # Green channel
        else:
            flattened_image[index] *= b_scale  # Blue channel

    # Reshape the flattened image back to its original shape
    processed_image = flattened_image.reshape(image.shape)
    
    # Convert back to uint16
    processed_image = np.clip(processed_image, 0, 65535).astype(np.uint16)
    return processed_image

if __name__ == '__main__':
    app.run(debug=True)
