document.getElementById('image-upload').addEventListener('change', handleImageUpload);
document.getElementById('process-image').addEventListener('click', processImage);
//document.getElementById('manual-white-balance').addEventListener('click', manualWhiteBalance);
document.getElementById('download-image').addEventListener('click', downloadImage);
document.getElementById('download-sample-image').addEventListener('click', downloadSampleImage);
document.getElementById('download-pdf-report').addEventListener('click', downloadPDFReport);

let uploadedImage = null;

function handleImageUpload(event) {
    hideError();
    const file = event.target.files[0];
    if (file && file.type !== 'image/tiff' && file.type !== 'image/x-tiff') {
        showError('Please upload a .tiff file.');
        return;
    }
    const reader = new FileReader();
    reader.onload = function(e) {
        uploadedImage = e.target.result;
        const img = new Image();
        img.src = uploadedImage;
        img.onload = function() {
            const canvas = document.getElementById('result-canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        }
    }
    reader.readAsDataURL(file);
}

async function processImage() {
    if (!uploadedImage) {
        showError('Please upload an image first.');
        return;
    }

    const whiteBalanceMethod = document.getElementById('white-balance-method').value;
    showProgressBar();

    try {
        const response = await fetch('/process-image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: uploadedImage,
                whiteBalanceMethod: whiteBalanceMethod,
            }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server Error: ${errorText}`);
        }

        const result = await response.json();
        if (result.error) {
            throw new Error(result.error);
        }

        const img = new Image();
        img.src = result.processedImage;
        img.onload = function() {
            const canvas = document.getElementById('result-canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        }
        
        hideError(); // Hide any previous errors
    } catch (error) {
        showError(error.message);
    } finally {
        hideProgressBar();
    }
}

// function manualWhiteBalance() {
//     alert('Manual white balance functionality is not implemented yet.');
// }

function downloadSampleImage() {
    const link = document.createElement('a');
    
    // Set the link href to the path of the sample image
    link.href = '/static/baby.tiff';
    
    // Set the download attribute with a filename
    link.download = 'baby.tiff';
    
    // Append the link to the document
    document.body.appendChild(link);
    
    // Programmatically click the link to trigger the download
    link.click();
    
    // Remove the link from the document
    document.body.removeChild(link);
}

function downloadPDFReport() {
    const link = document.createElement('a');
    link.href = '/static/ISP_Report.pdf';
    link.download = 'ISP_Report.pdf';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function downloadImage() {
    const canvas = document.getElementById('result-canvas');
    const link = document.createElement('a');
    link.download = 'processed_image.png';
    link.href = canvas.toDataURL();
    link.click();
}

function showError(message) {
    const errorDiv = document.getElementById('error-message');

    if (message.includes('Demosaicing failed: filter weights array has incorrect shape.')) {
        message = 'Demosaicing failed: The uploaded image may not be in the expected TIFF format.';
    }

    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}

function hideError() {
    const errorDiv = document.getElementById('error-message');
    errorDiv.style.display = 'none';
}

function showProgressBar() {
    const progressBarContainer = document.getElementById('progress-bar-container');
    const progressBar = document.getElementById('progress-bar');
    progressBar.style.width = '0%';
    progressBarContainer.style.display = 'block';

    // Simulate progress gradually to indicate that work is ongoing
    let progress = 0;
    const interval = setInterval(() => {
        if (progress >= 90) {
            clearInterval(interval);
        } else {
            progress += 10;
            progressBar.style.width = progress + '%';
        }
    }, 500);
}

function hideProgressBar() {
    const progressBarContainer = document.getElementById('progress-bar-container');
    const progressBar = document.getElementById('progress-bar');
    progressBar.style.width = '100%'; // Ensure it reaches 100% before hiding
    setTimeout(() => {
        progressBarContainer.style.display = 'none';
    }, 500); // Delay hiding to show the complete progress
}
