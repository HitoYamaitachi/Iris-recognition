from flask import Flask, request, render_template
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the request contains a file
        if 'file' not in request.files:
            return 'No file part', 400
        file = request.files['file']

        # Save the uploaded file
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the image and get the results
        outer_iris, inner_iris = process_iris_image(file_path)

        # Pass the results to the template
        return render_template(
            'index.html',
            outer_iris=outer_iris,
            inner_iris=inner_iris,
            image_path=filename
        )

    # Render the initial template
    return render_template('index.html')

def process_iris_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Apply median blur to reduce noise
    blurred = cv2.medianBlur(enhanced_gray, 5)

    # Detect the outer iris using edge detection and ellipse fitting
    edges = cv2.Canny(blurred, 50, 150)  # Adjusted thresholds
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)  # Fill gaps

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    outer_iris = None
    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            # Relaxed criteria for ellipse fitting
            if 50 < ellipse[1][0] < 500 and 50 < ellipse[1][1] < 500:
                outer_iris = {
                    'center_x': int(ellipse[0][0]),
                    'center_y': int(ellipse[0][1]),
                    'major_axis': int(ellipse[1][0] / 2),
                    'minor_axis': int(ellipse[1][1] / 2),
                    'angle': float(ellipse[2])
                }
                break

    # Detect the inner iris using circle detection
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=50
    )
    inner_iris = None
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            inner_iris = {
                'center_x': int(i[0]),
                'center_y': int(i[1]),
                'radius': int(i[2])
            }
            break

    return outer_iris, inner_iris

if __name__ == '__main__':
    app.run(debug=False)
