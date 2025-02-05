import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to apply Canny edge detection and visualize the results
def canny_edge_detection(image_path, gaussian_sigma=1, low_threshold=25, high_threshold=100):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Step 1: Gaussian smoothing
    smoothed_image = cv2.GaussianBlur(image, (5, 5), gaussian_sigma)

    # Step 2: Compute gradients using Sobel operators
    Ix = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=3)

    # Step 3: Compute gradient magnitude and direction
    gradient_magnitude = np.sqrt(Ix**2 + Iy**2)
    gradient_direction = np.arctan2(Iy, Ix)

    # Step 4: Perform non-maximum suppression
    suppressed_magnitude = np.zeros_like(gradient_magnitude)
    height, width = gradient_magnitude.shape

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            angle = gradient_direction[i, j] * 180.0 / np.pi
            q, r = 255, 255

            # Angle 0
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = gradient_magnitude[i, j + 1]
                r = gradient_magnitude[i, j - 1]
            # Angle 45
            elif (22.5 <= angle < 67.5):
                q = gradient_magnitude[i + 1, j - 1]
                r = gradient_magnitude[i - 1, j + 1]
            # Angle 90
            elif (67.5 <= angle < 112.5):
                q = gradient_magnitude[i + 1, j]
                r = gradient_magnitude[i - 1, j]
            # Angle 135
            elif (112.5 <= angle < 157.5):
                q = gradient_magnitude[i - 1, j - 1]
                r = gradient_magnitude[i + 1, j + 1]

            if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                suppressed_magnitude[i, j] = gradient_magnitude[i, j]

    # double thresholding : create the edge image
    edges = np.zeros_like(suppressed_magnitude)
    strong_edge_i, strong_edge_j = np.where(suppressed_magnitude >= high_threshold)
    weak_edge_i, weak_edge_j = np.where((suppressed_magnitude >= low_threshold) & (suppressed_magnitude < high_threshold))
    edges[strong_edge_i, strong_edge_j] = 255
    edges[weak_edge_i, weak_edge_j] = 50  # For visualization

    # Use edge tracking by hysteresis to connect weak edges to strong edges
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if edges[i, j] == 50:
                if (edges[i - 1, j - 1] == 255 or edges[i - 1, j] == 255 or edges[i - 1, j + 1] == 255
                    or edges[i, j - 1] == 255 or edges[i, j + 1] == 255
                    or edges[i + 1, j - 1] == 255 or edges[i + 1, j] == 255 or edges[i + 1, j + 1] == 255):
                    edges[i, j] = 255

    # Visualize the results
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(2, 3, 2)
    plt.imshow(smoothed_image, cmap='gray')
    plt.title('Gaussian Smoothed Image')

    plt.subplot(2, 3, 3)
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title('Gradient Magnitude')

    plt.subplot(2, 3, 4)
    plt.imshow(suppressed_magnitude, cmap='gray')
    plt.title('Suppressed Image')

    plt.subplot(2, 3, 5)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Image')

    plt.tight_layout()
    plt.show()

# Test the Canny edge detector on the provided images
image_paths = ['./images/cube_left.pgm', './images/cube_right.pgm']
for image_path in image_paths:
    canny_edge_detection(image_path)