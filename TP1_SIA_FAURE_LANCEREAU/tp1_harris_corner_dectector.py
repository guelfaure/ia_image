import matplotlib.pyplot as plt
import matplotlib.image as img
from scipy.ndimage import gaussian_filter1d, gaussian_filter, convolve
from scipy.ndimage import rotate, zoom, shift
from skimage import color
import cv2
import numpy as np



def harris_corner_detector(image):
    if len(image.shape) == 3:
        image = color.rgb2gray(image)

    # Define Harris corner detection parameters
    block_size = 2
    ksize = 3
    k = 0.04
    threshold = 0.01

    # Step 1: Compute approximations of Ix and Iy (seems to work better with Sobel masks than Gaussian filter)
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

    # Step 2: Compute products of derivatives
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    # Step 3: Compute smooth versions of Ix² Iy² and IxIy: Apply Gaussian filtering to the products
    Ix2 = gaussian_filter(Ix2, sigma=1)
    Iy2 = gaussian_filter(Iy2, sigma=1)
    Ixy = gaussian_filter(Ixy, sigma=1)

    # Step 4: Compute the Harris corner response
    detM = Ix2 * Iy2 - Ixy ** 2
    traceM = Ix2 + Iy2
    harris_response = detM - k * (traceM ** 2)

    # Step 5: Apply non-maximum suppression
    harris_response = np.float32(harris_response)
    corner_pixels = cv2.cornerHarris(harris_response, blockSize=block_size, ksize=ksize, k=k)
    corner_threshold = threshold * corner_pixels.max()
    corner_pixels = np.where(corner_pixels > corner_threshold)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('original image')
    # Visualize Harris response
    plt.subplot(1, 3, 2)
    plt.imshow(harris_response, cmap='gray')
    plt.title('Harris Response')

    # Visualize the detected corners
    plt.subplot(1, 3, 3)
    image_with_corners = np.copy(image)
    image_with_corners[corner_pixels] = 255  # Highlight corners in white
    plt.imshow(image_with_corners, cmap='gray')
    plt.title('Detected Corners')

    plt.show()

    return harris_response, corner_pixels


# Load the images
image1 = cv2.imread('./images/CircleLineRect.png', cv2.COLOR_BGR2GRAY)
image2 = cv2.imread('./images/zurlim.png',  cv2.COLOR_BGR2GRAY)

# Apply the Harris corner detector
response1, corners1 = harris_corner_detector(image1)
response2, corners2 = harris_corner_detector(image2)

# Rotate the images
rotated_image1 = rotate(image1, angle=45, reshape=False)
rotated_image2 = rotate(image2, angle=30, reshape=False)

# Apply the Harris corner detector to the rotated images
response_rotated1, corners_rotated1 = harris_corner_detector(rotated_image1)
response_rotated2, corners_rotated2 = harris_corner_detector(rotated_image2)



# # Translate the images (perform translation), not working with  cv2.IMREAD_GRAYSCALE
# image11 = cv2.imread('./images/CircleLineRect.png', cv2.IMREAD_GRAYSCALE)
# translated_image1 = shift(image11, (10, 20))

# # Apply the Harris corner detector to the translated images
# response_translated1, corners_translated1 = harris_corner_detector(translated_image1)
