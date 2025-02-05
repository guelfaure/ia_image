import numpy as np
from scipy.ndimage import sobel
from scipy.spatial import distance
from skimage import io
import matplotlib.pyplot as plt
def create_dense_grid(image_shape, n_points_x, n_points_y):
    # Step 1: Define a dense regular grid of points
    x_indices, y_indices = np.meshgrid(np.linspace(0, image_shape[1], n_points_x),
                                       np.linspace(0, image_shape[0], n_points_y))
    x_indices = x_indices.flatten().astype(int)  # Convert to integers
    y_indices = y_indices.flatten().astype(int)  # Convert to integers
    return x_indices, y_indices


def compute_hog_features(image, x_indices, y_indices):
    # Step 2: Compute HOG feature descriptors
    cell_height = 4
    cell_width = 4
    grid_size = 16  

    n_points = len(x_indices)
    hog_features = np.zeros((n_points, 128))  # Initialize HOG features array

    for i in range(n_points):
        x, y = x_indices[i], y_indices[i]
        # Convert x and y to integers (instead of numpy.int32)
        x = int(x)
        y = int(y)

        
        patch = image[y - grid_size // 2:y + grid_size // 2, x - grid_size // 2:x + grid_size // 2]
        
        gradient_x = sobel(patch, axis=0)
        gradient_y = sobel(patch, axis=1)
        
        
        gradient_direction = np.arctan2(gradient_y, gradient_x)
        gradient_direction[gradient_direction < 0] +=  np.pi  # Map to [-pi, pi]

        
        histograms = []
        for j in range(4):
            for k in range(4):
                cell_directions = gradient_direction[j * 4:(j + 1) * 4, k * 4:(k + 1) * 4]
                cell_histogram, _ = np.histogram(cell_directions, bins=8, range=(-np.pi, np.pi))
                histograms.append(cell_histogram)

        # Concatenate histograms to create the final 128-D HOG descriptor
        hog_features[i] = np.concatenate(histograms)

    return hog_features



image1 = io.imread('./img/1/1006.png')
image_shape = (96,96)
n_points_x = 10
n_points_y = 10
x_indices, y_indices = create_dense_grid(image_shape, n_points_x, n_points_y)
hog_descriptors = compute_hog_features(image1, x_indices, y_indices)
print(hog_descriptors.shape)

def k_means_clustering(hog_descriptors, K, max_iterations=100):
    num_samples, num_features = hog_descriptors.shape
    random_indices = np.random.permutation(num_samples)[:K]
    centroids = hog_descriptors[random_indices]

    for _ in range(max_iterations):
        distances = distance.cdist(hog_descriptors, centroids, 'euclidean')
        labels = np.argmin(distances, axis=1)
        
        new_centroids = []
        for k in range(K):
            if np.any(labels == k):
                new_centroids.append(hog_descriptors[labels == k].mean(axis=0))
            else:
                new_centroids.append(centroids[k])

        new_centroids = np.array(new_centroids)

        if np.array_equal(centroids, new_centroids): # Vérification de la convergence
            break
        centroids = new_centroids

    return centroids.reshape(-1, num_features)


def compute_bow_histograms(images_hog_descriptors, centroids):
    num_images = len(images_hog_descriptors)
    K = len(centroids)

    # Initialize an array to store the BoW histograms for all images
    bow_histograms = np.zeros((num_images, K))

    for i, hog_descriptors in enumerate(images_hog_descriptors):
        for j, centroid in enumerate(centroids):
            # Calculate the L2 (Euclidean) distance between hog_descriptors[i] and centroid[j]
            distance = np.linalg.norm(hog_descriptors - centroid)
            bow_histograms[i, j] = distance

    # for i in range(num_images):
    #     for j in range(K):
    #     # Calculate the L1 (Manhattan) distance between hog_descriptors[i] and centroid[j]
    #         distance = np.sum(np.abs(hog_descriptors[i] - centroid[j]))
    #         bow_histograms[i, j] = distance

    return bow_histograms


def nearest_neighbor_classifier(train_histograms, train_labels, test_histograms):
    num_test = test_histograms.shape[0]
    predicted_labels = np.zeros(num_test, dtype=int)

    for i in range(num_test):
        test_hist = test_histograms[i]
        distances = np.linalg.norm(train_histograms - test_hist, axis=1)  # Calculate L2 distances
        nearest_index = np.argmin(distances)
        predicted_labels[i] = train_labels[nearest_index]

    return predicted_labels



def random_classifier(num_classes, num_samples):
    return np.random.randint(0, num_classes, num_samples)