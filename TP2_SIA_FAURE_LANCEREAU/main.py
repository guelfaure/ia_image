import bow_functions
import images_extraction
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split




# Number of runs and clusters
K=100
num_runs = 10
# Perform multiple runs
accuracies = []
random_accuracies = []

i=0
for run in range(num_runs):
    # Simulate data or load your data here

    # #only two clases:
    # #Load images and labels for class 1
    # class_1_images, class_1_labels = images_extraction.load_images_from_directory("./img/1", class_label=1)

    # # Load images and labels for class 2
    # class_2_images, class_2_labels = images_extraction.load_images_from_directory("./img/2", class_label=2)

    # # Combine data from both classes
    # all_images = class_1_images + class_2_images
    # all_labels = class_1_labels + class_2_labels

    # # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

    # ## Convert data to numpy arrays
    # X_train = np.array(X_train)
    # X_test = np.array(X_test)
    # y_train = np.array(y_train)[:100]
    # y_test = np.array(y_test)[:100]


    #full data
    X_train = images_extraction.read_all_images(images_extraction.DATA_PATH)
    y_train = images_extraction.read_labels(images_extraction.LABEL_PATH)
    X_test= images_extraction.read_all_images(images_extraction.DATA_PATH_test)[:100]
    y_test = images_extraction.read_labels(images_extraction.LABEL_PATH_test)[:100]

    # K-means clustering on the training set
    n_points_x = 10
    n_points_y = 10
    image_shape = (96, 96)
    x_indices, y_indices = bow_functions.create_dense_grid(image_shape, n_points_x, n_points_y)
    hog_descriptors = bow_functions.compute_hog_features(X_train, x_indices, y_indices)
    hog_descriptors = hog_descriptors.reshape(hog_descriptors.shape[0], -1)
    centroids = bow_functions.k_means_clustering(hog_descriptors, K, max_iterations=10)


    # Calculate BoW histograms for training and test sets
    X_train_hog_descriptors = bow_functions.compute_hog_features(X_train, x_indices, y_indices)
    X_test_hog_descriptors = bow_functions.compute_hog_features(X_test, x_indices, y_indices)
    train_bow = bow_functions.compute_bow_histograms(X_train_hog_descriptors, centroids)
    test_bow = bow_functions.compute_bow_histograms(X_test_hog_descriptors, centroids)

    # Nearest neighbor classifier
    predicted_labels = bow_functions.nearest_neighbor_classifier(train_bow, y_train, test_bow)

    # Calculate accuracy for this run
    accuracy = accuracy_score(y_test, predicted_labels)
    accuracies.append(accuracy)

    num_classes = len(np.unique(y_test))
    num_samples = len(y_test)
    random_predictions = bow_functions.random_classifier(num_classes, num_samples)
    random_accuracy = accuracy_score(y_test, random_predictions)
    random_accuracies.append(random_accuracy)
    i+=1
    print(i)

# Report results
mean_accuracy = np.mean(accuracies)
mean_random_accuracy = np.mean(random_accuracies)
print(f"Mean Accuracy: {mean_accuracy:.2f}")
print(f"Random Classifier Mean Accuracy: {mean_random_accuracy:.2f}")



##go in bow_functions and comments the L2 norm, and uncomment the L1 norm if you want to compare