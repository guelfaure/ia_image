import cv2
import numpy as np
import random
video = cv2.VideoCapture('./video_sequences/synthetic/escrime-4-3.avi')

frames = []
while video.isOpened():
  ret, frame = video.read()
  if not ret:
    break
  frames.append(frame)
frames = frames[:27]

###Step 1
# Get the first frame of the video
# success, frame = video.read()

first_frame = frames[0].copy()
num_objects = 2
# Find the center and size of the rectangle
# Initialize lists to store selected ROIs
selected_rois = []
selected_centers = []
selected_sizes = []

for i in range(num_objects):
    # Select ROI for each object on the first frame
    cv2.imshow("Select ROI for Object {}".format(i + 1), first_frame)
    rect = cv2.selectROI("Select ROI for Object {}".format(i + 1), first_frame, False)
    selected_rois.append(rect)
    x, y, w, h = rect
    center_x = int(x + w / 2)
    center_y = int(y + h / 2)
    selected_centers.append((center_x, center_y))
    selected_sizes.append((w, h))
    cv2.destroyWindow("Select ROI for Object {}".format(i + 1))


###Step2
def calculate_histogram(image):
    # Calculate the histogram for each channel (R, G, B) separately
    hist_channels = [cv2.calcHist([image], [i], None, [256], [0, 256]) for i in range(3)]

    # Normalize each channel's histogram
    hist_normalized = [hist / np.sum(np.abs(hist)) for hist in hist_channels]
    return hist_normalized


ref_histograms = []

# Calculate and store reference histograms for each selected ROI for each channel
for roi in selected_rois:
    # Extract ROI for each channel (R, G, B)
    roi_frame = frames[0][roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    ref_hist = calculate_histogram(roi_frame)
    ref_histograms.append(ref_hist)

###Step3
def initialize_particles(num_objects, num_particles, centers, sizes):
    particles = []
    for i in range(num_objects):
        object_particles = []
        center_x, center_y = centers[i]
        w, h = sizes[i]
        for _ in range(num_particles):
            x = random.randint(center_x - w // 2, center_x + w // 2)
            y = random.randint(center_y - h // 2, center_y + h // 2)
            particle = [x, y, 1 / num_particles]
            object_particles.append(particle)
        particles.append(object_particles)
    return particles

def particle_rectangle_function(particle,w,h,i):
    particle_rectangle = frames[i][int(particle[1]):int(particle[1]+h),  
                      int(particle[0]):int(particle[0]+w)]
    return particle_rectangle

###Step4
#prediction
def transition_model(sigma_x, sigma_y, particles):
    new_particles = particles
    noise_mean = [0, 0]
    noise_covariance = np.diag([sigma_x, sigma_y])
    
    for elmt in new_particles:
        noise = np.random.multivariate_normal(noise_mean, noise_covariance)
        elmt[0] +=noise[0]
        elmt[1] +=noise[1]
        
    return new_particles




#correction
def calculate_distance(hist1, hist2):
    distance = np.sqrt(1-(np.sum(np.sqrt(np.multiply(hist1, hist2)))))
    return distance

def calculate_likelihood(reference_hist, particle_hist, lambda_value):
    distance = calculate_distance(reference_hist, particle_hist)
    likelihood = np.exp(-lambda_value * np.square(distance))
    return likelihood

def update_weights(particles, reference_hist, lambda_value, i):
    total_weight = 0.0

    for particle in particles:
        particle_hist = calculate_histogram(particle_rectangle_function(particle, w, h, i))
        likelihood = 1.0

        # Calculate likelihood for each color channel separately
        for channel in range(3):
            likelihood *= calculate_likelihood(reference_hist[channel], particle_hist[channel], lambda_value)
        particle[2] = likelihood
        total_weight += likelihood

    for particle in particles:
        particle[2] /= total_weight  # Normalize the weights

    return particles






### Step 5: Resampling


def resampling(particles):
    N = len(particles)
    new_particles = []
    weights = [particle[2] for particle in particles]

    # Normalize weights to ensure their sum is 1
    total_weight = sum(weights)
    if total_weight == 0:
        return particles  # No resampling if all weights are zero

    normalized_weights = [w / total_weight for w in weights]

    # Create an array for cumulative sum of normalized weights
    cumulative_weights = np.cumsum(normalized_weights)

    # Generate N equidistant points in [0,1/N] range
    u0 = random.uniform(0, 1 / N)
    u = [u0 + (i / N) for i in range(N)]

    j = 0
    for i in range(N):
        while u[i] > cumulative_weights[j]:
            j += 1
        new_particles.append(particles[j][:])  # Clone the particle at index j

    return new_particles




#Step6 estimation position tracked
def position_tracked_function(particles):    
    weighted_sum_x = sum(p[0] * p[2] for p in particles)
    weighted_sum_y = sum(p[1] * p[2] for p in particles)
    return [weighted_sum_x, weighted_sum_y]


### Step 7: Tracking Loop
sigma_x = [5, 1.8]
sigma_y = [5, 1.8]
lambda_value = [3, 3]

particles = initialize_particles(num_objects, 70, selected_centers, selected_sizes)
last_frame_index = len(frames) - 1
for i in range(0, len(frames)):
    print(i)
    # Prediction
   

    # Correction, Systematic Resampling
    for j in range(num_objects):
        particles[j] = transition_model(sigma_x[j], sigma_y[j], particles[j])
        updated_particles = update_weights(particles[j], ref_histograms[j], lambda_value[j], i)
        particles[j] = resampling(updated_particles)
        
    # Display the tracked objects and particles for each frame
    for j in range(num_objects):
        position_tracked = position_tracked_function(particles[j])
        print(f"Tracked Position Object {j + 1}:", position_tracked)

        top_left = (int(position_tracked[0] - selected_sizes[j][0] / 2),
                    int(position_tracked[1] - selected_sizes[j][1] / 2))
        bottom_right = (int(position_tracked[0] + selected_sizes[j][0] / 2),
                        int(position_tracked[1] + selected_sizes[j][1] / 2))

        for particle in particles[j]:
            x, y = int(particle[0]), int(particle[1])
            cv2.circle(frames[i], (x, y), 3, (0, 100*int(j), 255), -1)
        cv2.rectangle(frames[i], top_left, bottom_right, (255*int(j), 255-255*int(j), 0), 2)

    if i == last_frame_index:
        cv2.imshow("Tracked Objects - Last Frame", frames[i])
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    else:
        cv2.imshow("Tracked Objects", frames[i])
        cv2.waitKey(100) 


# After the loop ends, display the last frame
cv2.imshow("Tracked Object - Last Frame", frames[last_frame_index])
cv2.waitKey(0)  # Wait indefinitely until a key is pressed



