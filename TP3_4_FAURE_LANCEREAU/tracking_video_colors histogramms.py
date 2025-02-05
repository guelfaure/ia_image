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


###Step 1
# Get the first frame of the video
# success, frame = video.read()

# Find the center and size of the rectangle
x, y, w, h = cv2.selectROI(frames[0], False)

# Calculate the position of the center
center_x = int(x + w / 2)
center_y = int(y + h / 2)

# Print the result
print("Size of rectangle: (width={}, height={})".format(w, h))
print("Center of rectangle: ({}, {})".format(center_x, center_y))


###Step2
cropped_image = frames[0][int(y):int(y+h),  
                      int(x):int(x+w)] 
cv2.imshow("Cropped image", cropped_image) 
cv2.waitKey(0) 

def calculate_histogram(image):
    # Split the image into its BGR components
    b, g, r = cv2.split(image)

    # Calculate histograms for each channel
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256]) / np.sum(np.abs(cv2.calcHist([b], [0], None, [256], [0, 256])))
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256]) / np.sum(np.abs(cv2.calcHist([g], [0], None, [256], [0, 256])))
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256]) / np.sum(np.abs(cv2.calcHist([r], [0], None, [256], [0, 256])))

    return hist_b, hist_g, hist_r

ref_hist_b, ref_hist_g, ref_hist_r = calculate_histogram(cropped_image)


###Step3
def initialize_particles(N, center_x, center_y, w, h):
	particles = []
	for _ in range(N):
		x = random.randint(center_x - w//2, center_x + w//2)
		y = random.randint(center_y - h//2, center_y + h//2)
		particle = [x, y, 1/N]
		particles.append(particle)
	return particles

init_particles = initialize_particles(100, center_x, center_y, w, h)

def particle_rectangle_function(particle, w, h, i):
    particle_rectangle = frames[i][int(particle[1]):int(particle[1] + h),
                                    int(particle[0]):int(particle[0] + w)]
    return calculate_histogram(particle_rectangle)

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

def update_weights(particles, reference_hist_b, reference_hist_g, reference_hist_r, lambda_value, i):
    total_weight = 0.0

    for particle in particles:
        hist_b, hist_g, hist_r = particle_rectangle_function(particle, w, h, i)
        likelihood_b = calculate_likelihood(reference_hist_b, hist_b, lambda_value)
        likelihood_g = calculate_likelihood(reference_hist_g, hist_g, lambda_value)
        likelihood_r = calculate_likelihood(reference_hist_r, hist_r, lambda_value)
        particle[2] = likelihood_b * likelihood_g * likelihood_r
        total_weight += particle[2]

    for particle in particles:
        particle[2] /= total_weight  # Normalize the weights

    return particles






### Step 5: Resampling


def resampling(particles):
    N = len(particles)
    new_particles = []
    weights = [particle[2] for particle in particles]

    # Create an array for cumulative sum of weights
    cumulative_weights = np.cumsum(weights)

    # Generate N equidistant points in [0,1/N) range
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
sigma_x = 4
sigma_y = 4
lambda_value = 3

particles = init_particles
last_frame_index = len(frames) - 1
for i in range(0, len(frames)):
    print(i)
    # Prediction
    particles = transition_model(sigma_x, sigma_y, particles)

    # Correction
    updated_particles = update_weights(particles, ref_hist_b, ref_hist_g, ref_hist_r, lambda_value, i)
    # Systematic Resampling
    particles = resampling(updated_particles)

    # Display the tracked object and particles for each frame
    position_tracked = position_tracked_function(particles)
    print("Tracked Position:", position_tracked)
    
    top_left = (int(position_tracked[0] - w / 2), int(position_tracked[1] - h / 2))
    bottom_right = (int(position_tracked[0] + w / 2), int(position_tracked[1] + h / 2))
    
    for particle in particles:
        x, y = int(particle[0]), int(particle[1])
        cv2.circle(frames[i], (x, y), 3, (0, 0, 255), -1)  # draw a particle as a red circle !!it's the top left corner of the square of each particle!!
    cv2.rectangle(frames[i], top_left, bottom_right, (0, 255, 0), 2)
    if i == last_frame_index:
        cv2.imshow("Tracked Object - Last Frame", frames[i])
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed

    else:
        cv2.imshow("Tracked Object", frames[i])
        cv2.waitKey(100)  # Add a delay to display the frames

# After the loop ends, display the last frame
cv2.imshow("Tracked Object - Last Frame", frames[last_frame_index])
cv2.waitKey(0)  # Wait indefinitely until a key is pressed



