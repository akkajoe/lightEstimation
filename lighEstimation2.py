import cv2
import numpy as np
from scipy.special import sph_harm

# Load and preprocess the image
image_path = r"C:\Users\anush\GitHubRepos\lightEstimation\semi_transparent (1)\f96c65bd0351c92036c72e48eb7a8576.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("The image could not be loaded. Check the file path.")

# Convert to grayscale and normalize intensity
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)

# Mask and reduce pure white backgrounds
background_mask = cv2.inRange(gray_image, 246, 255)
gray_image[background_mask > 0] = 245

# K-means clustering to segment semi-transparent and opaque regions
pixel_values = gray_image.flatten().reshape((-1, 1)).astype(np.float32)
k = 5  # Number of clusters
_, labels, centers = cv2.kmeans(pixel_values, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 10, cv2.KMEANS_RANDOM_CENTERS)
labels_image = labels.reshape(gray_image.shape)

# Identify semi-transparent and opaque masks
brightest_cluster = np.argmax(centers)
semi_transparent_mask = (labels_image == brightest_cluster).astype(np.uint8) * 255
opaque_mask = (labels_image != brightest_cluster).astype(np.uint8) * 255

# Calculate gradients
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
gradient_direction = np.arctan2(sobel_y, sobel_x)

# Extract intensity and gradient directions for semi-transparent regions
semi_transparent_indices = np.where(semi_transparent_mask > 0)
semi_transparent_intensities = gray_image[semi_transparent_indices]
semi_transparent_directions = gradient_direction[semi_transparent_indices]

# Extract intensity and gradient directions for opaque regions
opaque_indices = np.where(opaque_mask > 0)
opaque_intensities = gray_image[opaque_indices]
opaque_directions = gradient_direction[opaque_indices]

# Spherical harmonic fitting function
def fit_spherical_harmonics(directions, intensities, order=1):
    """
    Fit spherical harmonics to intensity and gradient direction data.
    """
    coefficients = []
    for l in range(order + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, directions[:, 1], directions[:, 0])
            c_lm = np.dot(Y_lm, intensities)
            coefficients.append(c_lm)
    return coefficients

# Dominant light direction calculation
def dominant_light_direction(coefficients):
    """
    Calculate the dominant light direction from first-order spherical harmonics.
    """
    c_x, c_y, c_z = coefficients[1:4]  # Assuming order 1
    direction = np.array([c_x.real, c_y.real, c_z.real])
    return direction / np.linalg.norm(direction)

# Fit spherical harmonics and estimate light direction for semi-transparent regions
semi_transparent_coefficients = fit_spherical_harmonics(semi_transparent_directions, semi_transparent_intensities)
semi_transparent_direction = dominant_light_direction(semi_transparent_coefficients)

# Fit spherical harmonics and estimate light direction for opaque regions
opaque_coefficients = fit_spherical_harmonics(opaque_directions, opaque_intensities)
opaque_direction = dominant_light_direction(opaque_coefficients)

# Calculate the angular difference between the light directions
angle_difference = np.arccos(np.clip(np.dot(semi_transparent_direction, opaque_direction), -1, 1))
print(f"Angular difference between semi-transparent and opaque regions: {np.degrees(angle_difference):.2f} degrees")

# Visualize the dominant light direction
output_image = image.copy()
start_point = (output_image.shape[1] // 2, output_image.shape[0] // 2)
arrow_length = 100

# Draw arrow for semi-transparent light direction
end_point_semi_transparent = (
    int(start_point[0] + arrow_length * semi_transparent_direction[0]),
    int(start_point[1] + arrow_length * semi_transparent_direction[1]),
)
cv2.arrowedLine(output_image, start_point, end_point_semi_transparent, (255, 0, 0), 3, tipLength=0.3)

# Draw arrow for opaque light direction
end_point_opaque = (
    int(start_point[0] + arrow_length * opaque_direction[0]),
    int(start_point[1] + arrow_length * opaque_direction[1]),
)
cv2.arrowedLine(output_image, start_point, end_point_opaque, (0, 255, 0), 3, tipLength=0.3)

# Display results
cv2.imshow("Estimated Light Direction", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
