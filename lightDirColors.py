import cv2
import numpy as np
from scipy.optimize import least_squares
from sklearn.cluster import KMeans

image_path = "semi_transparent (1)/Images/le-gouter-1880.jpg"
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
output_image = image.copy()
output_image_dom_dir = image.copy()

# Apply Gaussian blur and Canny edge detection
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
canny_edges = cv2.Canny(blurred_image, 110, 200)
contours, _ = cv2.findContours(canny_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def compute_normals_from_tangents(contour):
    normals = []
    for i in range(1, len(contour) - 1):  # Avoid boundary points
        p_prev = contour[i - 1][0]
        p_next = contour[i + 1][0]
        tangent = p_next - p_prev
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 0:
            normal = np.array([-tangent[1], tangent[0]]) / tangent_norm
            normals.append(normal)
    return normals

def sample_intensities_along_tangent_normals(contour, gray_image):
    sampled_intensities = []
    normals = compute_normals_from_tangents(contour)
    for i, normal in enumerate(normals):
        point = contour[i + 1][0]
        x, y = point
        normal_length = 10
        intensities = []
        for d in range(-normal_length, normal_length + 1):
            sample_x = int(x + d * normal[0])
            sample_y = int(y + d * normal[1])
            if 0 <= sample_x < gray_image.shape[1] and 0 <= sample_y < gray_image.shape[0]:
                intensities.append(gray_image[sample_y, sample_x])

        if intensities:
            sampled_intensities.append(np.mean(intensities))

    return sampled_intensities, normals

def fit_reflectance_model(normals, intensities):
    def residuals(params):
        rho, theta = params
        predicted_intensities = []
        for normal in normals:
            cos_theta = max(np.dot(normal, [np.cos(theta), np.sin(theta)]), 0)
            predicted_intensities.append(rho * cos_theta)
        return np.array(predicted_intensities) - intensities

    initial_guess = [1.0, 0.0]
    result = least_squares(residuals, initial_guess, bounds=([0, -np.pi], [np.inf, np.pi]))
    return result.x

# Filter contours by size threshold
contour_area_threshold = 0
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > contour_area_threshold]

# Assign unique random colors to each contour
np.random.seed(42)
colors = [
    (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
    for _ in range(len(filtered_contours))
]

# Visualize all light directions 
output_image_contours = image.copy()
light_directions = []
for i, contour in enumerate(filtered_contours):
    intensities, normals = sample_intensities_along_tangent_normals(contour, gray_image)

    if len(intensities) > 0 and len(normals) > 0:
        intensities = np.array(intensities)
        normals = np.array(normals)

        # Fit the Lambertian reflectance model
        rho, theta = fit_reflectance_model(normals, intensities)
        light_direction = np.array([np.cos(theta), np.sin(theta)])
        light_directions.append(light_direction)

        # Visualize light directions for each point on the contour
        for point in contour:
            x, y = point[0]
            start_point = (int(x), int(y))

            # Calculate the end point based on the light direction
            end_point = (
                int(start_point[0] + 50 * light_direction[0]),
                int(start_point[1] - 50 * light_direction[1])  # Y-axis inverted for display
            )

            # Draw the light direction vector
            cv2.arrowedLine(
                output_image_contours,
                end_point,
                start_point,
                color=colors[i % len(colors)],
                thickness=1,
                tipLength=0.1
            )

# Perform K-Means clustering to find dominant direction
light_directions = np.array(light_directions)
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(light_directions)
dominant_cluster = np.argmax(np.bincount(kmeans.labels_))

# Compute the average direction of the dominant cluster
dominant_direction = np.mean(light_directions[kmeans.labels_ == dominant_cluster], axis=0)
dominant_direction /= np.linalg.norm(dominant_direction) 

# Draw the dominant light direction
center = (output_image_dom_dir.shape[1] // 2, output_image_dom_dir.shape[0] // 2)
end_point = (
    int(center[0] + 200 * dominant_direction[0]),
    int(center[1] - 200 * dominant_direction[1]) 
)
cv2.arrowedLine(
    output_image_dom_dir,
    end_point,
    center,
    color=(0, 0, 255),  # Red for the dominant direction
    thickness=3,
    tipLength=0.2
)

# Display the results
cv2.imshow("Light Directions for All Contour Points", output_image_contours)
cv2.imshow("Estimated Dominant Light Direction", output_image_dom_dir)
cv2.imshow("Original Image", image)
cv2.imshow("Contours", canny_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
