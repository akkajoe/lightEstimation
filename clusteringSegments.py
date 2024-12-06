import cv2
import numpy as np

# Load the image
#image_path = r"C:\Users\anush\GitHubRepos\lightEstimation\semi_transparent (1)\f2c45d39f877c507e7a60dc83dad857a.jpg"
#image_path = r"C:\Users\anush\GitHubRepos\lightEstimation\semi_transparent (1)\Images\Amanda-Coldicutt--700x1050.jpg"
#image_path = r"C:\Users\anush\GitHubRepos\lightEstimation\semi_transparent (1)\f96c65bd0351c92036c72e48eb7a8576.jpg"
image_path = r"C:\Users\anush\GitHubRepos\lightEstimation\semi_transparent (1)\semi_transparent\Bracquemond\le-gouter-1880.jpg"
#image_path = r"C:\Users\anush\GitHubRepos\lightEstimation\semi_transparent (1)\semi_transparent\Bracquemond\the-lady-in-white.jpg"
#image_path = r"C:\Users\anush\GitHubRepos\lightEstimation\semi_transparent (1)\Images\Amanda-Coldicutt--700x1050.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("The image could not be loaded. Check the file path.")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Remove pure white background (intensity > 245)
background_mask = cv2.inRange(gray_image, 246, 255) # Mask for intensities which are likely white background/color
gray_image[background_mask > 0] = 245

# Flatten for K-means clustering
pixel_values = gray_image.flatten().reshape((-1, 1)).astype(np.float32)

# Apply K-means clustering
k = 5  
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Reshape labels to the image dimensions
labels_image = labels.reshape(gray_image.shape)

# Sort clusters by intensity (centers)
sorted_clusters = np.argsort(centers.flatten())

# Create a mask for the brightest cluster
brightest_cluster = sorted_clusters[-1]
bright_mask = (labels_image == brightest_cluster).astype(np.uint8) * 255

# Calculate gradients
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)

# Adaptive gradient threshold
gradient_threshold = gradient_magnitude.max() * 0.1
gradient_mask = (gradient_magnitude > gradient_threshold).astype(np.uint8) * 255

# Combine bright mask and gradient mask
combined_mask = cv2.bitwise_and(bright_mask, gradient_mask)

# Exclude shadows by thresholding low-intensity regions
shadow_threshold = 120  # Adjust based on the image
combined_mask[gray_image < shadow_threshold] = 0

# Remove large shadow contours using contour area filtering
contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filtered_mask = np.zeros_like(combined_mask)
min_area = 50  # Minimum contour area to consider
max_area = 800  # Maximum contour area to exclude shadows

for contour in contours:
    area = cv2.contourArea(contour)
    average_intensity = np.mean([gray_image[point[0][1], point[0][0]] for point in contour])
    if min_area <= area <= max_area and average_intensity > shadow_threshold:
        cv2.drawContours(filtered_mask, [contour], -1, 255, -1)

# Highlight relevant regions
highlighted_image = image.copy()
highlighted_image[filtered_mask == 255] = [0, 0, 255]  # Red highlight for relevant regions

# Calculate direction of light based on filtered regions
contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    intensity_vectors = []
    for contour in contours:
        for point in contour:
            x, y = point[0]
            if 0 <= y < gradient_magnitude.shape[0] and 0 <= x < gradient_magnitude.shape[1]:
                angle = np.arctan2(sobel_y[y, x], sobel_x[y, x])  # Gradient direction
                intensity_vectors.append([np.cos(angle), np.sin(angle)])

    # Average direction of light
    average_direction = np.mean(intensity_vectors, axis=0)
    average_direction /= np.linalg.norm(average_direction)  # Normalize

    output_image = image.copy()
    start_point = (output_image.shape[1] // 2, output_image.shape[0] // 2)
    arrow_length = 200
    end_point = (
        int(start_point[0] + arrow_length * average_direction[0]),
        int(start_point[1] + arrow_length * average_direction[1]),
    )
    cv2.arrowedLine(output_image, start_point, end_point, (0, 0, 255), 3, tipLength=0.2)
else:
    output_image = image.copy()
    print("No significant regions found for light source estimation.")

cv2.imshow("Filtered Highlighted Regions", highlighted_image)
cv2.imshow("Estimated Light Source Direction", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()