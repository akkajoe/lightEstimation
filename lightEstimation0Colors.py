import cv2
import numpy as np

#image_path = "semi_transparent (1)\semi_transparent\Bracquemond\le-gouter-1880.jpg"
image_path = "semi_transparent (1)\semi_transparent\Photo\short.jpg"
# image_path = "semi_transparent (1)\Images\young-woman.jpg"

image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3) 
gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
gradient_direction = np.arctan2(sobel_y, sobel_x)

canny_edges = cv2.Canny(blurred_image, 110, 180)
contours, hierarchy = cv2.findContours(canny_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  

# Filter contours by size (optional for better visibility)
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]  # Threshold for small contours
if len(filtered_contours) < len(contours):
    print(f"Filtered out {len(contours) - len(filtered_contours)} small contours.")

# Create a blank image to visualize filtered contours
filtered_contour_image = np.zeros_like(image)
cv2.drawContours(filtered_contour_image, filtered_contours, -1, (0, 0, 255), 1)  # Draw filtered contours in red

# Occluding Contour Analysis - Draw normal vectors and sample intensity
intensity_values = []
direction_vectors = []
for contour in contours:
    for point in contour:
        x, y = point[0]

        if 0 <= y < gradient_direction.shape[0] and 0 <= x < gradient_direction.shape[1]: # To avoid index errors
            angle = gradient_direction[y, x] # Represents the direction at which the gradient is pointing at the pixel

            # Calculate normal vectors (outward)
            normal_length = 10
            normal_x = int(x + normal_length * np.sin(angle))
            normal_y = int(y - normal_length * np.cos(angle))

            # Sample pixel intensities along the normal vector
            sampled_intensities = []
            for i in range(1, normal_length + 1):
                sample_x = int(x + i * np.sin(angle))
                sample_y = int(y - i * np.cos(angle))
                if 0 <= sample_y < gray_image.shape[0] and 0 <= sample_x < gray_image.shape[1]:
                    intensity_value = gray_image[sample_y, sample_x]
                    sampled_intensities.append(intensity_value)
                    intensity_values.append(intensity_value)
            
            if len(sampled_intensities) > 1:
                intensity_gradients = np.diff(sampled_intensities) # np.diff(sampled_intensities): Calculates the difference between consecutive intensity values to create an array representing the gradient (change in intensity) along the normal vector.
                mean_gradient = np.mean(intensity_gradients) # To avoid noise using an average
                
            if mean_gradient > 0: # light source may be in direction of N
                normal_vector = np.array([np.sin(angle), -np.cos(angle)])
                direction_vectors.append((normal_vector, mean_gradient))

# Assign unique random colors to each contour
np.random.seed(42)
colors = [
    (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
    for _ in range(len(contours))
]

# Visualize all light directions with start points on contours
output_image_contours = image.copy()
for i, contour in enumerate(filtered_contours):
    for point in contour:
        x, y = point[0]
        start_point = (int(x), int(y))

        # Get direction vector corresponding to this point (use first valid direction as fallback)
        direction = direction_vectors[i % len(direction_vectors)][0]

        # Calculate the end point based on the direction vector
        end_point = (
            int(start_point[0] + 100 * direction[0]),
            int(start_point[1] - 100 * direction[1])
        )

        cv2.arrowedLine(output_image_contours, end_point, start_point, color=colors[i % len(colors)], thickness=1, tipLength=0.1)

if direction_vectors:
    weighted_sum = np.sum([v * weight for v, weight in direction_vectors], axis = 0)
    estimated_light_direction = weighted_sum/np.linalg.norm(weighted_sum)

output_image = image.copy()
start_point = (output_image.shape[1] // 2, output_image.shape[0] // 2) # center of the image
arrow_length = 200 
end_point = (
    int(start_point[0] + arrow_length * estimated_light_direction[0]),
    int(start_point[1] + arrow_length * estimated_light_direction[1])
)

# Draw the arrow on the image
cv2.arrowedLine(
    output_image,
    end_point,
    start_point,
    color=(0, 255, 0),  
    thickness=2,
    tipLength=0.3  
)

cv2.imshow("Original image", image)
cv2.imshow("Contors", canny_edges)
cv2.imshow("Filtered Contours", filtered_contour_image)
cv2.imshow("Estimated Light Source Direction", output_image)
cv2.imshow("Light Direction on Contours Colored", output_image_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()