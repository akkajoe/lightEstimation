import cv2
import numpy as np

# Load and convert the image to grayscale
image_path = r"C:\Users\anush\OneDrive\Documents\PSU\Research\semi_transparent (1)\semi_transparent\Bracquemond\le-gouter-1880.jpg"
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blurring
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Apply Sobel filters
sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3) # cv2.CV_64F: Data type to hold gradients with potential negative values.

# Calculate gradient magnitude and direction 
gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
gradient_direction = np.arctan2(sobel_y, sobel_x)

# Apply Canny edge detection
canny_edges = cv2.Canny(blurred_image, 50, 150)

# Find contours
contours, hierarchy = cv2.findContours(canny_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # cv2.RETR_TREE: Retrieves all contours and reconstructs a full hierarchy of nested contours
# cv2.CHAIN_APPROX_SIMPLE: Compresses horizontal, vertical, and diagonal segments and leaves only their end points, reducing memory usage.
# contours: A list of all detected contours, where each contour is an array of (x, y) coordinates representing a boundary point.

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

# Highlight bright regions in the image based on intensity threshold
threshold = 180 
mask = (gray_image > threshold).astype(np.uint8) * 255  # Create a binary mask

# Apply the mask as an overlay on the original image
highlighted_image = cv2.addWeighted(image, 0.4, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.6, 0)

# Stack the original image, edges, and highlighted image side by side
edges_colored = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)  # Convert edges to 3-channel for display
image_comparisons = np.hstack([image, edges_colored, highlighted_image])

# Display the images
cv2.imshow("Original, Edges, and Highlighted Image", image_comparisons)
cv2.waitKey(0)
cv2.destroyAllWindows()

''' Key Obeservations to make:
- Brightness Increase/Decrease: If the intensities increase outward from the contour, the light source may be in that direction. 
Conversely, if the intensities decrease, the light source might be behind or opposite that direction.
- If multiple sampled normals around the object show similar intensity changes, the average of these vectors can help point to the direction of the light source.
- Areas where the intensity sharply decreases indicate shadows (light is coming from the opposite direction), 
and areas where the intensity increases indicate highlights (light is coming from that direction).
'''

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
    start_point,
    end_point,
    color=(0, 255, 0),  
    thickness=2,
    tipLength=0.3  
)

cv2.imshow("Estimated Light Source Direction", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Try on more paintings

# Segment anything, compare cluster results
# Extract colors as features to predict semi-transparent 
# suyang