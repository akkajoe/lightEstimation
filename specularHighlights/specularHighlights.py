import cv2
import numpy as np

image = cv2.imread('eye_42_84.jpg')
if image is None:
    print("Image could not be loaded. Please check the path.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Enhance contrast using CLAHE, image preprocessing
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
blurred = cv2.GaussianBlur(enhanced, (9, 9), 0)
edges = cv2.Canny(blurred, 30, 100)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define expected sclera size range (proportion of image size)
height, width = gray.shape[:2]
min_radius = int(0.05 * height)  # Sclera radius lower limit
max_radius = int(0.2 * height)  # Sclera radius upper limit

# Initialize variables for best contour selection
best_ellipse = None
best_fit_score = float('inf')

# Iterate through all contours to find the best match
for cnt in contours:
    if len(cnt) >= 5:  # FitEllipse requires at least 5 points
        ellipse = cv2.fitEllipse(cnt)
        center, axes, angle = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)

        # Check if the ellipse fits within the expected size range
        if min_radius < major_axis / 2 < max_radius and min_radius < minor_axis / 2 < max_radius:
            aspect_ratio = major_axis / minor_axis
            if 0.6 <= aspect_ratio <= 1.6:  # To allow some deviation from a perfect circle
                # Compute fit score based on axis ratio and center proximity to the image center
                fit_score = abs(aspect_ratio - 1.0) + np.linalg.norm(
                    np.array(center) - np.array([width / 2, height / 2])
                )
                if fit_score < best_fit_score:
                    best_fit_score = fit_score
                    best_ellipse = ellipse

# Draw the best ellipse
output = image.copy()
if best_ellipse is not None:
    center, axes, angle = best_ellipse
    cv2.ellipse(output, best_ellipse, (0, 255, 0), 2)  # Green ellipse
    cv2.circle(output, (int(center[0]), int(center[1])), 3, (0, 0, 255), -1)  # Red center point
    print(f"Sclera detected: Center = {center}, Axes = {axes}, Angle = {angle}")
else:
    print("No suitable ellipse detected for the sclera.")

# Convert 2D normal vector to 3D
def convert_to_3d(normal_2d, radius=1.0):
    x, y = normal_2d
    z = np.sqrt(max(0, radius**2 - x**2 - y**2))  # Ensure non-negative sqrt
    return np.array([x, y, z])

# Example 2D normal vector (this should be derived from your highlight location)
N_2d = np.array([0.5, 0.5])  # Replace with actual 2D normal vector
N_3d = convert_to_3d(N_2d)

# Example view direction (camera's optical axis, assuming it's facing directly at the eye)
V = np.array([0, 0, 1])

# Dot product between N_3d and V
dot_product = np.dot(N_3d, V)

# Compute the light direction L using the reflection equation L = 2(N . V)N - V
L = 2 * dot_product * N_3d - V
L = L / np.linalg.norm(L) 

print(f"Light direction L: {L}")

# Visualize the light direction on the 2D image
arrow_length = 50  # Adjust the length of the arrow
arrow_end = (int(center[0] + L[0] * arrow_length), int(center[1] + L[1] * arrow_length))

cv2.arrowedLine(output, (int(center[0]), int(center[1])), arrow_end, (255, 0, 0), 3, tipLength=0.1)


output_resized = cv2.resize(output, (output.shape[1] * 2, output.shape[0] * 2))
cv2.imshow("Detected Sclera and Light Direction", output_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()