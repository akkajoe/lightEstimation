import cv2
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

#image_path = "semi_transparent (1)\Images\le-gouter-1880.jpg"
image_path = "semi_transparent (1)\semi_transparent\Photo\short.jpg"
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    sampled_normals = []

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
            sampled_normals.append(normal)

    return sampled_intensities, sampled_normals

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

# Collect data across all contours
all_intensities = []
all_normals = []

for contour in contours:
    intensities, normals = sample_intensities_along_tangent_normals(contour, gray_image)
    all_intensities.extend(intensities)
    all_normals.extend(normals)

# Fit the Lambertian reflectance model
all_intensities = np.array(all_intensities)
all_normals = np.array(all_normals)
if len(all_intensities) > 0 and len(all_normals) > 0:
    rho, theta = fit_reflectance_model(all_normals, all_intensities)

    illumination_angle = np.degrees(theta)
    print(f"Estimated illumination angle: {illumination_angle:.2f} degrees")
    print(f"Estimated albedo (rho): {rho:.2f}")

    predicted_intensities = []
    light_direction = np.array([np.cos(theta), np.sin(theta)])
    for normal in all_normals:
        cos_theta = max(np.dot(normal, light_direction), 0)
        predicted_intensities.append(rho * cos_theta)
    predicted_intensities = np.array(predicted_intensities)

    # Normalize the observed and predicted intensities
    all_intensities = all_intensities / np.max(all_intensities)
    predicted_intensities = predicted_intensities / np.max(predicted_intensities)
    residuals = all_intensities - predicted_intensities

    # Smooth the predicted intensities
    smoothed_predicted = gaussian_filter1d(predicted_intensities, sigma=10)

    # Sample data points for clearer visualization
    sample_indices = np.linspace(0, len(all_intensities) - 1, 500, dtype=int)
    sampled_observed = all_intensities[sample_indices]
    sampled_predicted = predicted_intensities[sample_indices]
    sampled_smoothed = smoothed_predicted[sample_indices]

    # Plot observed vs. predicted intensities
    plt.figure(figsize=(8, 6))
    plt.scatter(sample_indices, sampled_observed, label="Observed Intensities (Sampled)", color="blue", alpha=0.6, s=10)
    plt.plot(sample_indices, sampled_predicted, label="Predicted Intensities (Sampled)", color="red", linestyle="--", alpha=0.6)
    plt.plot(sample_indices, sampled_smoothed, label="Smoothed Predicted Trend", color="green", linewidth=2)
    plt.xlabel("Sample Index")
    plt.ylabel("Normalized Intensity")
    plt.title("Observed vs. Predicted Intensities (Sampled)")
    plt.legend()
    plt.grid()
    plt.show()

    # Sample residuals for clarity
    sampled_residuals = residuals[sample_indices]

    # Plot residuals
    plt.figure(figsize=(8, 6))
    plt.plot(sample_indices, sampled_residuals, label="Residuals", color="green")
    plt.axhline(0, color='black', linestyle='--', linewidth=1, label="Zero")
    plt.xlabel("Sample Index")
    plt.ylabel("Residual (Observed - Predicted)")
    plt.title("Residuals of Observed vs. Predicted Intensities (Sampled)")
    plt.legend()
    plt.grid()
    plt.show()

    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Residual: {np.mean(residuals):.4f}")
    print(f"Standard Deviation of Residuals: {np.std(residuals):.4f}")


    # Display results
    height, width, _ = image.shape
    center = (width // 2, height // 2)
    end_point = (
        int(center[0] + 200 * np.cos(theta)),
        int(center[1] - 200 * np.sin(theta))
    )
    output_image = image.copy()
    cv2.arrowedLine(
        output_image,
        end_point,
        center,
        color=(0, 255, 0),
        thickness=2,
        tipLength=0.1
    )
    cv2.imshow("Estimated Light Source Direction", output_image)
    cv2.imshow("Original Image", image)
    cv2.imshow("Contours", canny_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Insufficient data to estimate light direction.")
