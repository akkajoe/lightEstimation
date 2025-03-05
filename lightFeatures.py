import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('semi_transparent (1)\Images\le-gouter-1880.jpg')

# Luminance measurement using LAB color space
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L_channel, a_channel, b_channel = cv2.split(img_lab)

# 0 - 100 scale
L_channel_normalized = (L_channel / 255.0) * 100
luminance_values = L_channel_normalized.ravel()
# Calculate average luminance (L) and std
avg_luminance = np.mean(L_channel_normalized)
std_lum = np.std(luminance_values)
print(f"Average Luminance (L): {avg_luminance:.2f}")
print(f"Standard Deviation of Luminance (L): {std_lum:.2f}")

# # Chromaticity Measurement using XYZ Color Space
# img_xyz = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
# X_channel, Y_channel, Z_channel = cv2.split(img_xyz)

# # Adding small epsilon to prevent /0
# epsilon = 1e-6
# sum_xyz = X_channel + Y_channel + Z_channel + epsilon

# x_chromaticity = X_channel/ sum_xyz
# y_chromaticity = Y_channel/ sum_xyz

# avg_x = np.mean(x_chromaticity)
# avg_y = np.mean(y_chromaticity)
# print(f"Average Chromaticity (x, y): ({avg_x:.2f}, {avg_y:.2f})")


#BGR to RGB for mayplotlib display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Painting")

# Display histogram of the luminance (L) values.
plt.subplot(1, 2, 2)
plt.hist(L_channel_normalized.ravel(), bins=256, color='blue', alpha=0.7)
plt.title("Histogram of Luminance (L)")
plt.xlabel("Luminance (Cd/m^2)")
plt.ylabel("Frequency")

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# In the HSV color space, the V (value) channel represents brightness
v_channel = img_hsv[:, :, 2]
mean_hsv_brightness = np.mean(v_channel)
print(f"Mean Brightness (HSV V-channel): {mean_hsv_brightness:.2f}")
# The formula is: brightness = sqrt(0.299 * R^2 + 0.587 * G^2 + 0.114 * B^2)
R = img_rgb[:, :, 0].astype(np.float32)
G = img_rgb[:, :, 1].astype(np.float32)
B = img_rgb[:, :, 2].astype(np.float32)

plt.show()

# Compute perceived brightness for each pixel using HSP color model
perceived_brightness = np.sqrt(0.299 * (R ** 2) + 0.587 * (G ** 2) + 0.114 * (B ** 2))
mean_perceived_brightness = np.mean(perceived_brightness)
print(f"Mean Perceived Brightness (Finley's Formula): {mean_perceived_brightness:.2f}")

# --- Visualization ---
plt.figure(figsize=(12, 5))

# Display original image
plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')

# Display HSV brightness channel
plt.subplot(1, 3, 2)
plt.imshow(v_channel, cmap='gray')
plt.title("HSV Brightness (V-channel)")
plt.axis('off')

# Display the computed perceived brightness
plt.subplot(1, 3, 3)
plt.imshow(perceived_brightness, cmap='gray')
plt.title("Perceived Brightness")
plt.axis('off')

plt.tight_layout()
plt.show()

# 1) Convert to float in [0, 1] for processing
img_float = img.astype(np.float32) / 255.0
# Convert from BGR to RGB for consistent luminance calculation
img_rgb = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)
# Compute luminance (L) using a weighted sum (approx. human perception)
#    L = 0.27*R + 0.67*G + 0.06*B
R = img_rgb[:, :, 0]
G = img_rgb[:, :, 1]
B = img_rgb[:, :, 2]
L = 0.27 * R + 0.67 * G + 0.06 * B
# Log-transform the luminance to better handle large dynamic ranges
# Add a small delta to avoid log(0)
delta = 1e-6
log_L = np.log(L + delta)
'''Bilateral filter on log luminance to separate base & detail layers
    Tune the parameters (d, sigmaColor, sigmaSpace) for best results.
    - d: diameter of the pixel neighborhood
    - sigmaColor: how much color variance is tolerated
    - sigmaSpace: how far the filter considers neighbors in space'''
d = 15
sigmaColor = 0.4  # Range in log space
sigmaSpace = 15   # Spatial smoothing
log_base = cv2.bilateralFilter(log_L, d, sigmaColor, sigmaSpace)
log_detail = log_L - log_base
# Compress the base layer
# Apply a scale factor alpha < 1.0 to reduce overall brightness range
alpha = 0.8  # Adjust as needed
log_base_compressed = alpha * log_base
# Reconstruct tone-mapped luminance in log domain
log_tone_mapped = log_base_compressed + log_detail
L_tone_mapped = np.exp(log_tone_mapped)
#    For each channel: Out_c = (Channel / L) * L_tone_mapped
img_tone_mapped = np.zeros_like(img_rgb)
for c in range(3):
    img_tone_mapped[:, :, c] = (img_rgb[:, :, c] / (L + delta)) * L_tone_mapped
img_tone_mapped = np.clip(img_tone_mapped, 0, 1)
img_tone_mapped_8u = (img_tone_mapped * 255).astype(np.uint8)
img_tone_mapped_bgr = cv2.cvtColor(img_tone_mapped_8u, cv2.COLOR_RGB2BGR)

cv2.imshow("Original Image", img)
cv2.imshow("Locally Tone Mapped Image", img_tone_mapped_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

min_val = np.min(img_float)
max_val = np.max(img_float)
mean_val = np.mean(img_float)
std_val = np.std(img_float)

print(f"Min intensity: {min_val:.2f}")
print(f"Max intensity: {max_val:.2f}")
print(f"Mean intensity: {mean_val:.2f}")
print(f"Std intensity:  {std_val:.2f}")

# ratio of min to max (in percentage)
uniformity_ratio = (min_val / (max_val + 1e-6)) * 100
print(f"Uniformity (min/max * 100): {uniformity_ratio:.2f}%")

# Alternative measure: 1 - (std/mean)
uniformity_std_mean = (1.0 - (std_val / (mean_val + 1e-6))) * 100
print(f"Uniformity (1 - std/mean) * 100: {uniformity_std_mean:.2f}%")
# define a tolerance of +/- 5% around the mean.
tolerance_percentage = 5.0
lower_bound = mean_val * (1 - tolerance_percentage / 100.0)
upper_bound = mean_val * (1 + tolerance_percentage / 100.0)
# Count how many pixels fall outside the tolerance range
mask_outside = (img_float < lower_bound) | (img_float > upper_bound)
num_outside = np.count_nonzero(mask_outside)
total_pixels = img_float.size
percentage_outside = (num_outside / total_pixels) * 100
print(f"Tolerance range: [{lower_bound:.2f}, {upper_bound:.2f}]")
print(f"Pixels outside tolerance: {num_outside} ({percentage_outside:.2f}%)")
# take a horizontal line in the middle of the image
row_index = img_float.shape[0] // 2
line_profile = img_float[row_index, :]
# Plot the line profile
plt.figure(figsize=(10, 4))
plt.plot(line_profile, label=f"Row {row_index} profile")
plt.axhline(mean_val, color='red', linestyle='--', label='Mean Intensity')
plt.title("Line Profile Across Middle Row")
plt.xlabel("Pixel X-coordinate")
plt.ylabel("Intensity Value")
plt.legend()
plt.show()