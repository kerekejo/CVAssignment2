import numpy as np
import cv2
import matplotlib.pyplot as plt

#Load the image
image = cv2.imread(r"C:\Users\Joe\OneDrive\Desktop\PythonPrograms\barbara_512.png", cv2.IMREAD_GRAYSCALE)

#Forward difference in X-direction 
forward_diff_x = np.diff(image, axis=0)
forward_diff_x = np.vstack((forward_diff_x, forward_diff_x[-1])) 
    
#Backward difference in X-direction
backward_diff_x = np.diff(image, axis=0)
backward_diff_x = np.vstack((backward_diff_x[0], backward_diff_x))  
#Central difference in X-direction
central_diff_x = (np.roll(image, -1, axis=0) - np.roll(image, 1, axis=0)) / 2

#Second derivative in X-direction
second_derivative_x = np.roll(image, -1, axis=0) - 2 * image + np.roll(image, 1, axis=0)
    
#Second derivative in Y-direction
second_derivative_y = np.roll(image, -1, axis=1) - 2 * image + np.roll(image, 1, axis=1)


#Plot the original image and its derivatives
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

axs[0, 1].imshow(forward_diff_x, cmap='gray')
axs[0, 1].set_title('Forward Difference X')
axs[0, 1].axis('off')

axs[0, 2].imshow(backward_diff_x, cmap='gray')
axs[0, 2].set_title('Backward Difference X')
axs[0, 2].axis('off')

axs[1, 0].imshow(central_diff_x, cmap='gray')
axs[1, 0].set_title('Central Difference X')
axs[1, 0].axis('off')

axs[1, 1].imshow(second_derivative_x, cmap='gray')
axs[1, 1].set_title('Second Derivative X')
axs[1, 1].axis('off')

axs[1, 2].imshow(second_derivative_y, cmap='gray')
axs[1, 2].set_title('Second Derivative Y')
axs[1, 2].axis('off')

plt.tight_layout()
plt.show()

#Part2

def gaussian_kernel(size, sigma=1):
    k = size // 2
    ax = np.arange(-k, k+1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel) 

def apply_gaussian_filter(image, size, sigma):
    kernel = gaussian_kernel(size, sigma)
    return cv2.filter2D(image, -1, kernel)

def apply_gaussian_l1_filter(image, size, sigma):
    kernel = gaussian_kernel(size, sigma)
    kernel = np.abs(kernel)
    kernel /= np.sum(kernel) 
    return cv2.filter2D(image, -1, kernel)

def apply_gaussian_linf_filter(image, size, sigma):
    kernel = gaussian_kernel(size, sigma)
    kernel = np.max(kernel) - kernel  
    kernel /= np.sum(kernel)  
    return cv2.filter2D(image, -1, kernel)

#Apply and display Gaussian filters
filtered_image_l2 = apply_gaussian_filter(image, 3, 1)
filtered_image_l1 = apply_gaussian_l1_filter(image, 3, 1)
filtered_image_linf = apply_gaussian_linf_filter(image, 3, 1)

#Display the filtered images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(filtered_image_l2, cmap='gray')
plt.title('Gaussian Filter (l2 norm)')

plt.subplot(1, 3, 2)
plt.imshow(filtered_image_l1, cmap='gray')
plt.title('Gaussian Filter (l1 norm)')

plt.subplot(1, 3, 3)
plt.imshow(filtered_image_linf, cmap='gray')
plt.title('Gaussian Filter (l infinity norm)')

plt.tight_layout()
plt.show()

#Part 3
noisy_image = cv2.imread(r'C:\Users\Joe\OneDrive\Desktop\PythonPrograms\noisy_pollens.png', cv2.IMREAD_GRAYSCALE)

plt.hist(noisy_image.ravel(), bins=256, range=[0, 256])
plt.title('Histogram')
plt.show()

denoised_image = cv2.medianBlur(noisy_image, 3)

min_val = np.min(denoised_image)
max_val = np.max(denoised_image)
contrast_stretched = (denoised_image - min_val) * (255 / (max_val - min_val))

plt.imshow(contrast_stretched, cmap='gray')
plt.title('Denoised')
plt.show()
