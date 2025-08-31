import numpy as np
import cv2
import matplotlib.pyplot as plt 

def choose_weighted_centroid(pixels, centroids):
    distances = np.min(np.sqrt(((pixels - centroids[:, np.newaxis])**2).sum(axis=2)), axis=0)

    squared_distances = distances ** 2

    probabilities = squared_distances / np.sum(squared_distances)
    chosen_index = np.random.choice(len(pixels), size=1, p=probabilities)[0]
    return pixels[chosen_index]

def kmeans_clustering(image, k, max_iterations=10, visualize=True):
    l2_norms = []
    pixels = image.reshape(-1, 3).astype(np.float32)
    centroids = np.zeros((k, 3), dtype=np.float32)
    centroids[0] = pixels[np.random.choice(len(pixels))]
    for i in range(1, k):
        centroids[i] = choose_weighted_centroid(pixels, centroids[:i])
    
    if visualize:
        plt.ion()  # Turn on interactive mode
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.set_title("Original Image")
        ax1.imshow(image)
        ax1.axis('off')
        ax2.set_title(f"Quantized Image (k={k}, Iteration 0)")
        ax2.axis('off')
        # Initialize with a blank image
        img_display = ax2.imshow(np.zeros_like(image))
        plt.tight_layout()
    
    for iter in range(max_iterations):
        distances = np.sqrt(((pixels - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        new_centroids = np.zeros_like(centroids)
        num_of_beans = k
        for ke in range(num_of_beans):
            if np.sum(labels == ke) > 0:
                new_centroids[ke] = np.mean(pixels[labels == ke], axis=0)
            else:
                distances_to_k = np.min(distances, axis=0)
                farthest_pixel_idx = np.argmax(distances_to_k)
                new_centroids[ke] = pixels[farthest_pixel_idx]
        
        # Compute L2 norm for this iteration
        quantized_pixels = new_centroids[labels]
        quantized_image = quantized_pixels.reshape(image.shape).astype(np.float32)
        l2_norm = np.linalg.norm(image - quantized_image)
        print(f"Iteration {iter + 1}, L2 norm: {l2_norm:.2f}")
        l2_norms.append(l2_norm)
        
        if visualize:
            # Update the quantized image display
            img_display.set_data(quantized_image)
            ax2.set_title(f"Quantized Image (k={k}, Iteration {iter + 1}, L2 Norm: {l2_norm:.2f})")
            plt.draw()
            plt.pause(0.1)  # Pause to show update
            fig.canvas.flush_events()  # Ensure the canvas updates
        
        if np.allclose(centroids, new_centroids, atol=0.00001):
            break
        centroids = new_centroids
       
    if visualize:
        plt.ioff()  # Turn off interactive mode
        plt.close(fig)  # Close the figure
    
    return labels, centroids, l2_norms

def calculate_l2_norm(original, quantized):
    norm = np.linalg.norm(original - quantized).sum()
    # norm = np.sqrt(np.sum((original - quantized) ** 2))
    return norm

def kmeans_quantization(image, k=16, max_iterations=10, visualize=False):
    # Run k-means clustering
    labels, centroids , L2Norms = kmeans_clustering(image, k, max_iterations)
    
    # Create quantized image by assigning each pixel its centroid's RGB value
    quantized_pixels = centroids[labels]
    quantized_image = quantized_pixels.reshape(image.shape).astype(np.float32)
    
    # Compute L2 norm between original and quantized images
    l2_norm = calculate_l2_norm(image, quantized_image)
    print(f"L2 norm between original and quantized image: {l2_norm}")
    
    # Visualize results if requested
    if visualize:
        plt.figure(figsize=(12, 4))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.axis('off')
        plt.imshow(image)
        
        # Quantized image
        plt.subplot(1, 2, 2)
        plt.title(f"Quantized Image (k={k})")
        plt.axis('off')
        plt.imshow(quantized_image)
        
        # Show color palette
        plt.figure(figsize=(8, 2))
        plt.title("Color Palette")
        palette = np.repeat(centroids[np.newaxis, :, :], 100, axis=0)
        palette = np.repeat(palette, 100, axis=1)
        plt.imshow(palette)
        plt.axis('off')
        
        plt.show()
    
    return quantized_image , L2Norms


imageName = ['lena.png' , 'peppers.tif']
for image in imageName:
    # Load and preprocess image
    img = cv2.imread(r'./images/' + image).astype(np.float32)/255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    k = 16
    max_iterations = 20
    visualize = True

    # Process images
    quantized, lena_l2_norms = kmeans_quantization(img, k, max_iterations, visualize)

    # Calculate final L2 norms
    lena_l2 = calculate_l2_norm(img, quantized)


    # Save L2 norms
    with open('L2_norm_log.txt', 'w') as f:
        f.write(image +' L2 Norms:\n' + '\n'.join(map(str, lena_l2_norms)) + '\n')

    # Visualize results
    plt.imshow(quantized)
    plt.title(image+' Quantized')
    plt.axis('off')
    plt.show()

    # Save quantized image (convert back to [0, 255] and BGR for OpenCV)
    quantized_image_bgr = (cv2.cvtColor(quantized, cv2.COLOR_RGB2BGR) * 255.0).astype(np.uint8)
    cv2.imwrite('quantized_'+image + '.png', quantized_image_bgr)

