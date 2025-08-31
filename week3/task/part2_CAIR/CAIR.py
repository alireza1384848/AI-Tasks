import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
import os

def load_maps(img_dir, name):
    """
    Load image, saliency map, and depth map if they exist.
    """
    img_path = os.path.join(img_dir, f'{name}/{name}.png')
    img = plt.imread(img_path)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.max() > 1.0:
        img = img / 255.0

    smap = None
    smap_path = os.path.join(img_dir, f'{name}/{name}_SMap.png')
    if os.path.exists(smap_path):
        smap = plt.imread(smap_path)
        if len(smap.shape) == 3:
            smap = np.mean(smap, axis=2)
        smap = smap / np.max(smap) if np.max(smap) > 0 else smap
        print(smap)

    dmap = None
    dmap_path = os.path.join(img_dir, f'{name}/{name}_DMap.png')
    if os.path.exists(dmap_path):
        dmap = plt.imread(dmap_path)
        if len(dmap.shape) == 3:
            dmap = np.mean(dmap, axis=2)
        dmap = dmap / np.max(dmap) if np.max(dmap) > 0 else dmap

    return img, smap, dmap

def compute_sobel_energy(img):
    """
    Compute energy map using Sobel filters for gradients.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    dx = sobel(gray, axis=1, mode='constant')
    dy = sobel(gray, axis=0, mode='constant')
    energy = np.abs(dx) + np.abs(dy)
    return energy / np.max(energy) if np.max(energy) > 0 else energy

def combine_energy(sobel_energy, smap=None, dmap=None):
    """
    Combine Sobel energy, saliency map, and depth map into a single energy map.
    """
    energy = 0.45 * sobel_energy
    alpha = 0.05 # Weight for saliency
    beta = 0.5  # Weight for depth
    if smap is not None:
        energy = energy + alpha * smap
    if dmap is not None:
        energy = energy + beta * (dmap)
    return energy / np.max(energy) if np.max(energy) > 0 else energy

def find_vertical_seam(energy):
    """
    Find the lowest energy vertical seam using dynamic programming.
    Vectorized for speed where possible.
    """
    h, w = energy.shape
    M = energy.copy().astype(np.float64)
    large = 1e10
    for i in range(1, h):
        prev = M[i-1]
        left = np.roll(prev, 1)
        left[0] = large
        right = np.roll(prev, -1)
        right[-1] = large
        min_prev = np.minimum(np.minimum(left, right), prev)
        M[i] += min_prev
    j = np.argmin(M[-1])
    seam = np.zeros(h, dtype=int)
    seam[h-1] = j
    for i in range(h-2, -1, -1):
        min_j = j
        min_val = M[i][j]
        if j > 0 and M[i][j-1] < min_val:
            min_val = M[i][j-1]
            min_j = j - 1
        if j < w - 1 and M[i][j+1] < min_val:
            min_j = j + 1
        seam[i] = min_j
        j = min_j
    return seam

def remove_vertical_seam(img, seam):
    """
    Remove the seam from the image using masking for efficiency.
    """
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) == 3 else 1
    mask = np.ones((h, w), dtype=bool)
    for i in range(h):
        mask[i, seam[i]] = False
    if c > 1:
        mask_3d = np.repeat(mask[:, :, np.newaxis], c, axis=2)
        new_img = img[mask_3d].reshape(h, w-1, c)
    else:
        new_img = img[mask].reshape(h, w-1)
    return new_img

def seam_carve(img, num_seams, smap=None, dmap=None, visualize=False):
    """
    Perform seam carving to reduce width by num_seams columns.
    """
    carved_img = img.copy()
    original_shape = carved_img.shape
    is_color = len(original_shape) == 3
    
        # Initialize single figure for visualization
    if visualize:
      plt.ion()
      fig, ax = plt.subplots()
        

    for k in range(num_seams):
        sobel_energy = compute_sobel_energy(carved_img)
        energy = combine_energy(sobel_energy, smap, dmap)
        seam = find_vertical_seam(energy)
        
        if visualize:
            img_copy = carved_img.copy()
            for i in range(len(seam)):
                if is_color:
                    img_copy[i, seam[i]] = [1.0, 0, 0]
                else:
                    img_copy[i, seam[i]] = 1.0
            ax.clear()
            ax.imshow(img_copy)
            ax.set_title(f"Seam {k + 1}/{num_seams}")
            plt.draw()
            plt.pause(0.01)  # Brief pause to show update
        
        carved_img = remove_vertical_seam(carved_img, seam)
        
        # Update saliency and depth maps if provided
        if smap is not None:
            smap = remove_vertical_seam(smap[:, :], seam)[:, :]
        if dmap is not None:
            dmap = remove_vertical_seam(dmap[:, :], seam)[:, :]         
    return carved_img

# Main execution
num_seams = 200
visualize = True
img_dir = r'D:\techstack2025-ai\week3\task\part2_CAIR\images'
names = ['Snowman', 'Diana' , 'Baby']

for name in names:
    img, smap, dmap = load_maps(img_dir, name)
    resized = seam_carve(img, num_seams, smap, dmap, visualize=visualize)
    output_path = f'{name}_resized.png'
    plt.imsave(output_path, resized)
    print(output_path)
    print(f'Resized {name} saved to {output_path}')
    plt.imshow(resized)
    plt.show() 