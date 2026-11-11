import numpy as np
import matplotlib.pyplot as plt


pixel_type = np.uint8
image = np.array([
    [20,20,45],
    [30,45,60],
    [20,40,60],
    [100,200,240]
], dtype=pixel_type)

print("Image shape:", image.shape)
print("Image height:", image.shape[0])
print("Image width:", image.shape[1])
print("First pixel:", image[0,0])
print("Value at row 2, column 3:", image[1,2])
print("Last Pixel:", image[-1,-1])
print("Total pixels:", image.size)

def image_resizing_nearest_neighbor_interpolation(image,height,width):
    original_height, original_width = image.shape
    height_ratio = original_height / height
    width_ratio = original_width / width
    resized = np.zeros((height,width), dtype=pixel_type)
    
    for i in range(height):
        for j in range(width):
            original_row = int(i*height_ratio)
            original_column = int(j*width_ratio)
            resized[i,j] = image[original_row,original_column]
    return resized

def linear_interpolation(a, b, weight):
    if not 0 <= weight <= 1:
        raise ValueError("Weight must be between 0 and 1")
    return (1 - weight) * a + weight * b


def resize_bilinear(image,new_height,new_width):
    original_height,original_width = image.shape
    h_ratio = original_height / new_height
    w_ratio = original_width / new_width
    resized = np.zeros((new_height,new_width),dtype=pixel_type) 
    for i in range(new_height):
        for j in range(new_width):
            y = i*h_ratio
            x = j*w_ratio
            top_row= int(y)
            bottom_row= min(int(y) + 1, image.shape[0] - 1)
            left_col = int(x)
            right_col = min(int(x) + 1, image.shape[1] - 1)
            top_left = image[top_row, left_col]
            top_right = image[top_row, right_col]
            bottom_left = image[bottom_row, left_col]
            bottom_right = image[bottom_row, right_col]
            dx = x - left_col
            dy = y - top_row
            pixel_value = bilinear_interpolation(top_left,top_right,bottom_left,bottom_right,dx,dy)
            resized[i,j] = np.uint8(np.clip(np.round(pixel_value),0,255))
    return resized

def bilinear_interpolation(top_left,top_right,bottom_left,bottom_right,hor_weight,ver_weight):
    
    top_inter = linear_interpolation(top_left,top_right,hor_weight)
    bot_inter = linear_interpolation(bottom_left,bottom_right, hor_weight)
    return linear_interpolation(top_inter,bot_inter,ver_weight)


nearest_image = image_resizing_nearest_neighbor_interpolation(image, 12, 9)
bilinear_image = resize_bilinear(image, 12, 9)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(nearest_image, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Nearest Neighbor")
axes[0].axis("off")

axes[1].imshow(bilinear_image, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Bilinear Interpolation")
axes[1].axis("off")

plt.tight_layout()
plt.show()
print("Original image:")
print(image)

print("Bilinear resized image:")
print(bilinear_image)