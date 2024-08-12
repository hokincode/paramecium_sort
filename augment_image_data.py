import os
import cv2

# Set the paths for the input folders
mask_org_folder = "masks_org"
images_org_folder = "images_org"

# Set the paths for the output folders
mask_folder = "masks"
images_folder = "images"

# Create the output folders if they don't exist
os.makedirs(mask_folder, exist_ok=True)
os.makedirs(images_folder, exist_ok=True)

# Set the ROI size and step
roi_size = (256, 256)
roi_step = 128

# Get the list of file names in the mask_org folder
file_names = [file for file in os.listdir(mask_org_folder) if file.lower().endswith(('.tif', '.tiff'))]

# Iterate over the file names
for file_name in file_names:
    # Read the mask image
    mask_path = os.path.join(mask_org_folder, file_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Read the corresponding RGB image
    image_path = os.path.join(images_org_folder, file_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get the dimensions of the images
    mask_height, mask_width = mask.shape[:2]
    image_height, image_width = image.shape[:2]

    # Iterate over the ROI positions
    for y in range(0, mask_height - roi_size[0] + 1, roi_step):
        for x in range(0, mask_width - roi_size[1] + 1, roi_step):
            # Extract the ROI from the mask
            roi_mask = mask[y:y+roi_size[0], x:x+roi_size[1]]

            # Extract the ROI from the image
            roi_image = image[y:y+roi_size[0], x:x+roi_size[1]]

            # Save the ROI as a new image in the mask folder
            new_mask_path = os.path.join(mask_folder, f"{file_name}_{y}_{x}.tif")
            cv2.imwrite(new_mask_path, roi_mask)

            # Save the ROI as a new image in the images folder
            new_image_path = os.path.join(images_folder, f"{file_name}_{y}_{x}.tif")
            cv2.imwrite(new_image_path, roi_image)
