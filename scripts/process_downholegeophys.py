import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.image as mpimg
import pandas as pd
import os
import glob

######Process the downhole geophysics image to deskew it

'''def rotate_image(img, angle):
    """Rotate the image by the given angle in degrees."""
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR)

# Load grayscale image
folder_path = 'C://Users//00105010//OneDrive - UWA//Documents//Faults and Barriers Projects//Arrowsmith//Downhole geophysics for digitising//Cropped figures//'
image_files = glob.glob(os.path.join(folder_path, '*.png'))

print(f"Found {len(image_files)} images to process.")

for image_path in image_files:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
            print(f"Skipping {image_path}: could not read image.")
            continue
    print(f"Processing: {os.path.basename(image_path)}")

    # Setup interactive GUI
    initial_angle = 0
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    rotated_img = rotate_image(img, initial_angle)
    im_display = ax.imshow(rotated_img, cmap='gray')
    ax.set_title(f"Adjust skew: {os.path.basename(image_path)}")
    ax.axis('off')

    ax_angle = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_angle, 'Angle (°)', -10.0, 10.0, valinit=initial_angle, valstep=0.1)

    save_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    save_button = Button(save_ax, 'Save', hovercolor='0.975')

    def update(val):
        angle = slider.val
        new_img = rotate_image(img, angle)
        im_display.set_data(new_img)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def save_image(event):
        angle = slider.val
        corrected = rotate_image(img, angle)
        base, ext = os.path.splitext(image_path)
        save_path = base + '_deskewed' + ext
        cv2.imwrite(save_path, corrected)
        print(f"Saved corrected image to: {save_path}")
        plt.close(fig)  # close the current figure after saving to proceed to next image

    save_button.on_clicked(save_image)

    plt.show()'''  # shows the interactive GUI and pauses until closed

#####Once the figures are deskewed, then generate the points for each figure

# Load image
image_path = 'C://Users//00105010//Projects//Otorowiri//data//data_geology//downhole_geophysics//AR4_deskewed.png'
img = mpimg.imread(image_path)

'''
# Prepare filename for output
save_folder = 'C://Users//00105010//Projects//Otorowiri//data//data_geology//downhole_geophysics//processed_gamma//'
os.makedirs(save_folder, exist_ok=True)  # Make sure it exists
base_name = os.path.splitext(os.path.basename(image_path))[0]
csv_filename = os.path.join(save_folder, f"{base_name}_points.csv")

fig, ax = plt.subplots()
fig.canvas.manager.set_window_title('Digitise Points')
fig.canvas.manager.toolbar_visible = True  # Show zoom/pan toolbar
ax.imshow(img)
ax.set_title('Click on the data points you want to extract')
plt.axis('on')

# Click to collect data points
print("Click on the image to collect points. Right-click or press Enter to finish.")
points = plt.ginput(n=-1, timeout=0)  # Infinite clicks until user finishes
plt.close()

# Save to CSV using pandas
df = pd.DataFrame(points, columns=["Gamma", "Depth"])
df.to_csv(csv_filename, index=False)

print(f"Saved {len(points)} points to: {csv_filename}")'''

#######Once the points are collected, then convert the pixel coordinates to real-world coordinates

#show image and get two reference points
plt.imshow(img)
plt.title('Click TWO reference points (with known real-world coordinates)')
ref_pixels = plt.ginput(2, timeout=0)
plt.close()
print("You clicked (pixel coords):", ref_pixels)

real_coords = [(0, 0), (200, 300)] ###real-world coordinates to which the pixel coords correspond
##gamma max is 200, each segment is 50

# convert to numpy arrays
pix = np.array(ref_pixels)
real = np.array(real_coords)

# Calculate scale and offset for each axis
scale_x = (real[1, 0] - real[0, 0]) / (pix[1, 0] - pix[0, 0])
offset_x = real[0, 0] - scale_x * pix[0, 0]
scale_y = (real[1, 1] - real[0, 1]) / (pix[1, 1] - pix[0, 1])
offset_y = real[0, 1] - scale_y * pix[0, 1]
print(f"Scale X: {scale_x}, Offset X: {offset_x}")
print(f"Scale Y: {scale_y}, Offset Y: {offset_y}")

###upload the digitised points
points = pd.read_csv('C://Users//00105010//Projects//Otorowiri//data//data_geology//downhole_geophysics//processed_gamma//AR4_deskewed_points_trial.csv')

# Apply transformation to all points
points['X_real'] = points['Gamma'] * scale_x + offset_x
points['Y_real'] = points['Depth'] * scale_y + offset_y

# Save transformed points with a unique name
# Get the base name of the image (without extension)
save_folder = 'C://Users//00105010//Projects//Otorowiri//data//data_geology//downhole_geophysics//processed_gamma//'
base_name = os.path.splitext(os.path.basename(image_path))[0]
# Append '_transformed_points.csv' to the filename and save it in the specified folder
output_path = os.path.join(save_folder, f"{base_name}_transformed_points.csv")
points.to_csv(output_path, index=False)

print(f"Saved real-world coordinates to: {output_path}")