import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import customtkinter as ctk
from tkinter import filedialog, Tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# Load pre-trained CNN model
def load_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Preprocess image for CNN input
def preprocess_image(image):
    resized_image = cv2.resize(image, (100, 100))
    return resized_image.reshape(1, 100, 100, 3) / 255.0

# Enhance image quality using CNN predictions for contrast adjustment
def enhance_image_contrast(image, contrast_prediction):
    # Adjust contrast based on CNN prediction
    if contrast_prediction > 0:
        alpha = 1.0 + contrast_prediction / 2
    else:
        alpha = 1.0 - abs(contrast_prediction) / 2

    # Apply contrast adjustment
    enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return enhanced_image

# Apply contrast adjustment
def adjust_contrast(image, alpha):
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return adjusted_image

# Apply gamma correction
def adjust_gamma(image, gamma):
    look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    adjusted_image = cv2.LUT(image, look_up_table)
    return adjusted_image

# Apply sharpness adjustment
def adjust_sharpness(image, strength=1.0):
    # Define the sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1, 9 + strength, -1],
                       [-1, -1, -1]])
    # Apply the sharpening kernel to the image
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

# Apply saturation adjustment
def adjust_saturation(image, saturation_scale):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image = np.array(hsv_image, dtype=np.float64)
    hsv_image[:, :, 1] = hsv_image[:, :, 1] * saturation_scale
    hsv_image[:, :, 1][hsv_image[:, :, 1] > 255] = 255
    hsv_image = np.array(hsv_image, dtype=np.uint8)
    adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return adjusted_image

# Apply hue adjustment
def adjust_hue(image, hue_shift):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180
    adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return adjusted_image

# Apply brightness adjustment
def adjust_brightness(image, brightness_scale):
    adjusted_image = cv2.convertScaleAbs(image, alpha=brightness_scale, beta=0)
    return adjusted_image

def open_file_dialog():
    
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        process_image(file_path)

        global original_image_path
        original_image_path = file_path

def process_image(file_path):
    global enhanced_image, image, contrast_prediction

    image = cv2.imread(file_path)
    if image is None:
        print("Error: Image not loaded. Please check the file path.")
        return
    
    contrast_prediction = model.predict(preprocess_image(image))[0][0]
    enhanced_image = enhance_image_contrast(image, contrast_prediction)
    update_image()

    # Enable sliders
    contrast_slider.configure(state='normal')
    gamma_slider.configure(state='normal')
    sharpness_slider.configure(state='normal')
    saturation_slider.configure(state='normal')
    hue_slider.configure(state='normal')
    brightness_slider.configure(state='normal')

def update_image():
    global enhanced_image

    # Adjust gamma, sharpness, saturation, hue, and brightness based on slider values
    gamma = gamma_slider.get()
    sharpness = sharpness_slider.get()
    saturation = saturation_slider.get()
    hue = hue_slider.get()
    brightness = brightness_slider.get()

    adjusted_image = adjust_gamma(enhanced_image, gamma)
    adjusted_image = adjust_sharpness(adjusted_image, sharpness)
    adjusted_image = adjust_saturation(adjusted_image, saturation)
    adjusted_image = adjust_hue(adjusted_image, hue)
    adjusted_image = adjust_brightness(adjusted_image, brightness)

    # Convert BGR image to RGB for display with PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    enhanced_image_rgb = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB)

    # Convert to PIL image
    original_image = Image.fromarray(image_rgb)
    enhanced_image_pil = Image.fromarray(enhanced_image_rgb)

    # Resize images to fit in the label
    original_image.thumbnail((400, 200))
    enhanced_image_pil.thumbnail((400, 200))

    # Convert to ImageTk
    original_image_tk = ImageTk.PhotoImage(original_image)
    enhanced_image_tk = ImageTk.PhotoImage(enhanced_image_pil)

    # Update labels
    label_original.configure(image=original_image_tk, text="")
    label_original.image = original_image_tk
    label_enhanced.configure(image=enhanced_image_tk, text="")
    label_enhanced.image = enhanced_image_tk

def load_images(original_path, edited_path):
    original = cv2.imread(original_path)
    edited = cv2.imread(edited_path)
    return original, edited

def plot_histogram_comparison(original, edited):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    edited_gray = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)

    hist_original = cv2.calcHist([original_gray], [0], None, [256], [0, 256])
    hist_edited = cv2.calcHist([edited_gray], [0], None, [256], [0, 256])

    plt.figure()
    plt.plot(hist_original, color='blue', label='Original')
    plt.plot(hist_edited, color='red', label='Edited')
    plt.title('Histogram Comparison')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_pixel_intensity_scatter(original, edited):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    edited_gray = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)

    original_flat = original_gray.flatten()
    edited_flat = edited_gray.flatten()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(original_flat, original_flat, alpha=0.5, s=0.1, color='blue')
    plt.title('Original Pixel Intensity')
    plt.xlabel('Original Intensity')
    plt.ylabel('Original Intensity')

    plt.subplot(1, 2, 2)
    plt.scatter(original_flat, edited_flat, alpha=0.5, s=0.1, color='red')
    plt.title('Edited Pixel Intensity')
    plt.xlabel('Original Intensity')
    plt.ylabel('Edited Intensity')

    plt.tight_layout()
    plt.show()

def plot_color_histogram_comparison(original, edited):
    colors = ('b', 'g', 'r')

    for i, color in enumerate(colors):
        hist_original = cv2.calcHist([original], [i], None, [256], [0, 256])
        hist_edited = cv2.calcHist([edited], [i], None, [256], [0, 256])
        
        plt.figure()
        plt.plot(hist_original, color=color, linestyle='dashed', label=f'Original {color}')
        plt.plot(hist_edited, color=color, label=f'Edited {color}')
        plt.title(f'Color Histogram Comparison ({color.upper()})')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

def plot_box_plot(original, edited):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    edited_gray = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)

    original_flat = original_gray.flatten()
    edited_flat = edited_gray.flatten()

    plt.figure()
    plt.boxplot([original_flat, edited_flat], labels=['Original', 'Edited'])
    plt.title('Box Plot of Pixel Intensities')
    plt.ylabel('Intensity')
    plt.show()

def plot_sharpness_comparison(original, edited):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    edited_gray = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)

    edges_original = cv2.Canny(original_gray, 100, 200)
    edges_edited = cv2.Canny(edited_gray, 100, 200)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(edges_original, cmap='gray')
    plt.title('Edges in Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(edges_edited, cmap='gray')
    plt.title('Edges in Edited Image')
    plt.axis('off')

    plt.show()

def plot_saturation_hue_comparison(original, edited):
    original_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    edited_hsv = cv2.cvtColor(edited, cv2.COLOR_BGR2HSV)

    hue_original, saturation_original, _ = cv2.split(original_hsv)
    hue_edited, saturation_edited, _ = cv2.split(edited_hsv)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.hist(hue_original.flatten(), bins=180, range=[0, 180], color='blue', alpha=0.5, label='Original Hue')
    plt.hist(hue_edited.flatten(), bins=180, range=[0, 180], color='red', alpha=0.5, label='Edited Hue')
    plt.title('Hue Histogram Comparison')
    plt.xlabel('Hue Value')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.hist(saturation_original.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.5, label='Original Saturation')
    plt.hist(saturation_edited.flatten(), bins=256, range=[0, 256], color='red', alpha=0.5, label='Edited Saturation')
    plt.title('Saturation Histogram Comparison')
    plt.xlabel('Saturation Value')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

def visualize(original_path, edited_path):
    original, edited = load_images(original_path, edited_path)
    plot_histogram_comparison(original, edited)
    plot_pixel_intensity_scatter(original, edited)
    plot_color_histogram_comparison(original, edited)
    plot_box_plot(original, edited)
    plot_sharpness_comparison(original, edited)
    plot_saturation_hue_comparison(original, edited)

def save_image():
    if enhanced_image is None:
        print("Error: No image to save. Please process an image first.")
        return
    
    # Convert enhanced_image to RGB format before saving
    adjusted_image = adjust_gamma(enhanced_image, gamma_slider.get())
    adjusted_image = adjust_sharpness(adjusted_image, sharpness_slider.get())
    adjusted_image = adjust_saturation(adjusted_image, saturation_slider.get())
    adjusted_image = adjust_hue(adjusted_image, hue_slider.get())
    adjusted_image = adjust_brightness(adjusted_image, brightness_slider.get())
    enhanced_image_rgb = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB)
    enhanced_image_pil = Image.fromarray(enhanced_image_rgb)
    
    file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
    if file_path:
        enhanced_image_pil.save(file_path)
        print(f"Image saved to {file_path}")

        # Visualize comparison
        visualize(original_image_path, file_path)

def drop(event):
    file_path = event.data.strip('{}')  # Remove braces if present
    process_image(file_path)

def schedule_update(func, delay=200):
    def wrapper(*args, **kwargs):
        if hasattr(wrapper, 'after_id'):
            root.after_cancel(wrapper.after_id)
        wrapper.after_id = root.after(delay, func, *args, **kwargs)
    return wrapper

update_image_with_delay = schedule_update(update_image)

def update_contrast(value):
    global enhanced_image
    alpha = float(value)
    enhanced_image = adjust_contrast(image, alpha)
    update_image_with_delay()

def update_gamma(value):
    update_image_with_delay()

def update_sharpness(value):
    update_image_with_delay()

def update_saturation(value):
    update_image_with_delay()

def update_hue(value):
    update_image_with_delay()

def update_brightness(value):
    update_image_with_delay()

def create_gui():
    global label_original, label_enhanced, enhanced_image, contrast_slider, gamma_slider, sharpness_slider, saturation_slider, hue_slider, brightness_slider, model, contrast_prediction, image, root

    enhanced_image = None
    image = None
    contrast_prediction = 0.0

    model = load_model()

    # Integrate customtkinter with TkinterDnD
    root = TkinterDnD.Tk()
    root.title("Image Enhancer")

    # Get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Set full screen
    root.geometry(f"{screen_width}x{screen_height}+0+0")

    # Bind escape key to exit fullscreen mode
    root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))

    # Main window using customtkinter
    ctk_frame = ctk.CTkFrame(master=root, fg_color="#ffffff", bg_color="#ffffff")
    ctk_frame.pack(fill="both", expand=True)

    # Configure grid layout
    ctk_frame.grid_rowconfigure(0, weight=1)
    ctk_frame.grid_columnconfigure(0, weight=7)  # Left section (70%)
    ctk_frame.grid_columnconfigure(1, weight=3)  # Right section (30%)

    # Left section for image input and output
    left_frame = ctk.CTkFrame(master=ctk_frame, fg_color="#ffffff", bg_color="#070F2B")
    left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    ctk.CTkLabel(left_frame, text="Original Image", fg_color="#ffffff", bg_color="#ffffff").grid(row=0, column=0, padx=10, pady=10)
    ctk.CTkLabel(left_frame, text="Enhanced Image", fg_color="#ffffff", bg_color="#ffffff").grid(row=0, column=1, padx=10, pady=10)

    label_original = ctk.CTkLabel(left_frame, text="", fg_color="#eef3f6", bg_color="#eef3f6")
    label_original.grid(row=1, column=0, padx=10, pady=10, sticky="new")

    label_enhanced = ctk.CTkLabel(left_frame, text="", fg_color="#eef3f6", bg_color="#eef3f6")
    label_enhanced.grid(row=1, column=1, padx=10, pady=10, sticky="new")

    # Make the labels expand with the window
    left_frame.grid_rowconfigure(1, weight=1)
    left_frame.grid_columnconfigure(0, weight=1)
    left_frame.grid_columnconfigure(1, weight=1)

    # Right section for sliders
    right_frame = ctk.CTkFrame(master=ctk_frame, fg_color="#ffffff", bg_color="#ffffff")
    right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

    button_frame = ctk.CTkFrame(right_frame, fg_color="#ffffff", bg_color="#ffffff")
    button_frame.pack(pady=10)

    button_open = ctk.CTkButton(button_frame, text="Select File", command=open_file_dialog, width=200, height=50, fg_color="#505b91", bg_color="#505b91")
    button_open.grid(row=0, column=0, padx=5)

    button_save = ctk.CTkButton(button_frame, text="Save Image", command=save_image, width=200, height=50, fg_color="#505b91", bg_color="#505b91")
    button_save.grid(row=0, column=1, padx=5)

    slider_frame = ctk.CTkFrame(right_frame, fg_color="#eef3f6", bg_color="#eef3f6")
    slider_frame.pack(pady=10, fill="x")

    ctk.CTkLabel(slider_frame, text="Contrast", fg_color="#eef3f6", bg_color="#eef3f6").pack(side="top", padx=10, pady=10)
    contrast_slider = ctk.CTkSlider(slider_frame, from_=0.5, to=3.0, number_of_steps=100, command=update_contrast, state='normal')
    contrast_slider.set(1.0)
    contrast_slider.pack(side="top", padx=10, pady=10)

    ctk.CTkLabel(slider_frame, text="Gamma", fg_color="#eef3f6", bg_color="#eef3f6").pack(side="top", padx=10, pady=10)
    gamma_slider = ctk.CTkSlider(slider_frame, from_=0.1, to=2.0, number_of_steps=100, command=update_gamma, state='normal')
    gamma_slider.set(1.0)
    gamma_slider.pack(side="top", padx=10, pady=10)
    
    ctk.CTkLabel(slider_frame, text="Sharpness", fg_color="#eef3f6", bg_color="#eef3f6").pack(side="top", padx=10, pady=10)
    sharpness_slider = ctk.CTkSlider(slider_frame, from_=0.1, to=2.0, number_of_steps=100, command=update_sharpness, state='normal')
    sharpness_slider.set(1.0)
    sharpness_slider.pack(side="top", padx=10, pady=10)

    ctk.CTkLabel(slider_frame, text="Saturation", fg_color="#eef3f6", bg_color="#eef3f6").pack(side="top", padx=10, pady=10)
    saturation_slider = ctk.CTkSlider(slider_frame, from_=0.5, to=2.0, number_of_steps=100, command=update_saturation, state='normal')
    saturation_slider.set(1.0)
    saturation_slider.pack(side="top", padx=10, pady=10)

    ctk.CTkLabel(slider_frame, text="Hue", fg_color="#eef3f6", bg_color="#eef3f6").pack(side="top", padx=10, pady=10)
    hue_slider = ctk.CTkSlider(slider_frame, from_=-90, to=90, number_of_steps=180, command=update_hue, state='normal')
    hue_slider.set(0)
    hue_slider.pack(side="top", padx=10, pady=10)

    ctk.CTkLabel(slider_frame, text="Brightness", fg_color="#eef3f6", bg_color="#eef3f6").pack(side="top", padx=10, pady=10)
    brightness_slider = ctk.CTkSlider(slider_frame, from_=0.5, to=2.0, number_of_steps=100, command=update_brightness, state='normal')
    brightness_slider.set(1.0)
    brightness_slider.pack(side="top", padx=10, pady=10)

    # Set default slider values
    default_contrast = (0.5 + 3.0) / 2  # Middle value between the minimum and maximum contrast
    default_gamma = (0.1 + 2.0) / 2  # Middle value between the minimum and maximum gamma
    default_sharpness = (0.1 + 2.0) / 2  # Middle value between the minimum and maximum sharpness
    default_saturation = (0.5 + 2.0) / 2  # Middle value between the minimum and maximum saturation
    default_hue = (-90 + 90) / 2  # Middle value between the minimum and maximum hue
    default_brightness = (0.5 + 2.0) / 2  # Middle value between the minimum and maximum brightness

    # Set default slider positions
    contrast_slider.set(default_contrast)
    gamma_slider.set(default_gamma)
    sharpness_slider.set(default_sharpness)
    saturation_slider.set(default_saturation)
    hue_slider.set(default_hue)
    brightness_slider.set(default_brightness)

    root.drop_target_register(DND_FILES)
    root.dnd_bind('<<Drop>>', drop)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
