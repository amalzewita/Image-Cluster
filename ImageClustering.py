# Required Libraries
import tkinter as tk  # GUI library
from tkinter import filedialog  # Dialog to select files
from PIL import ImageTk, Image  # Image processing
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Plotting
from sklearn.cluster import KMeans  # Clustering algorithm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input  # Deep Learning model and preprocessing
from tensorflow.keras.applications.resnet50 import ResNet50  # Another deep learning model
import random  # To select random items
import warnings  # To handle potential warnings

# Global variables to store image paths, clustered images, and default values for clusters and images
image_paths = []
clustered_images = []
num_clusters = 2
num_images = 3

# Function to process user input for number of clusters
def process_value():
    global num_clusters, entry1
    num_clusters = int(entry1.get())

# Function to change the number of images displayed per cluster
def change_number_of_images():
    global num_images, entry2
    num_images = int(entry2.get())

# Function to allow user to browse and select image files
def browse_files():
    global image_paths
    image_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png")])

# Function to cluster selected images
def cluster_images():
    global clustered_images, num_clusters

    # Return if no images selected
    if len(image_paths) == 0:
        return

    # Load pre-trained VGG16 model for image feature extraction
    model = VGG16(weights='imagenet', include_top=False)

    images = []
    for path in image_paths:
        img = Image.open(path)
        img = img.resize((224, 224))  # Resize image to fit model input size
        img_arr = np.array(img)
        img_arr = preprocess_input(img_arr)  # Pre-process image for model
        images.append(img_arr)

    # Convert list to numpy array and extract features using VGG16
    images = np.array(images)
    features = model.predict(images)
    features = features.reshape(features.shape[0], -1)

    # Use KMeans to cluster image features
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)
    labels = kmeans.labels_

    # Group images by their cluster label
    clustered_images = [[] for _ in range(num_clusters)]
    for i, label in enumerate(labels):
        clustered_images[label].append(image_paths[i])

    # Visualize the clusters
    visualize_clusters()

# Function to visualize clustered images
def visualize_clusters():
    global clustered_images

    # Return if no images to display
    if len(clustered_images) == 0:
        return

    # Plot images
    plt.figure(figsize=(12, 8))
    for i, images in enumerate(clustered_images):
        for j in range(1, num_images+1):
            plt.subplot(num_images, len(clustered_images), i+1+len(clustered_images)*(j-1))
            if j == 1:
                plt.title(f"Cluster {i+1} has {len(clustered_images[i])} images.")
            plt.axis('off')
            if len(images) > 0:
                img = Image.open(images[random.randint(0, len(clustered_images[i])-1)])  # Randomly select an image
                img = img.resize((200, 200))
                plt.imshow(img)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

# Function to sequence the operations
def sequence():
    process_value()
    change_number_of_images()
    cluster_images()

# Main GUI window
root = tk.Tk()
root.title("milestone 1")
root.geometry("300x200")

# GUI elements to get user inputs and display options
label = tk.Label(root, text="Enter number of clusters:")
label.pack()

entry1 = tk.Entry(root)
entry1.pack()

label = tk.Label(root, text=" ")
label.pack()

label = tk.Label(root, text="Select images to cluster:")
label.pack()

browse_button = tk.Button(root, text="Browse Files", command=browse_files)
browse_button.pack()

label = tk.Label(root, text=" ")
label.pack()

label = tk.Label(root, text="Enter number of images per cluster:")
label.pack()

entry2 = tk.Entry(root)
entry2.pack()

button = tk.Button(root, text="Set", command=sequence)
button.pack()

# Start the GUI event loop
root.mainloop()
