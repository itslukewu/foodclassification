import os
import cv2
import random
import glob
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import pathlib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
import shutil
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Flatten, MaxPooling2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras import regularizers
print(tf.config.experimental.list_physical_devices('GPU'))



# Define paths
root_folder = '/home/ubuntu/capstone/Food_Classification_dataset'
#===============================hyper-parameter==========================================
LR = 1e-3
batch_size = 32
DROPOUT = 0.5
num_classes = 5
random_state = 123
shuffle = True
epochs = 50
img_size = (224, 224)

# #=================================================================================
# # COUNT PLOT
# #=================================================================================
# Initialize lists to store folder names and image counts.
folder_names = []
image_counts = []
for folder_name in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder_name)

    # Check if the item is a directory.
    if os.path.isdir(folder_path):
        # Get a list of image files in the subfolder.
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        # Count the number of image files in the subfolder.
        image_count = len(image_files)

        # Append folder name and image count to the lists.
        folder_names.append(folder_name)
        image_counts.append(image_count)

# Sort the folders based on image counts in descending order.
sorted_folders = sorted(zip(folder_names, image_counts), key=lambda x: x[1], reverse=True)

folder_names, image_counts = zip(*sorted_folders)

# Create a bar chart to visualize the top 15 folders with the most images.
plt.figure(figsize=(24, 12))
plt.bar(folder_names, image_counts)
plt.xlabel('Folder Names')
plt.ylabel('Number of Images')
plt.title('number of images in each folder')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()
#=================================create target dataset for the top14 classes==========================
top5_classes = ["Hot Dog", "Sandwich", "Donut", "Crispy Chicken", "Taquito"]

top5_path = "/home/ubuntu/capstone/top5"
# Create the target directory if it doesn't exist
os.makedirs(top5_path, exist_ok=True)

# Loop through the subfolders and copy images of top 14 classes to the target folder
for food_class in top5_classes:
    source_folder = os.path.join(root_folder, food_class)
    target_folder = os.path.join(top5_path, food_class)

    # Create the target folder for the current food class
    os.makedirs(target_folder, exist_ok=True)

    # Copy images from the source folder to the target folder
    for filename in os.listdir(source_folder):
        source_file = os.path.join(source_folder, filename)
        target_file = os.path.join(target_folder, filename)

        # Copy the file
        shutil.copy(source_file, target_file)
# #====================================================
# # HANDLING CLASS IMBALANCE
# #====================================================
from collections import defaultdict
from shutil import copyfile
# Define the desired number of samples for each class
desired_samples_per_class = 1000
# Create a dictionary to store the file paths for each class
class_files = defaultdict(list)

# Walk through the dataset directory and collect file paths for each class
for root, dirs, files in os.walk(top5_path):
    for file in files:
        class_name = os.path.basename(root)
        class_files[class_name].append(os.path.join(root, file))

# Create a directory to store the downsampled dataset
downsampled_for5classes_path = "/home/ubuntu/capstone/downsampledfor5"
os.makedirs(downsampled_for5classes_path, exist_ok=True)

# Downsample all classes to the desired number of samples
for class_name, files in class_files.items():
    # Shuffle the files to randomize the selection
    random.seed(123)
    random.shuffle(files)
    selected_files = files[:desired_samples_per_class]
    class_path = os.path.join(downsampled_for5classes_path, class_name)
    os.makedirs(class_path, exist_ok=True)

    # Copy the selected files to the downsampled directory
    for file in selected_files:
        filename = os.path.basename(file)
        dest_path = os.path.join(class_path, filename)
        copyfile(file, dest_path)

# After the downsample process, count the number of images in each class
for class_name in os.listdir(downsampled_for5classes_path):
    class_folder = os.path.join(downsampled_for5classes_path, class_name)
    num_images = len(os.listdir(class_folder))
    print(f"Class '{class_name}' has {num_images} images.")

# have to remove the target dataset on terminal and re-run to show they are balanced.

#=====================visualizing the size of the downsampled dataset==========================================================================
# Initialize an empty dictionary to store the counts for each class
class_counts = {}

# Iterate through subdirectories (each subdirectory represents a class)
for class_name in os.listdir(downsampled_for5classes_path):
    class_folder = os.path.join(downsampled_for5classes_path, class_name)

    # Count the number of files in the class folder
    num_images = len(os.listdir(class_folder))

    # Store the count in the dictionary with the class name as the key
    class_counts[class_name] = num_images

# Extract class names and counts for plotting
class_names = list(class_counts.keys())
counts = list(class_counts.values())

# Create a bar plot
plt.figure(figsize=(12, 6))
plt.bar(class_names, counts)
plt.xlabel('Class Name')
plt.ylabel('Number of Images')
plt.title('Number of Images per Class in "downsampled" Directory')
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()
#==================================
# CHECKING SIZE
#==================================
uniform_shape = True
# Initialize variables to store the reference shape
reference_shape = None
shape_distribution = []
# Iterate through the subfolders (classes)
for class_name in os.listdir(downsampled_for5classes_path):
    class_path = os.path.join(downsampled_for5classes_path, class_name)

    if not os.path.isdir(class_path):
        continue  # Skip non-directory entries

    # Iterate through the images in the current class
    for filename in os.listdir(class_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path)

            if image is not None:
                current_shape = image.shape
                shape_distribution.append(current_shape)

                if reference_shape is None:
                    reference_shape = current_shape
                elif current_shape != reference_shape:
                    uniform_shape = False

if uniform_shape:
    print("All images have the same shape.")
else:
    print("Images have different shapes.")
#===============================Resize all the images=====================================================
img_size = (224, 224)  # Use the ResNet50 input size
# Output directory for resized images
resized_for5_path = "/home/ubuntu/capstone/resized_for5"

# Create the output directory if it doesn't exist
os.makedirs(resized_for5_path, exist_ok=True)

# Iterate through the subfolders (classes)
for class_name in os.listdir(downsampled_for5classes_path):
    class_path = os.path.join(downsampled_for5classes_path, class_name)

    if not os.path.isdir(class_path):
        continue  # Skip non-directory entries

    # Create a subdirectory in the output directory for the current class
    class_output_directory = os.path.join(resized_for5_path, class_name)
    os.makedirs(class_output_directory, exist_ok=True)

    # Iterate through the images in the current class
    for filename in os.listdir(class_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path)

            if image is not None:
                # Resize the image to (224, 224)
                resized_image = cv2.resize(image, (224, 224))

                # Save the resized image to the output directory
                output_path = os.path.join(class_output_directory, filename)
                cv2.imwrite(output_path, resized_image)

                current_shape = resized_image.shape
                shape_distribution.append(current_shape)

                if reference_shape is None:
                    reference_shape = current_shape
                elif current_shape != reference_shape:
                    uniform_shape = False

if uniform_shape:
    print("All images have the same shape (224, 224).")
else:
    print("Images have different shapes, and they have been resized to (224, 224).")
#==============================checking size on a random pic======================================
# Load the image
# img1 = plt.imread('/home/ubuntu/capstone/resized_for5/Donut/Donut \(495\).jpeg')
img1 = plt.imread(r'/home/ubuntu/capstone/resized_for5/Donut/Donut (495).jpeg')

# Get the shape of the image (rows, columns, channels)
shape = img1.shape
# Print the information
print(f"Shape: {shape}")

#==================================split the datatset===================================================
# Define the paths for the train, test, and validation folders
train_for5_path = '/home/ubuntu/capstone/train_for5'
test_for5_path = '/home/ubuntu/capstone/test_for5'
val_for5_path = '/home/ubuntu/capstone/val_for5'

# Define the split ratios (adjust as needed)
train_ratio = 0.7  # 70% for training
test_ratio = 0.15  # 15% for testing
val_ratio = 0.15  # 15% for validation

# Create the train, test, and validation directories if they don't exist
os.makedirs(train_for5_path, exist_ok=True)
os.makedirs(test_for5_path, exist_ok=True)
os.makedirs(val_for5_path, exist_ok=True)

# Loop through each food class
for food_class in os.listdir(resized_for5_path):
    class_folder = os.path.join(resized_for5_path, food_class)

    # Create subdirectories in train, test, and validation for the current class
    train_class_folder = os.path.join(train_for5_path, food_class)
    test_class_folder = os.path.join(test_for5_path, food_class)
    val_class_folder = os.path.join(val_for5_path, food_class)

    os.makedirs(train_class_folder, exist_ok=True)
    os.makedirs(test_class_folder, exist_ok=True)
    os.makedirs(val_class_folder, exist_ok=True)

    # Get a list of all image filenames in the class folder
    image_filenames = os.listdir(class_folder)

    # Shuffle the list of filenames to randomize the order
    random.shuffle(image_filenames)

    # Calculate the number of images for each split
    num_images = len(image_filenames)
    num_train = int(num_images * train_ratio)
    num_test = int(num_images * test_ratio)
    num_val = num_images - num_train - num_test

    # Split the images into train, test, and validation sets
    train_images = image_filenames[:num_train]
    test_images = image_filenames[num_train:num_train + num_test]
    val_images = image_filenames[num_train + num_test:]

    # Copy images to their respective split folders
    for image in train_images:
        src_path = os.path.join(class_folder, image)
        dst_path = os.path.join(train_class_folder, image)
        shutil.copy(src_path, dst_path)

    for image in test_images:
        src_path = os.path.join(class_folder, image)
        dst_path = os.path.join(test_class_folder, image)
        shutil.copy(src_path, dst_path)

    for image in val_images:
        src_path = os.path.join(class_folder, image)
        dst_path = os.path.join(val_class_folder, image)
        shutil.copy(src_path, dst_path)

print("Dataset splitting completed.")
#=================================load the split dataset=======================================================================
# Create the training dataset
trainfor5_ds = tf.keras.utils.image_dataset_from_directory(
    train_for5_path,
    labels='inferred',
    label_mode='categorical',  # Ensure labels are one-hot encoded
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True,

    interpolation='bilinear',
    color_mode='rgb'
)

# Create the validation dataset
valfor5_ds = tf.keras.utils.image_dataset_from_directory(
    val_for5_path,
    labels='inferred',
    label_mode='categorical',  # Ensure labels are one-hot encoded
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True,
    interpolation='bilinear',
    color_mode='rgb'
)

# Create the test dataset
testfor5_ds = tf.keras.utils.image_dataset_from_directory(
    test_for5_path,
    labels='inferred',
    label_mode='categorical',  # Ensure labels are one-hot encoded
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True,
    interpolation='bilinear',
    color_mode='rgb'
)
#====================================double check the balance of the dataset==================================
# List of valid image file extensions
valid_image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]

# Initialize a counter for the number of images
image_count = 0

# Loop through the files in the directory
for filename in os.listdir(train_for5_path):
    if os.path.isfile(os.path.join(train_for5_path, filename)):
        # Check if the file has a valid image extension
        if any(filename.lower().endswith(ext) for ext in valid_image_extensions):
            image_count += 1

print(f"Number of image files in the directory: {image_count}")

#===============================data augmentatuon=========================================
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
# Define data preprocessing for the validation set
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0
)

# Specify the target image size
target_image_size = (224, 224)

# Use the flow_from_directory method to apply the transformations to your dataset
train_generator = train_datagen.flow_from_directory(
    train_for5_path,  # Specify the directory for the training data
    target_size=target_image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_for5_path,  # Specify the directory for the validation data
    target_size=target_image_size,
    batch_size=batch_size,
    class_mode='categorical'
)



#==============================================================================
modelfor5_path='/home/ubuntu/capstone/modelfor5/'
# ModelCheckpoint callback
model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=modelfor5_path,
                                                      save_best_only=True,
                                                      save_weights_only=True)
# ReduceLROnPlateau callback
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.1,
                patience=10,
                verbose=1,
                mode='max',
                min_delta=0.0001,
                min_lr=1e-7
             )


# Load the pre-trained ResNet50 model
# base_model = tf.keras.applications.ResNet50(
#     weights='imagenet',
#     include_top=False,
#     input_shape=(224, 224, 3),
#     pooling='max'
# )
base_model = tf.keras.applications.efficientnet.EfficientNetB3(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
    pooling='max'
)
# base_model.trainable = False
base_model.trainable = True

# Define additional layers
flatten = tf.keras.layers.Flatten()
dense1 = tf.keras.layers.Dense(128, activation='relu')
dropout4 = tf.keras.layers.Dropout(0.4)

# Build the model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = flatten(x)
x = dense1(x)
x = dropout4(x)
output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=output)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy','AUC'],
    # class_weight=class_weight_dict,
)

# Train the model
history = model.fit(
    train_generator,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=validation_generator,
    callbacks=[lr_schedule, model_checkpoint_cb]
)