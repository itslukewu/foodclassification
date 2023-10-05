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

#===========================================================================================
# display random images
#===========================================================================================
# Iterate through the subfolders in the main directory.
# for folder_name in os.listdir(root_folder):
#     folder_path = os.path.join(root_folder, folder_name)
#
#     # Check if the item is a directory.
#     if os.path.isdir(folder_path):
#         # Get a list of image files in the subfolder.
#         image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
#
#         # Check if there are any image files in the subfolder.
#         if image_files:
#             # Choose a random image file.
#             random_image_file = random.choice(image_files)
#
#             # Get the full path to the random image file.
#             random_image_path = os.path.join(folder_path, random_image_file)
#
#             # Open and display the random image using Pillow (PIL).
#             img = Image.open(random_image_path)
#             plt.imshow(img)
#             plt.title(f"Random Image from '{folder_name}'")
#             plt.axis('off')
#             plt.show()
#         else:
#             print(f"No image files found in '{folder_name}'.")
# #=====================================================================================
# # IMAGE COUNT
# #=====================================================================================
# # Initialize a dictionary to store the count of image files for each folder.
# folder_image_counts = {}
# # Iterate through the subfolders in the main directory.
# for folder_name in os.listdir(root_folder):
#     folder_path = os.path.join(root_folder, folder_name)
#
#     # Check if the item is a directory.
#     if os.path.isdir(folder_path):
#         # Initialize a counter for image files in the folder.
#         image_count = 0
#
#         # Iterate through the files in the folder.
#         for filename in os.listdir(folder_path):
#             file_path = os.path.join(folder_path, filename)
#
#             # Check if the item is a file and has an image file extension (e.g., .jpg, .png, .jpeg).
#             if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.png', '.jpeg')):
#                 image_count += 1
#
#         # Store the image count in the dictionary with the folder name as the key.
#         folder_image_counts[folder_name] = image_count
#
# # Print the image counts for each folder.
# for folder, count in folder_image_counts.items():
#     print(f"Folder '{folder}' contains {count} image(s).")
#
# #=================================================================================
# # COUNT PLOT
# #=================================================================================
# # Initialize lists to store folder names and image counts.
# folder_names = []
# image_counts = []
#
# # Iterate through the subfolders in the main directory.
# for folder_name in os.listdir(root_folder):
#     folder_path = os.path.join(root_folder, folder_name)
#
#     # Check if the item is a directory.
#     if os.path.isdir(folder_path):
#         # Get a list of image files in the subfolder.
#         image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
#
#         # Count the number of image files in the subfolder.
#         image_count = len(image_files)
#
#         # Append folder name and image count to the lists.
#         folder_names.append(folder_name)
#         image_counts.append(image_count)
#
# # Create a bar chart to visualize the image counts.
# plt.figure(figsize=(12, 6))
# plt.bar(folder_names, image_counts)
# plt.xlabel('Folder Names')
# plt.ylabel('Number of Images')
# plt.title('Number of Images in Each Folder')
# plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
# plt.tight_layout()
#
# # Show the bar chart.
# plt.show()
#
# # Display the top 14 folders with the most images
# for folder_name in os.listdir(root_folder):
#     folder_path = os.path.join(root_folder, folder_name)
#
#     # Check if the item is a directory.
#     if os.path.isdir(folder_path):
#         # Get a list of image files in the subfolder.
#         image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
#
#         # Count the number of image files in the subfolder.
#         image_count = len(image_files)
#
#         # Append folder name and image count to the lists.
#         folder_names.append(folder_name)
#         image_counts.append(image_count)
#
# # Sort the folders based on image counts in descending order.
# sorted_folders = sorted(zip(folder_names, image_counts), key=lambda x: x[1], reverse=True)
#
# # Extract the top 14 folders and their image counts.
# top_14_folders = sorted_folders[:14]
#
# # Separate folder names and image counts.
# top_14_folder_names, top_14_image_counts = zip(*top_14_folders)
#
# # Create a bar chart to visualize the top 14 folders with the most images.
# plt.figure(figsize=(12, 6))
# plt.bar(top_14_folder_names, top_14_image_counts)
# plt.xlabel('Folder Names')
# plt.ylabel('Number of Images')
# plt.title('Top 14 Folders with the Most Images')
# plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
# plt.tight_layout()
#
# # Show the bar chart.
# plt.show()
#
# #====================================================
# # HANDLING CLASS IMBALANCE
# #===================================================
# import numpy as np
# from sklearn.utils.class_weight import compute_class_weight
# import tensorflow as tf
# from tensorflow.keras import layers, models
#
# # Example class labels and class counts (replace with your actual data)
# class_labels = ['Hot Dog', 'Sandwich','Donut', 'Crispy Chicken', 'Taquito', 'Taco', 'Fries', 'Baked Potato', 'chicken_curry', 'cheesecake', 'apple_pie', 'sushi',  'omelette',  'ice_cream']
# class_counts = [1548, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1050, 1050, 1049, 1048, 1048, 1000]
# # [('Hot Dog', 1548), ('Sandwich', 1500), ('Donut', 1500), ('Crispy Chicken', 1500), ('Taquito', 1500), ('Taco', 1500), ('Fries', 1500), ('Baked Potato', 1500), ('chicken_curry', 1050), ('cheesecake', 1050), ('apple_pie', 1049), ('sushi', 1048), ('omelette', 1048), ('ice_cream', 1000)]
#
# # Compute class weights using sklearn
# class_weights = compute_class_weight("balanced", np.unique(class_labels), class_labels)
#
# # Convert class weights to a dictionary
# class_weight_dict = dict(zip(np.unique(class_labels), class_weights))

#===================================================================================
# moodel
#===================================================================================
# Define top 14 classes
top_14_classes = ["Hot Dog", "Sandwich", "Donut", "Crispy Chicken", "Taquito", "Taco", "Fries", "Baked Potato",
                  "chicken_curry", "cheesecake", "apple_pie", "sushi", "omelette", "ice_cream"]

# Define paths
root_folder = '/home/ubuntu/capstone/Food_Classification_dataset'
target_dataset_path = "/home/ubuntu/capstone/target"

# Create the target directory if it doesn't exist
os.makedirs(target_dataset_path, exist_ok=True)

# Loop through the subfolders and copy images of top 14 classes to the target folder
for food_class in top_14_classes:
    source_folder = os.path.join(root_folder, food_class)
    target_folder = os.path.join(target_dataset_path, food_class)

    # Create the target folder for the current food class
    os.makedirs(target_folder, exist_ok=True)

    # Copy images from the source folder to the target folder
    for filename in os.listdir(source_folder):
        source_file = os.path.join(source_folder, filename)
        target_file = os.path.join(target_folder, filename)

        # Copy the file
        shutil.copy(source_file, target_file)

# # Create the "others" class
# others_folder = os.path.join(target_dataset_path, "others")
# os.makedirs(others_folder, exist_ok=True)

# # Loop through all subfolders in the root folder that are not in the top 14 classes
# for folder_name in os.listdir(root_folder):
#     if folder_name not in top_14_classes:
#         source_folder = os.path.join(root_folder, folder_name)
#
#         # Copy images from the source folder to the "others" folder
#         for filename in os.listdir(source_folder):
#             source_file = os.path.join(source_folder, filename)
#             target_file = os.path.join(others_folder, filename)
#
#             # Copy the file
#             shutil.copy(source_file, target_file)

print("Dataset creation completed.")
#=====================visualizing the size of the target dataset==========================================================================
# Initialize an empty dictionary to store the counts for each class
class_counts = {}

# Iterate through subdirectories (each subdirectory represents a class)
for class_name in os.listdir(target_dataset_path):
    class_folder = os.path.join(target_dataset_path, class_name)

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
plt.title('Number of Images per Class in "target" Directory')
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for readability
plt.tight_layout()

# Show the plot
plt.show()
#==================================
# CHECKING SIZE
#==================================
# # Initialize variables to store the reference values
# reference_shape = (168, 300, 3)
# reference_channels = 3
# Initialize a flag to track uniformity
uniform_shape = True
# Initialize variables to store the reference shape
reference_shape = None
shape_distribution = []

# Iterate through the subfolders (classes)
for class_name in os.listdir(target_dataset_path):
    class_path = os.path.join(target_dataset_path, class_name)

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

# Plot the distribution of image shapes
shape_distribution = [str(shape) for shape in shape_distribution]
plt.hist(shape_distribution, bins=50)
plt.xlabel("Image Shapes")
plt.ylabel("Frequency")
plt.title("Distribution of Image Shapes")
plt.xticks(rotation=45)
plt.show()

#===============================Resize all the images=====================================================
img_size = (224, 224)  # Use the ResNet50 input size
# Output directory for resized images
resized_img_path = "/home/ubuntu/capstone/resized_images"

# Create the output directory if it doesn't exist
os.makedirs(resized_img_path, exist_ok=True)

# Iterate through the subfolders (classes)
for class_name in os.listdir(target_dataset_path):
    class_path = os.path.join(target_dataset_path, class_name)

    if not os.path.isdir(class_path):
        continue  # Skip non-directory entries

    # Create a subdirectory in the output directory for the current class
    class_output_directory = os.path.join(resized_img_path, class_name)
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

# # Plot the distribution of image shapes
# shape_distribution = [str(shape) for shape in shape_distribution]
# plt.hist(shape_distribution, bins=50)
# plt.xlabel("Image Shapes")
# plt.ylabel("Frequency")
# plt.title("Distribution of Resized Image Shapes (224, 224)")
# plt.xticks(rotation=45)
# plt.show()
#==============================checking size on a random pic======================================
# Load the image
img1 = plt.imread('/home/ubuntu/capstone/resized_images/sushi/953649.jpg')
# Get the shape of the image (rows, columns, channels)
shape = img1.shape
# Print the information
print(f"Shape: {shape}")

#==================================================================================================
# Define the paths for the train, test, and validation folders
train_path = '/home/ubuntu/capstone/train'
test_path = '/home/ubuntu/capstone/test'
val_path = '/home/ubuntu/capstone/val'

# Define the split ratios (adjust as needed)
train_ratio = 0.7  # 70% for training
test_ratio = 0.15  # 15% for testing
val_ratio = 0.15  # 15% for validation

# Create the train, test, and validation directories if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

# Loop through each food class
for food_class in os.listdir(resized_img_path):
    class_folder = os.path.join(resized_img_path, food_class)

    # Create subdirectories in train, test, and validation for the current class
    train_class_folder = os.path.join(train_path, food_class)
    test_class_folder = os.path.join(test_path, food_class)
    val_class_folder = os.path.join(val_path, food_class)

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
#================================plot the number of images of each class in train===================================
# Initialize an empty dictionary to store the counts for each class
class_counts = {}

# Iterate through subdirectories (each subdirectory represents a class)
for class_name in os.listdir(train_path):
    class_folder = os.path.join(train_path, class_name)

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
plt.title('Number of Images per Class in Training Dataset')
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for readability
plt.tight_layout()

# Show the plot
plt.show()
# %% ----------------------------------- Load the data --------------------------------------------------------------
LR = 1e-3
N_EPOCHS = 30
batch_size = 32
DROPOUT = 0.5
num_classes = 14
random_state = 123
shuffle = True
epochs = 50

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    labels='inferred',
    batch_size=batch_size,
    image_size=img_size,
    shuffle= True,
    interpolation='bilinear',
    color_mode='rgb'
)
# Create the validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_path,
    labels='inferred',
    batch_size=batch_size,
    image_size=img_size,
    shuffle= True,
    interpolation='bilinear',
    color_mode='rgb'
)

# Create the test dataset (complementary to training and validation)
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
    labels='inferred',
    batch_size=batch_size,
    image_size=img_size,
    shuffle= True,
    interpolation='bilinear',
    color_mode='rgb'
)

#===================================DATA AUGMENTATION========================================================
# RandomState, RandomBrightness, RandomContrast
data_augmentation = tf.keras.Sequential([
  # RandomFlip(mode='horizontal', seed=random_state),
  RandomRotation(factor=[-0.5, 0.5], fill_mode='constant', fill_value=0, interpolation='bilinear'),
  RandomBrightness(factor=[0, 0.3], value_range=[0, 1.], seed=random_state),
  RandomContrast(factor=[0, 0.3], seed=random_state),
  # GaussianNoise(stddev=tf.math.sqrt(0.05), seed=random_state)
])


#=========================================Define evaluation function===========================================================


#=========================================Define callbacks(check points, early stopping and lr schedule function)================
# Where to store the best model and its weights
model_path='/home/ubuntu/capstone/model/'

# ModelCheckpoint callback
model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
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

#=========================================Define base model and save=================================================================
# Load a pretrained model (ResNet50)
from tensorflow.keras.applications.resnet50 import ResNet50

base_model = ResNet50(weights='imagenet',
                      include_top=False,
                      input_shape=(224, 224, 3),
                      pooling='max')

base_model.trainable = False
base_model.summary()
#=============================define new layers===================================================
# Get the output tensor from the convolutional base
# x = base_model.output
flatten = Flatten()

# add a dense layer
dense1 = Dense(128, activation='relu')

# add Dropout layer
# dropout2 = tf.keras.layers.Dropout(0.2)
dropout4 = tf.keras.layers.Dropout(0.4)

#====================================================================
# inputs = tf.keras.Input(shape=(256, 256, 1))
x = base_model(inputs=base_model.input, training=False)
x = flatten(x)
x = dense1(x)
x = dropout4(x)
output = Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

# x = Dense(256, kernel_regularizer=regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006),
#           bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
# x = Dropout(rate=.4, seed=123)(x)
# view the full model summary
model.summary()
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=val_ds,
                    callbacks=[lr_schedule, model_checkpoint_cb])

# model.save('bestmodel')
#======================================Plot the accuracy and loss curve======================================================




#=======================================Plot the confusion matrix=========================================================



#===========================================Define an evaluative criteria to judge the model==================

# =====================================Define a function to plot the training data==================================================


#=======================================Define a function to make prediction=======================================================



# print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")












#===============================================================================================
# model2 = Sequential([
#                 InputLayer(input_shape=[256, 256, 1], batch_size=batch_size),
#                 Rescaling(1./255),
#                 data_augmentation,
#                 BatchNormalization(),
#                 Conv2D(filters=32, kernel_size=5, activation='relu'), # 252x252
#                 MaxPool2D(), # 126x126
#                 BatchNormalization(),
#                 Conv2D(filters=64, kernel_size=3, activation='relu'), # 124x124
#                 MaxPool2D(), # 62x62
#                 BatchNormalization(),
#                 Conv2D(filters=128, kernel_size=3, activation='relu'), # 60x60
#                 MaxPool2D(), # 30x30
#                 BatchNormalization(),
#                 Flatten(),
#                 Dense(128, activation='relu'),
#                 BatchNormalization(),
#                 Dense(10, activation='relu'),
#                 BatchNormalization(),
#                 Dense(1, activation='sigmoid')
#             ])
#
# model2.summary()

# # Create augmented datasets
# AUTOTUNE = tf.data.AUTOTUNE

# # Assuming 'train' is a list of image paths and 'train_labels' is a list of labels
# train = ['/home/ubuntu/capstone/target/train']  # Replace with your image paths
# # labels =  ["Hot Dog", "Sandwich", "Donut", "Crispy Chicken", "Taquito", "Taco", "Fries", "Baked Potato",
# #           "chicken_curry", "cheesecake", "apple_pie", "sushi", "omelette", "ice_cream"]
#
# # Convert the lists to TensorFlow tensors
# train = tf.constant(train)
# train_labels = tf.constant(labels)
#
# # Create a dataset from the tensors
# dataset = tf.data.Dataset.from_tensor_slices((train, train_labels))



# # Initialize empty lists to store image paths and labels
# image_paths = []
# train_labels = []
#
# # List all subdirectories (class folders) in the root directory
# class_folders = os.listdir(train_path)
#
# # Create a dictionary to map class names to integer labels
# class_to_label = {class_name: label for label, class_name in enumerate(class_folders)}
#
# # Iterate through class folders
# for class_folder in class_folders:
#     class_path = os.path.join(train_path, class_folder)
#
#     # List all files (images) in the class folder
#     class_images = os.listdir(class_path)
#
#     # Collect image paths and labels
#     image_paths.extend([os.path.join(class_path, image) for image in class_images])
#     train_labels.extend([class_to_label[class_folder]] * len(class_images))
#
# # Now, 'image_paths' contains the paths to all your images, and 'labels' contains the corresponding labels
#
# batch_size = 32
# AUTOTUNE = tf.data.AUTOTUNE
#

#
# custom_layers = tf.keras.Sequential([
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dense(num_classes, activation='softmax')
# ])
#
# model = tf.keras.Sequential([
#     base_model,
#     custom_layers
# ])
#
# for layer in base_model.layers:
#     layer.trainable = False
#
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# # Train the model
# model.fit(augmented_train_dataset, epochs=epochs, validation_data=val_dataset)
#
# # Save the model if needed
# model.save("/path/to/saved/model")


# You can proceed to split and augment this dataset as needed.

# # Get folder names and image counts
# folder_names = []
# image_counts = []
# for folder_name in os.listdir(root_folder):
#     folder_path = os.path.join(root_folder, folder_name)
#
#     # Check if the item is a directory.
#     if os.path.isdir(folder_path):
#         # Get a list of image files in the subfolder.
#         image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
#
#         # Count the number of image files in the subfolder.
#         image_count = len(image_files)
#
#         # Append folder name and image count to the lists.
#         folder_names.append(folder_name)
#         image_counts.append(image_count)
#
# # Sort the folders based on image counts in descending order.
# sorted_folders = sorted(zip(folder_names, image_counts), key=lambda x: x[1], reverse=True)
#
# # Extract the top 14 folders and their image counts.
# top_14_folders = sorted_folders[:14]
#
# # Separate folder names and image counts.
# top_14_folder_names, top_14_image_counts = zip(*top_14_folders)
#
# # Create a new dataset with the top 14 folders and their respective paths.
# top_14_dataset = []
#
# for folder_name in top_14_folder_names:
#     folder_path = os.path.join(root_folder, folder_name)
#     top_14_dataset.append((folder_name, folder_path))
#
# # Print the top 14 folder names and image counts
# for folder_name, image_count in top_14_dataset:
#     print(f"Folder Name: {folder_name}, Image Count: {image_count}")


# # Initialize variables to store the dimensions of the first image.
# first_image_width = None
# first_image_height = None
#
# # Flag to track if all images have the same size.
# all_images_same_size = True
#
# # Iterate through the files in the folder.
# for filename in os.listdir(root_folder):
#     file_path = os.path.join(root_folder, filename)
#
#     # Check if the item is a file and has a recognized image file extension.
#     if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.png', '.jpeg')):
#         # Open the image using Pillow (PIL).
#         img = Image.open(file_path)
#
#         # Get the dimensions (width and height) of the image.
#         width, height = img.size
#
#         # If this is the first image, store its dimensions.
#         if first_image_width is None:
#             first_image_width = width
#             first_image_height = height
#         else:
#             # Compare the dimensions of the current image to the first image.
#             if width != first_image_width or height != first_image_height:
#                 all_images_same_size = False
#                 break  # Exit the loop early if a different-sized image is found.
#
# # Check if all images have the same size and print the result.
# if all_images_same_size:
#     print("All images have the same size.")
#     print(f"Image size: {first_image_width} x {first_image_height}")
# else:
#     print("Not all images have the same size.")

# data = []
# labels = []
# root_folder = '/home/ubuntu/capstone/Food_Classification_dataset'
#
# for folder_name in os.listdir(root_folder):
#     folder_path = os.path.join(root_folder, folder_name)
#     if os.path.isdir(folder_path):
#         for img_name in os.listdir(folder_path):
#             img_path = os.path.join(folder_path, img_name)
#             img = cv2.imread(img_path)
#             if img is not None:
#                 data.append(img)
#                 labels.append(folder_name)
#
# num_samples = len(data)
# print(f"Total number of samples: {num_samples}")
#
#
# def check_image_shapes(dataset_path):
#     # List all files in the dataset directory
#     image_files = os.listdir(dataset_path)
#
#     for image_file in image_files:
#         # Construct the full path to the image
#         image_path = os.path.join(dataset_path, image_file)
#
#         # Load the image using OpenCV
#         image = cv2.imread(image_path)
#
#         # Get the shape of the image (height, width, channels)
#         shape = image.shape
#
#         print(f"Image {image_file}: Shape = {shape}")
#
# # Provide the path to your dataset directory
# dataset_path = "path_to_your_dataset_directory"
# check_image_shapes(dataset_path)
#
#
#
#
# #=========================================
# # CREATE A DATASET FOR THE 14 FOOD CLASS
# #=========================================
# # folder_names = []
# # image_counts = []
# import shutil
#
# # Define the top 14 food classes
# top_15_classes = ["hot_dog", "sandwich", "donut", "crispy_chicken", "taquito", "taco", "fries", "baked_potato", "chicken_curry", "cheesecake", "apple_pie", "sushi", "omelets", "ice_cream", "others"]
#
#
# # Define paths
# original_dataset_path = root_folder
# target_dataset_path = "/home/ubuntu/capstone/target"
#
# # Create subfolders for each class in the target dataset
# for class_name in top_15_classes:
#     class_dir = os.path.join(target_dataset_path, class_name)
#     os.makedirs(class_dir, exist_ok=True)
#
# # Iterate through the original dataset and filter images by class
# for root, _, files in os.walk(original_dataset_path):
#     for file in files:
#         if file.lower().endswith(('.jpg', '.png', '.jpeg')):
#             image_path = os.path.join(root, file)
#             class_name = root.split(os.path.sep)[-1]
#
#             if class_name in top_15_classes:
#                 # Copy the image to the corresponding class folder in the target dataset
#                 target_dir = os.path.join(target_dataset_path, class_name)
#                 shutil.copy(image_path, target_dir)
#
# #=======================================================
# # SPLITTING DATASET
# #=======================================================
# # Define paths
# target_dataset_path = target_dataset_path
#
# # List all image files and their corresponding labels
# all_image_files = []
# all_labels = []
#
# for class_name in top_15_classes:
#     class_dir = os.path.join(target_dataset_path, class_name)
#     class_images = [os.path.join(class_dir, file) for file in os.listdir(class_dir)]
#     all_image_files.extend(class_images)
#     all_labels.extend([class_name] * len(class_images))
#
# # Split the dataset
# train_files, test_files, train_labels, test_labels = train_test_split(
#     all_image_files, all_labels, test_size=0.15, random_state=42)
# train_files, val_files, train_labels, val_labels = train_test_split(
#     train_files, train_labels, test_size=0.15, random_state=42)
# # Renaming the subsets
# train = (train_files, train_labels)
# test = (test_files, test_labels)
# val = (val_files, val_labels)
#
#
#
#
# #======================================================================================================================
# # CHECKING SIZE
# #===========================================================================================================
#
# #======================================================================================================================
# # DATA AUGMENTATION
# #===========================================================================================================
# IMG_SIZE = (224, 224)
# def augment_image(image, label, img_size=IMG_SIZE):
#     # Cast the image to float32.
#     image = tf.cast(image, tf.float32)
#
#     # Resize the image to the specified size.
#     image = tf.image.resize(image, [img_size, img_size])
#
#     # Normalize pixel values to the range [0, 1].
#     image = image / 255.0
#
#     # Apply random cropping to the image.
#     image = tf.image.random_crop(image, size=[img_size, img_size, 3])
#
#     # Apply random brightness adjustment.
#     image = tf.image.random_brightness(image, max_delta=0.5)
#
#     # Apply random saturation adjustment.
#     image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  # Adjust saturation range as needed
#
#     # Apply random rotation adjustment.
#     random_angle = tf.random.uniform(shape=[], minval=-30, maxval=30)  # Adjust rotation range as needed
#     image = tf.image.stateless_random_flip_up_down(image, seed=None)
#     image = tf.image.rot90(image, k=tf.cast(random_angle / 90, tf.int32))
#
#     return image, label
#
# # Assuming you have a training dataset named 'train_dataset' and batch size 'batch_size'
#
# # Apply data augmentation to the training dataset.
# batch_size = 32
# augmented_train_dataset = (
#     train
#     .shuffle(1000)  # Optional: Shuffle the dataset
#     .map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
#     .batch(batch_size)
#     .prefetch(tf.data.AUTOTUNE)
# )
#
#
# #================================================================
# # CNN
# #=================================================================
# from tensorflow.keras.applications import ResNet50
#
# epochs = 50
# num_classes = 15
# # Load the pretrained ResNet-50 model (with weights pre-trained on ImageNet)
# base_model = ResNet50(weights='imagenet', include_top=False)
#
# # Add custom layers on top of the pretrained model
# # You can define your own classification layers based on your task
# custom_layers = tf.keras.Sequential([
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dense(num_classes, activation='softmax')  # num_classes is the number of classes in your task
# ])
#
# # Combine the pretrained base model and custom layers
# model = tf.keras.Sequential([
#     base_model,
#     custom_layers
# ])
#
# # Optional: Fine-tune specific layers if needed
# for layer in base_model.layers:
#     layer.trainable = False
#
# # Compile the model with an optimizer, loss function, and metrics
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',  # Use the appropriate loss function
#               metrics=['accuracy'],class_weight=class_weight_dict)
#
# # Train your model using the class-weighted loss
# model.fit(train, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(val_data, val_labels))
#
#
# augmented_train_dataset = (
#     tf.data.Dataset.from_tensor_slices((train_images, train_labels))
#     .shuffle(1000)
#     .map(augment_image, num_parallel_calls=AUTOTUNE)
#     .batch(batch_size)
#     .prefetch(AUTOTUNE)
# )
#
# val_dataset = (
#     tf.data.Dataset.from_tensor_slices((val_files, val_labels))
#     .map(augment_image, num_parallel_calls=AUTOTUNE)
#     .batch(batch_size)
#     .prefetch(AUTOTUNE)
# )

# def augment_image(image, label, img_size=IMG_SIZE):
#     # Cast the image to float32.
#     image = tf.cast(image, tf.float32)
#     #
#     # # Resize the image to the specified size.
#     image = tf.image.resize(image, [img_size, img_size, 3])
#     #
#     # # Normalize pixel values to the range [0, 1].
#     image = image / 255.0
#
#     # Apply random cropping to the image.
#     image = tf.image.random_crop(image, size=[img_size, img_size, 3])
#
#     # Apply random brightness adjustment.
#     image = tf.image.random_brightness(image, max_delta=0.5)
#
#     # Apply random saturation adjustment.
#     image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
#
#     # Apply random rotation adjustment.
#     random_angle = tf.random.uniform(shape=[], minval=-30, maxval=30)
#     image = tf.image.stateless_random_flip_up_down(image, seed=None)
#     image = tf.image.rot90(image, k=tf.cast(random_angle / 90, tf.int32))
#
#     return image, label