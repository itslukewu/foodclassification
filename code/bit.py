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
# import ipython
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Flatten, MaxPooling2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
# from tensorflow.keras import regularizers
# from IPython.display import clear_output as cls

# Data Loading
from keras.preprocessing.image import ImageDataGenerator as IDG
# Model
from keras.models import Sequential
from tensorflow_hub import KerasLayer as KL
from keras.layers import Dense, InputLayer, Dropout

# Optimizer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay as PwCD

# Callbacks
from keras.callbacks import EarlyStopping as ES, ModelCheckpoint as MC
print(tf.config.experimental.list_physical_devices('GPU'))

# Define paths
root_folder = '/home/ubuntu/capstone/Food_Classification_dataset'
#===============================hyper-parameter==========================================
batch_size = 32
DROPOUT = 0.5
n_classes = 14
random_state = 123
shuffle = True
epochs = 25
img_size = (256, 256)


top_14_classes = ["Hot Dog", "Sandwich", "Donut", "Crispy Chicken", "Taquito", "Taco", "Fries", "Baked Potato",
                  "chicken_curry", "cheesecake", "apple_pie", "sushi", "omelette", "ice_cream"]

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

# #=================================================================================
# # COUNT PLOT
# #=================================================================================
# Initialize lists to store folder names and image counts.
folder_names = []
image_counts = []
for folder_name in os.listdir(target_dataset_path):
    folder_path = os.path.join(target_dataset_path, folder_name)

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

# Extract the top 15 folders and their image counts.
# top_15_folders = sorted_folders[:15]

# Separate folder names and image counts.
# top_15_folder_names, top_15_image_counts = zip(*top_15_folders)
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
for root, dirs, files in os.walk(target_dataset_path):
    for file in files:
        class_name = os.path.basename(root)
        class_files[class_name].append(os.path.join(root, file))

# Create a directory to store the downsampled dataset
downsampled_path = "/home/ubuntu/capstone/downsampled"
os.makedirs(downsampled_path, exist_ok=True)

# Downsample all classes to the desired number of samples
for class_name, files in class_files.items():
    # Shuffle the files to randomize the selection
    random.seed(123)
    random.shuffle(files)
    selected_files = files[:desired_samples_per_class]
    class_path = os.path.join(downsampled_path, class_name)
    os.makedirs(class_path, exist_ok=True)

    # Copy the selected files to the downsampled directory
    for file in selected_files:
        filename = os.path.basename(file)
        dest_path = os.path.join(class_path, filename)
        copyfile(file, dest_path)

# After the downsample process, count the number of images in each class
for class_name in os.listdir(downsampled_path):
    class_folder = os.path.join(downsampled_path, class_name)
    num_images = len(os.listdir(class_folder))
    print(f"Class '{class_name}' has {num_images} images.")

# have to remove the target dataset on terminal and re-run to show they are balanced.

#=====================visualizing the size of the downsampled dataset==========================================================================
# Initialize an empty dictionary to store the counts for each class
class_counts = {}

# Iterate through subdirectories (each subdirectory represents a class)
for class_name in os.listdir(downsampled_path):
    class_folder = os.path.join(downsampled_path, class_name)

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
for class_name in os.listdir(downsampled_path):
    class_path = os.path.join(downsampled_path, class_name)

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
# img_size = (224, 224)  # Use the ResNet50 input size
# Output directory for resized images
resized_img_path = "/home/ubuntu/capstone/resized_images"

# Create the output directory if it doesn't exist
os.makedirs(resized_img_path, exist_ok=True)

# Iterate through the subfolders (classes)
for class_name in os.listdir(downsampled_path):
    class_path = os.path.join(downsampled_path, class_name)

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
                resized_image = cv2.resize(image, (256, 256))

                # Save the resized image to the output directory
                output_path = os.path.join(class_output_directory, filename)
                cv2.imwrite(output_path, resized_image)

                current_shape = resized_image.shape
                shape_distribution.append(current_shape)

                if reference_shape is None:
                    reference_shape = current_shape
                elif current_shape != reference_shape:
                    uniform_shape = False


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
    print(food_class)
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
#==============================checking size on a random pic======================================
# Load the image
img1 = plt.imread('/home/ubuntu/capstone/resized_images/sushi/953649.jpg')
# Get the shape of the image (rows, columns, channels)
shape = img1.shape
# Print the information
print(f"Shape: {shape}")

#=================================load the split dataset=======================================================================
# Create the training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    labels='inferred',
    label_mode='categorical',  # Ensure labels are one-hot encoded
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True,

    interpolation='bilinear',
    color_mode='rgb'
)

# Create the validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_path,
    labels='inferred',
    label_mode='categorical',  # Ensure labels are one-hot encoded
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True,
    interpolation='bilinear',
    color_mode='rgb'
)

# Create the test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
    labels='inferred',
    label_mode='categorical',  # Ensure labels are one-hot encoded
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True,
    interpolation='bilinear',
    color_mode='rgb'
)

from skimage import exposure, img_as_ubyte
from skimage.filters import unsharp_mask
from skimage.restoration import denoise_tv_chambolle
# Function for contrast stretching
def contrast_stretching(img):
    p2, p98 = np.percentile(img, (2, 98))
    return exposure.rescale_intensity(img, in_range=(p2, p98))

# Function for noise reduction using Gaussian filtering
def reduce_noise(img):
    return denoise_tv_chambolle(img)

# Function for sharpness enhancement using unsharp masking
def enhance_sharpness(img):
    return img_as_ubyte(unsharp_mask(img))

# Combined preprocessing function
def custom_preprocessing(img):
    img = contrast_stretching(img)
    # img = adjust_brightness()
    img = reduce_noise(img)
    img = enhance_sharpness(img)
    return img

# Setting up ImageDataGenerator with custom preprocessing functions
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    brightness_range=[0.5, 1.5], # Example of adjusting brightness within a range
    preprocessing_function=custom_preprocessing  # Combined custom preprocessing function
)

# Define data preprocessing for the validation set
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0
)

# Specify the target image size
target_image_size = (256, 256)

# Use the flow_from_directory method to apply the transformations to your dataset
train_generator = train_datagen.flow_from_directory(
    train_path,  # Specify the directory for the training data
    target_size=target_image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_path,  # Specify the directory for the validation data
    target_size=target_image_size,
    batch_size=batch_size,
    class_mode='categorical'
)


# Model URL
url = "https://tfhub.dev/google/bit/m-r50x1/1"

# Load Model
bit = KL(url)
from tensorflow.keras import regularizers
# Model Name
model_name = "Fast-Food-Classification-BiT"

# Model Architecture
model_bit = Sequential([
    InputLayer(input_shape=(256, 256, 3)),
    bit,
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
    Dense(256, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006), bias_regularizer=regularizers.l1(0.006) ,activation='relu'),
    Dropout(0.3),
    Dense(n_classes, activation='softmax', kernel_initializer='zeros')
], name=model_name)




# Model Summary
model_bit.summary()


# Learning Rate Scheduler
lr = 5e-3

lr_scheduler = PwCD(boundaries=[30,50,80],values=[lr*0.1, lr*0.01, lr*0.001, lr*0.0001])

# Compile
model_bit.compile(
    # loss='sparse_categorical_crossentropy',
    loss = 'categorical_crossentropy',
    # optimizer='adam',
    optimizer='sgd',
    metrics=['accuracy']
)

# Callbacks
cbs = [ES(patience=5, restore_best_weights=True), MC(model_name+".h5", save_best_only=True)]

# Training
history = model_bit.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=cbs)

# Evaluate the model on the testing dataset
test_loss, test_accuracy = model_bit.evaluate(test_ds)

# Print the testing accuracy
print(f'Testing accuracy: {test_accuracy}')


from tensorflow.keras.preprocessing import image

category = {
    0: ['Hot Dog', 'Hot Dog'], 1: ['Sandwich', 'Sandwich'], 2: ['Donut', 'Donut'],
    3: ['Crispy Chicken', 'Crispy Chicken'], 4: ['Taquito', 'Taquito'], 5: ['Taco', 'Taco'],
    6: ['Fries', 'Fries'], 7: ['Baked Potato', 'Baked Potato'], 8: ['chicken_curry', 'Chicken Curry'],
    9: ['cheesecake', 'Cheesecake'], 10: ['apple_pie', 'Apple Pie'], 11: ['sushi', 'Sushi'],
    12: ['omelette', 'Omelette'], 13: ['ice_cream', 'Ice Cream']
}


# Load and preprocess the image
online_img1_path = '/home/ubuntu/foodclassification/online_food_images/sandwich.png'
online_img2_path = '/home/ubuntu/foodclassification/online_food_images/apple_pie.png'
online_img3_path = '/home/ubuntu/foodclassification/online_food_images/cheesecake.png'
online_img4_path = '/home/ubuntu/foodclassification/online_food_images/chicken_curry.png'
online_img5_path = '/home/ubuntu/foodclassification/online_food_images/Donut.png'
online_img6_path = '/home/ubuntu/foodclassification/online_food_images/Fries.png'
online_img7_path = '/home/ubuntu/foodclassification/online_food_images/apple_pie1.png'
online_img8_path = '/home/ubuntu/foodclassification/online_food_images/Hot Dog.png'
online_img9_path = '/home/ubuntu/foodclassification/online_food_images/ice_cream.png'
online_img10_path = '/home/ubuntu/foodclassification/online_food_images/sushi.png'
online_img11_path = '/home/ubuntu/foodclassification/online_food_images/Taco.png'

def predict_image(filename, model_bit):
    img_ = image.load_img(filename, target_size=(256, 256))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.

    prediction = model_bit.predict(img_processed)
    index = np.argmax(prediction)

    plt.title("Prediction - {}".format(category[index][1]))
    plt.imshow(img_)
    # plt.title(f"Prediction - {index}")
    plt.show()


def predict_dir(filedir, model_bit):
    cols = 5
    pos = 0
    images = []
    total_images = len(os.listdir(filedir))
    rows = total_images // cols + 1

    true = filedir.split('/')[-1]

    fig = plt.figure(1, figsize=(25, 25))

    for i in sorted(os.listdir(filedir)):
        images.append(os.path.join(filedir, i))

    for subplot, imggg in enumerate(images):
        img_ = image.load_img(imggg, target_size=(299, 299))
        img_array = image.img_to_array(img_)

        img_processed = np.expand_dims(img_array, axis=0)

        img_processed /= 255.
        prediction = model_bit.predict(img_processed)
        index = np.argmax(prediction)

        pred = category.get(index)[0]
        if pred == true:
            pos += 1

        fig = plt.subplot(rows, cols, subplot + 1)
        fig.set_title(category.get(index)[1], pad=10, size=18)
        plt.imshow(img_array)
        # plt.show()

    acc = pos / total_images
    print("Accuracy of Test : {:.2f} ({pos}/{total})".format(acc, pos=pos, total=total_images))
    plt.tight_layout()

# Assuming you have defined 'online_img_path' and 'model'
predict_image(online_img1_path, model_bit)
predict_image(online_img2_path, model_bit)
predict_image(online_img3_path, model_bit)
predict_image(online_img4_path, model_bit)
predict_image(online_img5_path, model_bit)
predict_image(online_img6_path, model_bit)
predict_image(online_img7_path, model_bit)
predict_image(online_img8_path, model_bit)
predict_image(online_img9_path, model_bit)
predict_image(online_img10_path, model_bit)
predict_image(online_img11_path, model_bit)

predict_dir("/home/ubuntu/foodclassification/online_food_images",model_bit)
