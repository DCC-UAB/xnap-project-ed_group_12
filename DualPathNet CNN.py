import wandb
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
np.random.seed(42)

from matplotlib import style
style.use('fivethirtyeight')

wandb.init(project='traffic-sign-detection')

# Assign dataset paths
data_dir = r'C:\Users\pc\Desktop\Deep Learning\proyect\Dataset - German Traffic Signs'
train_path = r'C:\Users\pc\Desktop\Deep Learning\proyect\Dataset - German Traffic Signs\Train'
test_path = r'C:\Users\pc\Desktop\Deep Learning\proyect\Dataset - German Traffic Signs\Test'

# Resizing the images to 30x30x3
IMG_HEIGHT = 30
IMG_WIDTH = 30
channels = 3

# Finding total classes
NUM_CATEGORIES = len(os.listdir(train_path))
NUM_CATEGORIES

# Label Overview
classes = {0: 'Speed limit (20km/h)',
           1: 'Speed limit (30km/h)',
           2: 'Speed limit (50km/h)',
           3: 'Speed limit (60km/h)',
           4: 'Speed limit (70km/h)',
           5: 'Speed limit (80km/h)',
           6: 'End of speed limit (80km/h)',
           7: 'Speed limit (100km/h)',
           8: 'Speed limit (120km/h)',
           9: 'No passing',
           10: 'No passing veh over 3.5 tons',
           11: 'Right-of-way at intersection',
           12: 'Priority road',
           13: 'Yield',
           14: 'Stop',
           15: 'No vehicles',
           16: 'Veh > 3.5 tons prohibited',
           17: 'No entry',
           18: 'General caution',
           19: 'Dangerous curve left',
           20: 'Dangerous curve right',
           21: 'Double curve',
           22: 'Bumpy road',
           23: 'Slippery road',
           24: 'Road narrows on the right',
           25: 'Road work',
           26: 'Traffic signals',
           27: 'Pedestrians',
           28: 'Children crossing',
           29: 'Bicycles crossing',
           30: 'Beware of ice/snow',
           31: 'Wild animals crossing',
           32: 'End speed + passing limits',
           33: 'Turn right ahead',
           34: 'Turn left ahead',
           35: 'Ahead only',
           36: 'Go straight or right',
           37: 'Go straight or left',
           38: 'Keep right',
           39: 'Keep left',
           40: 'Roundabout mandatory',
           41: 'End of no passing',
           42: 'End no passing veh > 3.5 tons'}

# Collecting the training data
image_data = []
image_labels = []

for i in range(NUM_CATEGORIES):
    path = data_dir + '/Train/' + str(i)
    images = os.listdir(path)

    for img in images:
        try:
            image = cv2.imread(path + '/' + img)
            image_fromarray = Image.fromarray(image, 'RGB')
            resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
            image_data.append(np.array(resize_image))
            image_labels.append(i)
        except:
            print("Error in " + img)

# Changing the list to numpy array
image_data = np.array(image_data)
image_labels = np.array(image_labels)

print(image_data.shape, image_labels.shape)

# Shuffling training data
shuffle_indexes = np.arange(image_data.shape[0])
np.random.shuffle(shuffle_indexes)
image_data = image_data[shuffle_indexes]
image_labels = image_labels[shuffle_indexes]

# Splitting the data into train and validation set
X_train, X_val, y_train, y_val = train_test_split(image_data, image_labels, test_size=0.3, random_state=42, shuffle=True)

X_train = X_train / 255
X_val = X_val / 255

print("X_train.shape", X_train.shape)
print("X_valid.shape", X_val.shape)
print("y_train.shape", y_train.shape)
print("y_valid.shape", y_val.shape)

# One hot encoding the labels
y_train = keras.utils.to_categorical(y_train, NUM_CATEGORIES)
y_val = keras.utils.to_categorical(y_val, NUM_CATEGORIES)

print(y_train.shape)
print(y_val.shape)

# Making the DualPath98 model
def dualpath_block(x, filters, strides=(1, 1)):
    shortcut = x

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='same')(x)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    if strides != (1, 1):
        shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(shortcut)

    output = keras.layers.Concatenate(axis=-1)([shortcut, x])
    return output



def build_dualpath98_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = dualpath_block(x, filters=64, strides=(1, 1))
    x = dualpath_block(x, filters=128, strides=(2, 2))
    x = dualpath_block(x, filters=256, strides=(2, 2))

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(rate=0.5)(x)
    x = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.models.Model(inputs=inputs, outputs=x)
    return model


model = build_dualpath98_model((IMG_HEIGHT, IMG_WIDTH, channels), NUM_CATEGORIES)
model.summary()

lr = 0.001
epochs = 30

# Initialize WandB's Keras Callback
wandb_callback = wandb.keras.WandbCallback()
wandb.config.lr = lr
wandb.config.epochs = epochs

opt = Adam(lr=lr, decay=lr / (epochs * 0.5))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Augmenting the data and training the model
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

history = model.fit(aug.flow(X_train, y_train, batch_size=32), epochs=epochs, validation_data=(X_val, y_val), callbacks = [wandb_callback])

# Log training history
wandb.log({"Training Loss": history.history['loss'],
           "Validation Loss": history.history['val_loss'],
           "Training Accuracy": history.history['accuracy'],
           "Validation Accuracy": history.history['val_accuracy']})

# Evaluating the model
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# Loading the test data and running the predictions
test = pd.read_csv(data_dir + '/Test.csv')

labels = test["ClassId"].values
imgs = test["Path"].values

data = []

for img in imgs:
    try:
        image = cv2.imread(data_dir + '/' + img)
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
        data.append(np.array(resize_image))
    except:
        print("Error in " + img)
X_test = np.array(data)
X_test = X_test / 255

pred = model.predict_classes(X_test)

# Accuracy with the test data
test_accuracy = accuracy_score(labels, pred) * 100
print('Test Data accuracy: ', test_accuracy)
wandb.log({"Test Accuracy": test_accuracy})

# Visualizing the confusion matrix
from sklearn.metrics import confusion_matrix
wandb.log({"Confusion Matrix": wandb.Image(plt)})
cf = confusion_matrix(labels, pred)

import seaborn as sns
df_cm = pd.DataFrame(cf, index=classes, columns=classes)
plt.figure(figsize=(20, 20))
sns.heatmap(df_cm, annot=True)

# Classification report
from sklearn.metrics import classification_report
wandb.log({"Classification Report": classification_report(labels, pred, target_names=list(classes.values()), output_dict=True)})
print(classification_report(labels, pred))

# Predictions on Test Data
plt.figure(figsize=(25, 25))

start_index = 0
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    prediction = pred[start_index + i]
    actual = labels[start_index + i]
    col = 'g'
    if prediction != actual:
        col = 'r'
    plt.xlabel('Actual={} || Pred={}'.format(actual, prediction), color=col)
    plt.imshow(X_test[start_index + i])
plt.show()
wandb.finish()
