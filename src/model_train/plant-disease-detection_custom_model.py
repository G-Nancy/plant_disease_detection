# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel) (Local)
#     language: python
#     name: conda-base-py
# ---

# %%
# # !pip install -U tensorflow
# # !pip install -U 'protobuf>=3.4.0'

# %%
# # !unzip -q archive.zip

# %% _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" papermill={"duration": 24.752299, "end_time": "2025-08-07T04:36:39.958092", "exception": false, "start_time": "2025-08-07T04:36:15.205793", "status": "completed"}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.optimizers import Adam


from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.resnet import ResNet101
from tensorflow.keras.applications import ResNet50
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import EfficientNetV2L

import random
import os
import warnings
warnings.filterwarnings('ignore')
print('compelet')

# %% papermill={"duration": 29.449478, "end_time": "2025-08-07T04:37:09.412144", "exception": false, "start_time": "2025-08-07T04:36:39.962666", "status": "completed"}
image_shape = (224,224)
batch_size = 64

#input/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train
train_dir="input/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train"
valid_dir="input/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/valid"

# apply scaling only becouse data already augmented
train_datagen = ImageDataGenerator(rescale=1/255., validation_split=0.2)
test_datagen = ImageDataGenerator(rescale = 1/255.)

# load training data
print("Training Images:")
train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=image_shape,
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               shuffle=True,
                                               subset='training')

# load validation data (20% of training data)
print("Validating Images:")
valid_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=image_shape,
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               shuffle=False,
                                               subset='validation')

# load test data (consider validation data as test data)
print('Test Images:')
test_data = test_datagen.flow_from_directory(valid_dir,
                                               target_size=image_shape,
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               shuffle=False)

# %% papermill={"duration": 0.524235, "end_time": "2025-08-07T04:37:09.940692", "exception": false, "start_time": "2025-08-07T04:37:09.416457", "status": "completed"}
# show how data store 
images, labels = next(iter(train_data))
print(f'shape of image is : {images[0].shape}')
print(f'label  \n{labels[0]}')

# %% papermill={"duration": 0.010434, "end_time": "2025-08-07T04:37:09.955373", "exception": false, "start_time": "2025-08-07T04:37:09.944939", "status": "completed"}
# show all diseases in dataset
diseases = os.listdir(train_dir)
print(diseases)

# %% papermill={"duration": 0.010293, "end_time": "2025-08-07T04:37:09.969645", "exception": false, "start_time": "2025-08-07T04:37:09.959352", "status": "completed"}
# identify uniqe plant in dataset
plants = []
NumberOfDiseases = 0
for plant in diseases:
    if plant.split('___')[0] not in plants:
        plants.append(plant.split('___')[0])
print(f'number of different plants is :{len(plants)}')
print(plants)

# %% papermill={"duration": 0.602425, "end_time": "2025-08-07T04:37:10.576158", "exception": false, "start_time": "2025-08-07T04:37:09.973733", "status": "completed"}
# show number of each class
dic = {}
for Class in diseases:
    dic[Class] = len(os.listdir(train_dir + '/' + Class))

df = pd.DataFrame(list(dic.items()), columns=["Disease Class", "Number of Images"])

# df = df.sort_values(by="Number of Images", ascending=False)

plt.figure(figsize=(15,5))
sns.barplot(data=df ,x='Disease Class' ,y= 'Number of Images' )
plt.xticks(rotation=90)
plt.show()

# %% papermill={"duration": 2.155648, "end_time": "2025-08-07T04:37:12.737979", "exception": false, "start_time": "2025-08-07T04:37:10.582331", "status": "completed"}
import random
# select a specific batch
images, labels = next(iter(train_data))

# select 16 image by random
indices = random.sample(range(len(images)), 16)
selected_images = images[indices]
selected_labels = labels[indices]

class_names = list(train_data.class_indices.keys())

# plotting
plt.figure(figsize=(12, 12))
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(selected_images[i])
    plt.title(class_names[np.argmax(selected_labels[i])])  
    plt.axis("off")  

plt.tight_layout()
plt.show()


# %% papermill={"duration": 0.048032, "end_time": "2025-08-07T04:37:12.827652", "exception": false, "start_time": "2025-08-07T04:37:12.779620", "status": "completed"}
def plot_learning_curves(history):
    plt.figure(figsize=(12, 4))

    # accuracy curve
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    # loss curve
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


# %% papermill={"duration": 0.05555, "end_time": "2025-08-07T04:37:12.924891", "exception": false, "start_time": "2025-08-07T04:37:12.869341", "status": "completed"}
def predict_labels_and_display(model_path, test_dir='test/test', image_size=(224, 224)):
    # load the best model
    best_model = load_model(model_path)

    true_labels = []
    predicted_labels = []
    images = []

    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')): 
            # load test images
            img_path = os.path.join(test_dir, filename)
            img = load_img(img_path, target_size=image_size)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # predict
            prediction = best_model.predict(img_array, verbose=0)
            predicted_class = np.argmax(prediction, axis=1)[0]

            # extract the label (name of image)
            true_label = filename.split('.')[0]

            # get the prediction class
            class_labels = list(train_data.class_indices.keys())
            predicted_label = class_labels[predicted_class]

            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
            images.append(img)

    # randomly select three images
    selected_indices = random.sample(range(len(images)), 3)

    # show selected images
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(selected_indices):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[idx])
        plt.title(f'True: {true_labels[idx]}\nPredicted: {predicted_labels[idx]}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# %% papermill={"duration": 4.321852, "end_time": "2025-08-07T04:37:17.292012", "exception": false, "start_time": "2025-08-07T04:37:12.970160", "status": "completed"}
# Model Architecture
model = Sequential()

model.add(Conv2D(32,(3,3),activation = 'relu',input_shape=(224,224,3), kernel_initializer=GlorotNormal()))
model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer=GlorotNormal()))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer=GlorotNormal()))
model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer=GlorotNormal()))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer=GlorotNormal()))
model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer=GlorotNormal()))
model.add(MaxPooling2D(2,2))

# model.add(Flatten())
model.add(GlobalAveragePooling2D())

model.add(Dense(256, activation='relu', kernel_initializer=GlorotNormal()))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(128, activation='relu', kernel_initializer=GlorotNormal()))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu', kernel_initializer=GlorotNormal()))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(38, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.summary()

# %% papermill={"duration": 1838.643814, "end_time": "2025-08-07T05:07:55.983789", "exception": false, "start_time": "2025-08-07T04:37:17.339975", "status": "completed"}
# train the model
model_checkpoint = ModelCheckpoint('model/working/cnn_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, mode='max', restore_best_weights=True)

# history = model.fit(train_data,
#                     validation_data=valid_data,
#                     epochs=10,
#                     batch_size=64, 
#                     callbacks=[model_checkpoint, early_stopping])

# %% papermill={"duration": 0.353584, "end_time": "2025-08-07T05:07:56.596862", "exception": false, "start_time": "2025-08-07T05:07:56.243278", "status": "completed"}
# model.save('model/working/cnn_model.keras')

# %% papermill={"duration": 0.640715, "end_time": "2025-08-07T05:07:57.494152", "exception": false, "start_time": "2025-08-07T05:07:56.853437", "status": "completed"}
# show learning curves
# plot_learning_curves(history)

# %% papermill={"duration": 166.400488, "end_time": "2025-08-07T05:10:44.160646", "exception": false, "start_time": "2025-08-07T05:07:57.760158", "status": "completed"}
# showe accuracy on test data (model evaluation)
best_model = load_model('model/working/cnn_model.keras')

test_loss, test_accuracy = best_model.evaluate(test_data)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# %% papermill={"duration": 4.815566, "end_time": "2025-08-07T05:10:49.247721", "exception": false, "start_time": "2025-08-07T05:10:44.432155", "status": "completed"}
# show random sample of prediction of model on test data
predict_labels_and_display('model/working/cnn_model.keras')


# %% papermill={"duration": 2.474493, "end_time": "2025-08-07T05:10:52.017579", "exception": false, "start_time": "2025-08-07T05:10:49.543086", "status": "completed"}
from tensorflow.keras.applications import InceptionV3

inception_v3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in inception_v3.layers:
    layer.trainable = False


# %% papermill={"duration": 0.351083, "end_time": "2025-08-07T05:10:52.656756", "exception": false, "start_time": "2025-08-07T05:10:52.305673", "status": "completed"}
last_output = inception_v3.output

x = GlobalAveragePooling2D()(last_output)

x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Dense(38, activation='softmax')(x)  # 38 classes

# Build the final model
inception_v3_model = Model(inputs=inception_v3.input, outputs=x)

# Compile
inception_v3_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %% papermill={"duration": 0.597557, "end_time": "2025-08-07T05:10:53.537689", "exception": false, "start_time": "2025-08-07T05:10:52.940132", "status": "completed"}
inception_v3_model.summary()

# %% papermill={"duration": 4543.211149, "end_time": "2025-08-07T06:26:37.060134", "exception": false, "start_time": "2025-08-07T05:10:53.848985", "status": "completed"}
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Callbacks
model_checkpoint = ModelCheckpoint('model/working/inception_v3_model.keras', 
                                   monitor='val_accuracy', 
                                   save_best_only=True, 
                                   mode='max', 
                                   verbose=1)

early_stopping = EarlyStopping(monitor='val_accuracy', 
                               patience=5, 
                               verbose=1, 
                               mode='max', 
                               restore_best_weights=True)

model_ReduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', 
                                            factor=0.1, 
                                            patience=10, 
                                            min_lr=1e-6)

# Fit modelhistory = inception_v3_model.fit(train_data,
#                                  validation_data=valid_data,
#                                  epochs=20,
#                                  batch_size=64,
#                                  callbacks=[model_checkpoint, early_stopping, model_ReduceLROnPlateau])
# 

# %% papermill={"duration": 1.345356, "end_time": "2025-08-07T06:26:39.454852", "exception": false, "start_time": "2025-08-07T06:26:38.109496", "status": "completed"}
# show learning curves
# plot_learning_curves(history)

# %% papermill={"duration": 100.733119, "end_time": "2025-08-07T06:28:21.241102", "exception": false, "start_time": "2025-08-07T06:26:40.507983", "status": "completed"}
from tensorflow.keras.models import load_model

# Load the best InceptionV3 model
# best_model = load_model('model/working/inception_v3_model.keras')

# # Evaluate the model on test data
# test_loss, test_accuracy = best_model.evaluate(test_data)
# print(f"Test Loss: {test_loss:.4f}")
# print(f"Test Accuracy: {test_accuracy:.4f}")


# %% papermill={"duration": 139.471262, "end_time": "2025-08-07T06:30:41.854320", "exception": false, "start_time": "2025-08-07T06:28:22.383058", "status": "completed"}
from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(model_path, test_data, model_name):
    # load model
    model = tf.keras.models.load_model(model_path)

    # predict
    y_pred = model.predict(test_data)
    y_pred_classes = y_pred.argmax(axis=1)

    # get true classes
    y_true = test_data.classes

    class_labels = list(test_data.class_indices.keys())

    accuracy = accuracy_score(y_true, y_pred_classes)

    report = classification_report(y_true, y_pred_classes, target_names=class_labels, output_dict=True)

    df_report = pd.DataFrame(report).transpose()

    df_report.loc['accuracy'] = [accuracy, None, None, None]

    df_report['model'] = model_name

    return df_report

cnn_model_path = 'model/working/cnn_model.keras'
inception_v3_model_path = 'model/working/inception_v3_model.keras'


cnn_report = evaluate_model(cnn_model_path, test_data, 'CNN')
inception_report = evaluate_model(inception_v3_model_path, test_data, 'InceptionV3')



all_reports = pd.concat([cnn_report ,inception_report])

all_reports = all_reports.reset_index().rename(columns={'index': 'metric'})

all_reports

# %% papermill={"duration": 1.122733, "end_time": "2025-08-07T06:30:44.002276", "exception": false, "start_time": "2025-08-07T06:30:42.879543", "status": "completed"}
pd.set_option('display.max_rows', 82)

all_reports

# %% papermill={"duration": 1.099445, "end_time": "2025-08-07T06:30:46.178881", "exception": false, "start_time": "2025-08-07T06:30:45.079436", "status": "completed"}
x =all_reports[all_reports['metric'] == 'macro avg']
x


#Try mobilenetv2 model transfer learning
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
def build_mobilenet():
    # Load the pre-trained MobileNetV2 model
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create the new model
    
    # Build the model using Sequential
    model = Sequential()

    # Add the base model (Xception)
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(38, activation="softmax"))

    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

# Create the model
mobilenet_model = build_mobilenet()

# Display model summary
mobilenet_model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Callbacks
mobilenet_model_checkpoint = ModelCheckpoint('model/working/mobilenetv2_model.keras', 
                                   monitor='val_accuracy', 
                                   save_best_only=True, 
                                   mode='max', 
                                   verbose=1)

mobilenet_early_stopping = EarlyStopping(monitor='val_accuracy', 
                               patience=5, 
                               verbose=1, 
                               mode='max', 
                               restore_best_weights=True)

mobilenet_model_ReduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', 
                                            factor=0.1, 
                                            patience=10, 
                                            min_lr=1e-6)

# Fit model
history = mobilenet_model.fit(train_data,
                                 validation_data=valid_data,
                                 epochs=20,
                                 batch_size=64,
                                 callbacks=[mobilenet_model_checkpoint, mobilenet_early_stopping, mobilenet_model_ReduceLROnPlateau])