from __future__ import print_function ## To use the print() function from python-3 to python 2.6+
import keras
from keras.preprocessing.image import ImageDataGenerator ## to manipulate the image
from keras.models import Sequential ## to build models as a simple stack of layers
from keras. layers import Dense, Dropout, Activation, Flatten, BatchNormalization ## automatically standardize the inputs to a layer
from keras.layers import Conv2D, MaxPooling2D
import os

num_classes = 5 ## no of emotions
img_rows, img_cols = 48, 48 ## pixel size
batch_size = 32

train_data_dir = r'/train'
validation_data_dir = r'/validation/'

train_datagen = ImageDataGenerator(
                                        rescale = 1./255,
                                        rotation_range = 30,
                                        shear_range = 0.3,
                                        zoom_range = 0.3,
                                        width_shift_range = 0.4,
                                        height_shift_range = 0.4,
                                        horizontal_flip = True,
                                        fill_mode = 'nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                                        train_data_dir,
                                        color_mode = 'grayscale',
                                        target_size = (img_rows, img_cols),
                                        batch_size = batch_size,
                                        class_mode = 'categorical',
                                        shuffle = True)

validation_generator = validation_datagen.flow_from_directory(
                                        validation_data_dir,
                                        color_mode = 'grayscale',
                                        target_size = (img_rows, img_cols),
                                        batch_size = batch_size,
                                        class_mode = 'categorical',
                                        shuffle = True)

model = Sequential()

## Block - 1
## Only first layer needs input_shape, rest of the layers will be maintained in shape automatically

model.add(Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal', input_shape=(img_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal', input_shape=(img_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


## Block - 2

model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


## Block - 3

model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


## Block - 4

model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3) , padding ='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


## Block - 5

model.add(Flatten())
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))


## Block - 6

model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


## Block - 7

model.add(Dense(num_classes, kernel_initializer='he_normal'))
model.add(Activation('elu'))

print(model.summary())


## Training of model begins
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint('facial_expression_model.h5',
                             monitor = 'val_loss',
                             mode = 'min',
                             save_best_only = True,
                             verbose = 1)
## It is an approach where a snapshot of the state of the system is taken in case of system failure.
## If there is a problem, not all is lost. The checkpoint may be used directly, or used as the starting point for
## a new run, picking up where it left off.


earlystop = EarlyStopping(monitor = 'val_loss',
                      min_delta = 0,
                      patience = 3, 
                      verbose = 1,
                      restore_best_weights = True)
## A compromise is to train on the training dataset but to stop training at the point when performance on a
## validation dataset starts to degrade. This simple, effective, and widely used approach to
## training neural networks is called early stopping

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                              factor = 0.2,
                              patience = 3,
                              verbose = 1,
                              min_delta = 0.0001)
## Reducing Learning rate will help model to get better results.

callbacks = [checkpoint, earlystop, reduce_lr]

model.compile(loss = 'categorical_crossentropy', 
              optimizer = Adam(lr=0.001),
              metrics = ['accuracy'])


nb_train_samples = 24282
nb_validation_samples = 5937
epochs = 2

history = model.fit_generator(train_generator,
                              steps_per_epoch = nb_train_samples // batch_size,
                              epochs = epochs,
                              callbacks = callbacks,
                              validation_data = validation_generator,
                              validation_steps = nb_validation_samples // batch_size)
