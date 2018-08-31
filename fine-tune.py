from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense,GlobalAveragePooling2D,Convolution2D,MaxPooling2D,ZeroPadding2D
import os
import h5py
from keras.applications.vgg16 import VGG16
from keras.models import load_model
from keras import backend as K
from keras.utils import to_categorical
K.set_image_dim_ordering('th')

# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'train1'
validation_data_dir = 'validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 10
batch_size = 16
nb_classes = 2

#model = VGG16(weights='imagenet', include_top=False)
base_model = VGG16(weights='imagenet',  include_top=False,input_shape=(3,224, 224))
x = base_model.output
x= Flatten(name='flatten')(x)#使用flatten层的时候，需要设定input_shape
#x = GlobalAveragePooling2D()(x)#可以使用GlobalAveragePooling2D代替flatten层，不需要设定input_shape
# let's add a fully-connected layer
x = Dense(4096, activation='relu',name='fc1')(x)
x = Dense(4096, activation='relu',name='fc2')(x)
x = Dense(nb_classes, activation='sigmoid')(x)
model = Model(inputs=base_model.inputs, outputs=x)
print(model.summary())

# set the first 13 layers (up to the last conv block)把随后一个卷积快之前的权重设置为不训练
# to non-trainable (weights will not be updated)
for layer in base_model.layers:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')
image_numbers = train_generator.samples
# fine-tune the model
model.fit_generator(
    train_generator,
    steps_per_epoch = image_numbers,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps = batch_size
    )
model.save('weights1.h5')
