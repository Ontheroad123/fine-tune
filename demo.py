#!/usr/bin/env python
# _*_ coding: utf-8 _*_

from keras.models import Model
from keras.layers import Dense
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras import optimizers
import math
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
# 训练的batch_size
batch_size = 16
# 训练的epoch
epochs = 10

# 图像Generator，用来构建输入数据
train_datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True)

# 从文件中读取数据，目录结构应为train下面是各个类别的子目录，每个子目录中为对应类别的图像
train_generator = train_datagen.flow_from_directory('./train1', target_size = (224, 224), batch_size = batch_size)

# 训练图像的数量
image_numbers = train_generator.samples

# 输出类别信息
print(train_generator.class_indices)

# 生成测试数据
test_datagen = ImageDataGenerator()
validation_generator = test_datagen.flow_from_directory('./validation', target_size = (224, 224), batch_size = batch_size)

# 使用ResNet的结构，不包括最后一层，且加载ImageNet的预训练参数
base_model = VGG16(weights = 'imagenet', include_top = False, pooling = 'avg')
#print(base_model.output.shape)
# 构建网络的最后一层
predictions = Dense(2, activation='sigmoid')(base_model.output)

# 定义整个模型
model = Model(inputs=base_model.input, outputs=predictions)
print(model.summary())
best_weights_filepath = 'best_weights.txt'
# 编译模型，loss为交叉熵损失
model.compile(optimizer=optimizers.SGD(lr=0.0001,momentum=0.9), loss='binary_crossentropy')
earlyStopping=EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
saveBestModel = ModelCheckpoint(best_weights_filepath,
            monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
#reduce_lr = ReduceLROnPlateau(monitor='val_loss',
#            factor=1/math.e,verbose=1, patience=10, min_lr=0.0001)
# 训练模型
model.fit_generator(train_generator,
                    steps_per_epoch = image_numbers ,
                    epochs = epochs,
                    validation_data = validation_generator,
                    validation_steps = batch_size,
                    callbacks = [earlyStopping,saveBestModel]
                    )

# 保存训练得到的模型
model.save('weights.h5')
