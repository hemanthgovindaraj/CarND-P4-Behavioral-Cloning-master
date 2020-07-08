# Read data
import csv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Input,Flatten,Dense,Lambda,Cropping2D,Convolution2D,Dropout,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
import sklearn



model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(24,(5,5),subsample=(2,2),activation='elu'))
model.add(Dropout(.5))
model.add(Convolution2D(36,(5,5),subsample=(2,2),activation='elu'))
model.add(Dropout(.5))
model.add(Convolution2D(48,(5,5),subsample=(2,2),activation='elu'))
model.add(Dropout(.5))
model.add(Convolution2D(64,(3,3),activation='elu'))
model.add(Dropout(.5))
model.add(Convolution2D(64,(3,3),activation='elu'))
model.add(Dropout(.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

# Train the model
batch_size = 18
epochs = 3

samples =[]
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        samples.append(row)
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples):
    images = []
    measurements = []
    
    for sample in samples:
        steering_center = float(sample[3])
        correction = 0.10
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        for i in range(3):
            source_path = sample[0]
            filename = source_path.split('/')[-1]
            current_path = 'data/IMG/'
            if i==0:
                img = np.asarray(mpimg.imread(current_path + filename))
                images.append(img)
                measurements.append(steering_center)
                images.append(np.fliplr(img))
                measurements.append(steering_center*(-1))
            elif i==1:
                img = np.asarray(mpimg.imread(current_path + filename))
                images.append(img)
                measurements.append(steering_left)
                images.append(np.fliplr(img))
                measurements.append(steering_left*(-1))
            elif i==2:
                img = np.asarray(mpimg.imread(current_path + filename))
                images.append(img)
                measurements.append(steering_right)
                images.append(np.fliplr(img))
                measurements.append(steering_right*(-1))
                
    X_data = np.array(images)
    y_data = np.array(measurements)
                
    return X_data, y_data

def sample_generator(samples,batch_size=18):
    while 1:
        num_samples = len(samples)
        shuffle(samples)
        X_sample = []
        y_sample = []
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            X_data, y_data = generator(batch_samples)
        yield X_data, y_data

train_generator = sample_generator(train_samples,batch_size)
validation_generator = sample_generator(train_samples,batch_size)             

object_X1 = model.fit_generator(train_generator,
                                steps_per_epoch=np.ceil((len(train_samples))/batch_size),epochs=epochs,verbose=1,
                                validation_data=validation_generator,
                                validation_steps=np.ceil((len(validation_samples))/batch_size))

print(object_X1.history.keys())
### plot the training and validation loss for each epoch
plt.plot(object_X1.history['loss'])
plt.plot(object_X1.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')