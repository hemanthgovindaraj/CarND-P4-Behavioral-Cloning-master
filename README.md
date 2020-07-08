# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)


[image1]: ./examples/left_camera.jpg "Left camera Image"
[image2]: ./examples/center_camera.jpg "Center camera Image"
[image3]: ./examples/right_camera.jpg "Right camera Image"
[image4]: ./examples/left_camera_flipped.jpg "Left camera Flipped Image"
[image5]: ./examples/center_camera_flipped.jpg "Center camera Flipped Image"
[image5]: ./examples/right_camera_flipped.jpg "Right camera Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script for reading and processing the dataset and to create and train the model that learns how to drive
* drive.py for driving the car in autonomous mode
* model.h5 containing the trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Code  for autonomous driving
Based on the Udacity driven camera images, i have generated a neural network model that can be used to drive the simulator automatically. 
```sh
python drive.py model.h5
```

#### 3. Model.py - code for generating and training the model for autonomous driving of simulator.

The model.py file contains the code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the architecture based on the NVIDIA paper. 
The separate architecture section below details the architecture used

#### 2. Attempts to reduce bias in the model

I started with a model having more dropout layers. This made the loss to be always in the higher side. There was not much improvement in the loss even after training for 10 epochs. Reducing the dropout layers one by one, i ended up with a model that had only 1 dropout layer  and it gave me a much lower value of loss. The validation loss was also decreasing continuously , hence there wasnt any overfitting.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data use was the one provided by the udacity team. I also tried generating my own data by a combination of center lane driving for 3 laps, a lap of going in the opposide track directiond and a lap where i swerve left and right within the track generating around 24000 * 3 images. But i could see better results and less computation time with just the 8000 x 3 images from the udacity team.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started with using the Inception V3 or GoogLeNet for the training. But i had an underfitting/bias issue and misjudged it to be the complex Inception that was causing the underfitting. I then modified my model to the one from NVIDIA.

Having been suggested in one of the videos and simple to debug the underfitting issue that i was facing, it ended up being a good architecture for the problem in hand.

I used the 80-20 distribution of the data samples for training and validation. 

Then I trained the model using the camera images as the input X, and steering angle as the y output.

The final step was to run the simulator to see how well the car was driving around track one. The vehicle fell off the road in the first big curve itself when i started , then i trained the model for lesser value of validation loss.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes.

Here is a visualization of the architecture

Layer (type)                 Output Shape              Param 
_________________________________________________________________
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 70, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 33, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 15, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 6, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 35, 64)         27712     
_________________________________________________________________
dropout_1 (Dropout)          (None, 4, 35, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 2, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 4224)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               422500    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        

Total params: 559,419
Trainable params: 559,419
Non-trainable params: 0
_________________________________________________________________

#### 3. Creation of the Training Set & Training Process

I finally randomly shuffled the data set and put 20% of the data into a validation set. 
I used all the 3 camera images to train the model and with an offset of 0.2 given to the left and right camera images, i was able to train the model well.
Here are 3 images from the left , center and right cameras.

![cameraLeft][image1]
![cameraCenter][image2]
![cameraRight][image3]

I also performed the augmentation of images using the fliplr function of numpy.
Here are the same 3 images after augmentation.

![cameraLeftFlipped][image1]
![cameraCenterFlipped][image2]
![cameraRightFlipped][image3]

Cropping and normalizing the data was performed using the Lambda layer and a Cropping layer at the top of the model.
The generator appraoch also reduced a lot of time in training the model. I was able to see good results after training the model for 7 epoch though i trained the model for 10 epochs.I also used the adam optimizer , hence tuning the learning rate wasnt really necessary.

The tuned hyperparameters are below.
Batch size for training - 16
Number of epochs - 10
Steeering correction - 0.2

The final video can be seen in the submission file run1.mp4
