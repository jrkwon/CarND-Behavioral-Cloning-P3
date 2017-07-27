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

[image1]: ./writeup_images/final_model.png "Model Visualization"
[image2]: ./writeup_images/center_2017_07_24_13_04_59_177.jpg "Center Image"
[image3]: ./writeup_images/center_2017_07_21_09_46_07_301-recovery1.jpg "Recovery Image"
[image4]: ./writeup_images/center_2017_07_21_09_46_07_301-recovery2.jpg "Recovery Image"
[image5]: ./writeup_images/center_2017_07_21_09_46_07_301-recovery3.jpg "Recovery Image"
[image6]: ./writeup_images/left_2017_07_21_09_45_41_362.jpg "Normal Image"
[image7]: ./writeup_images/left_2017_07_21_09_45_41_362_flip.jpg "Flipped Image"
[image8]: ./writeup_images/mean_square_error_loss.png "Model Error Loss"
[image9]: ./writeup_images/mean_square_error_loss_with_earlystopping.png "Model Error Loss with Early Stopping"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py - containing the script to create and train the model with the generator
* clone.py - containing the script to create and train the model without using the generator
* drive.py - for driving the car in autonomous mode
* model.h5 - containing a trained convolution neural network using the generator 
* writeup_report.md - summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of three convolutional neural networks with 5x5 filter size and depths between 24, 36, and 48, and two more convolutional neural networks with 3x3 filter size and depths 64 are followed (model.py lines 96-100).

The model includes RELU layers to introduce nonlinearity (code line 96-100), and the data is normalized in the model using a Keras lambda layer (code line 94). 

The model also includes four fully connected layers after the convolutional networks. The numbers of the nodes in the layers are 100, 50, 10, and 1.

#### 2. Attempts to reduce overfitting in the model

I added a dropout layer with 20% rate.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 54-55). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 110).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and recovering from the left and right sides of the road. The left and right camera images were also used to help the car maintain the center lane.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a well known network that showed its successful performance in the autonomous vehicle control. 

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because that worked with my traffic sign classifier project. But the mean square error loss of the validation set did not decrease. So, I changed the network architecture to the one that NVIDIA used in [this](https://arxiv.org/pdf/1604.07316) paper. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added a dropout layer in the fully connected layers with the 20% dropout rate to the original NVIDIA model. Also, overfitting can be addressed by using more training data. Thus, I added more training data by driving the track more laps.

Then I have similar loss values from my training and validation sets. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I went to those particular positions and drove to car from the view where the car fell off the track to the center lane. I repeated this multiple times to collect enough data to make the neural network remember the cases. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture. This is identical to the NVIDIA's network except the input image size and the number of node in the first fully connected layer. They used 200x66x3, but I used 90x320x3. The first fully connected layer containing the 1,164 neurons in the original NVIDIA's network was removed because the layer made the model too big for this project submission. I was able to make a successful model without the layer by adding a dropout and adding more training data. Note that the input image size is 160x320x3, but the input image is cropped to remove some sections that are not part of the road. I took this image from [the NVIDIA's paper](https://arxiv.org/pdf/1604.07316).

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three and half laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to go back to the center lane. These images show what a recovery looks like starting from the right edge of the road :

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles thinking that this would help to avoid overfitting. For example, here are images that show the original and the flipped one:

![alt text][image6]
![alt text][image7]

After the collection process, I had 12,000 number of data points. I then preprocessed this data by adding the left and right camera input with  calculated steering angles from the angle value based on the center image. The calculation can be done by adding 0.2 to the angle for the left and by subtracting 0.2 from the anle for the right. This will help to maintain the center of the road.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The plot below shows the changes of mean square error loss values of the training and validation set when I didn't use the generator. ModelCheckPoint and EarlyStopping were not used as well. As the plot indicates, in the fifth epoch the mean square error loss of the training set is lower than the loss of the validation set. This indicates overfitting is happening. 

![alt text][image8]

After introducting the generator and dropout, the data set that I used to train the convolutional neural network without them was not able to train the network. So I had to collect more training data in the positions where the vehicle fell over. 
 
Finally, I used the ModelCheckpoint function to choose the correct number of epochs, and the EarlyStopping function was used to stop the training after the model stops improving. When it comes to the learning rate, I used an adam optimizer so that manually training the learning rate was not necessary.

![alt text][image9]

The training was stopped early in the case where I used ModelCheckpoint and EarlyStopping since the validation loss was not improved in the later epoch. 