# Introduction

This project is to train a model with .pb format which can be loaded with python or C++ code to counting people from images or videos.

# Requirement

- conda create -n keras python=3.6
- source activate keras
- conda install tensorflow-gpu==1.4.1
- pip install keras==2.1
- pip install opencv-contrib-python

# Usage

we assume that you want to put the project at some place called ***ROOT_PATH***

- cd ROOT_PATH
- git clone https://git.light-field.tech/CVIntern/image_clip.git

under ***ROOT_PATH***, you should also create a folder named ***dataset*** to put dataset 

- mkdir dataset 

So, you now have a folder named ***crowd_counting*** and a folder named ***dataset*** under ***ROOT_PATH***

## 1.Making dataset

### (1)Generate images from video

- python ./generate_dataset/generate_image_from_video.py video_path save_path \

  num_frame frame_interval init_frame

### (2)Annotate images

We recommand you to annotate images with the tool ***LabelImg*** 

you can download ***LabelImg*** from [here](https://github.com/tzutalin/labelImg), then follow the instructions in github.

As a result, you should have **a image folder** which contains from 00000.jpg to xxxxx.jpg

and **a labels folder** which contains from 00000.xml to xxxxx.xml

if you have 2000 images to annotate , you have let 4 people to do that ,then you might have 

4 image folders and 4 labels folders like bellow:

![1533295350519](/tmp/1533295350519.png)

![1533295379783](/tmp/1533295379783.png)

we recommand you to name folders like above, and 

- put those image folders in ***ROOT_PATH/dataset/dataset_name/images/***
- put those gt folders in ***ROOT_PATH/dataset/dataset_name/gt/***

### (3)Merge folders into one folder

you may have several image folders and same number gt folders ,then you need to put them together:

- cd ROOT_PATH/crowd_counting
- source activate keras
- python ./generate_dataset/hongqiao/1-merge_groups_to_one.py  data_path  folder_numbers

then you will get a folder named ***images*** under ***dataset/dataset_name/images***

and get a folder named ***gt*** under ***dataset/dataset_name/gt***

### (4)generate ground truth density map from .xml

Now , you need to generate ground truth density map from those labels with .xml format

- python ./generate_dataset/hongqiao/2-generate_density_for_hongqiao.py

You will get 2 folders under ***dataset/dataset_name/gt*** like bellow:

![1533296806463](/tmp/1533296806463.png)

density_map contains xxxxx.npy which is the ground truth you will use to train,

density_image contains xxxxx.jpg which just shows how the density map looks like.

### (5)Test or augment you dataset

you may want to test whether your generated density map is correct or you want to augment your data to get more data. To do that you can run the scripts bellow,but I'm sorry that I can't put too much time to explain how those scripts work.

- to test your density map:  python ./generate_dataset/hongqiao/3-test_generated_density.py

- to augment data: python ./generate_dataset/hongqiao/4-data_augmentation

## 2.Training

### （1）Save the whole dataset into one nparray

For fast training, I choose to load the whole dataset and saved all the image data and gt data 

into one nparray ,like bellow:

![1533297428414](/tmp/1533297428414.png)

- xxxxx_gt4.npy: a nparray contains all the gt , with the shape: (num_samples, height ,width, channels)

- xxxxx_X4.npy: a nparray contains all the image data, shape: (num_samples, height ,width, channels)

to do this you should change the ***data_load_plan*** to ***1*** in train_on_hongqiao.py and run:

- python train_on_hongqiao.py

### （2）Training the model

you will have two choice to train the model:

#### ① Train a new model from the dataset:

change ***data_load_plan*** to ***2*** and run: 

- python train_on_hongqiao.py

#### ② Fine-tune a trained model with new dataset

change ***data_load_plan*** to ***3*** and run: 

- python train_on_hongqiao.py

### （3）Test trained model

#### ① Test trained model with images:

- python image_test.py

#### ② Test trained model with videos:

- python video_test.py video_path if_add_brightness  #if_add_brightness=0:not add or =1 :add

#### （4）Convert model from .h5 to .pb

- python h5_to_pb.py

you can also test if the .pb model can works like the .h5 model:

- python load_pb_test



#### Now congratulations , you have got a useful .pb model and you can load it with my C++ API to test images or videos with C++!