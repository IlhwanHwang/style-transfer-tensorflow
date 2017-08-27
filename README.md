style-transfer-tensorflow
=============
# Introduction

Neural artistic style transfer implemented with tensorflow.

# Dependencies

	Python 3.5.3
	Tensorflow-GPU 1.2.1

# How to use

First of all, you need to download pre-trained ImageNet VGG-19 network. The file can be found [here][file-link]. If the file is ready, place it with main.py and simply

	python main.py

It will transfer style from Starry Night to Rose. You can try another style and content modifying 'meta' variable in main.py.

![content](resource/rose.jpg =x200)
![style](resource/starry_night.jpg =x200)
![result](result/result_step_final.png =x200)

# Reference

The original paper is <[Image Style Transfer Using Convolutional Neural Networks][paper]>. Check out the paper for the theoretical details.

[file-link]: http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
[paper]: http://ieeexplore.ieee.org/document/7780634/