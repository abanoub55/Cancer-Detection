from django.contrib.auth import authenticate
from django.shortcuts import render
import os
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import cv2
import numpy as np
from django.http import HttpResponse
from django.urls import reverse_lazy
from django.views import generic
from .forms import CustomUserCreationForm

# Create your views here.
LR = 1e-3
IMG_SIZE = 50
MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')


def define_model():
    tf.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')
    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')
    return model


def index(request):
    return render(request, 'index.html')


def about(request):
    return render(request, 'about.html')


def visualize(request):
    return render(request, 'visualization.html')


def stats(request):
    return render(request, 'stats.html')


def login(request):
    return render(request, 'registration/login.html')

def base(request):
    return render(request, 'base.html')


def contact(request):
    return render(request, 'contact.html')


def diagnosis(request):
    return render(request, 'diagnosis.html')


def upload(request):
    if request.method == 'POST':
        # Get the file from post request
        f = request.FILES['file']
        # print('image name', f, type(f))
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            "/home/abanoub/data/cats_dogs/test", str(f))
        # f.save(file_path)
        print(file_path)
        model = define_model()
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data = img.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]
        if np.argmax(model_out) == 1:
            str_label = 'Dog'
        else:
            str_label = 'Cat'

        return HttpResponse(str_label)
    return HttpResponse('error has occurred!')


class SignUp(generic.CreateView):
    form_class = CustomUserCreationForm
    success_url = reverse_lazy('login')
    template_name = 'signup.html'
