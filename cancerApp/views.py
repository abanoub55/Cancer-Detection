import math
import os
import shutil
import zipfile
import cv2
import matplotlib.pyplot as plt
import numpy as np
import \
    pandas as pd  # for some simple data analysis (right now, just to load in the labels data and quickly reference it)
import pydicom  # for reading dicom files
import tensorflow as tf
from django.http import HttpResponse
from django.shortcuts import render
from django.urls import reverse_lazy
from django.views import generic
from tqdm import tqdm
from .forms import CustomUserCreationForm

# Create your views here.
home_dir = 'project_data/'
data_dir = home_dir + 'patients/'
labels = pd.read_csv('Labels.csv')

IMG_PX_SIZE = 80

HM_SLICES = 40

n_classes = 2

batch_size = 10

keep_rate = 0.9


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


class SignUp(generic.CreateView):
    form_class = CustomUserCreationForm
    success_url = reverse_lazy('login')
    template_name = 'signup.html'


def prediction(request):
    if request.method == 'POST':

        f = request.FILES['file']
        if os.path.isdir(data_dir):
            shutil.rmtree(data_dir)
            print("folder deleted!!")
        else:
            os.mkdir(data_dir)
            print('folder created!!')
        with zipfile.ZipFile(f, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            print('extracted successfully!!')
        prepareImage()
        much_data = np.load("much_data.npy")
        pred_x = np.array([])
        try:
            for data in much_data:
                pred_x = data[0]
        except Exception:
            pass
        tf.reset_default_graph()
        pred_x = np.reshape(pred_x, [-1, IMG_PX_SIZE, IMG_PX_SIZE, HM_SLICES, 1])
        x = tf.placeholder('float')
        weights = {'W_conv1': tf.Variable(tf.random_normal([3, 3, 3, 1, 32])),
                   #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
                   'W_conv2': tf.Variable(tf.random_normal([3, 3, 3, 32, 64])),
                   #                                  64 features
                   'W_conv3': tf.Variable(tf.random_normal([3, 3, 3, 64, 128])),
                   'W_fc': tf.Variable(tf.random_normal([64000, 1024])),
                   'W_fc2': tf.Variable(tf.random_normal([1024, 1024])),
                   'out': tf.Variable(tf.random_normal([1024, n_classes]))}

        biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
                  'b_conv2': tf.Variable(tf.random_normal([64])),
                  'b_conv3': tf.Variable(tf.random_normal([128])),
                  'b_fc': tf.Variable(tf.random_normal([1024])),
                  'b_fc2': tf.Variable(tf.random_normal([1024])),
                  'out': tf.Variable(tf.random_normal([n_classes]))}
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, home_dir + 'savedModel')
            #                            image X      image Y        image Z

            x = tf.reshape(x, shape=[-1, IMG_PX_SIZE, IMG_PX_SIZE, HM_SLICES, 1])

            conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
            conv1 = maxpool3d(conv1)

            conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
            conv2 = maxpool3d(conv2)

            conv3 = tf.nn.relu(conv3d(conv2, weights['W_conv3']) + biases['b_conv3'])
            conv2 = maxpool3d(conv3)

            fc = tf.reshape(conv2, [-1, 64000])
            fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
            fc = tf.nn.dropout(fc, keep_rate)

            fc2 = tf.nn.relu(tf.matmul(fc, weights['W_fc2']) + biases['b_fc2'])

            output = tf.matmul(fc2, weights['out']) + biases['out']

            result = sess.run(tf.argmax(output, 1)[0], feed_dict={x: pred_x})

            if result == 0:
                return HttpResponse('patient is healthy')
            elif result == 1:
                return HttpResponse('patient is suspected to have cancer')
            else:
                return HttpResponse('unknown result')


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def med(l):
    return sum(l) / len(l)

def visualizeFn():


def process_data(patient, labels_df, img_px_size=50, hm_slices=20, visualize=False):
    label = labels_df.loc[labels_df['PatientID'] == patient]

    label = label['Label']
    label = np.array(label);
    label = label[0]

    seriries_id = os.listdir(data_dir + patient + '/')

    path = data_dir + patient + '/' + seriries_id[0]

    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key=lambda x: int(x.SliceLocation))

    new_slices = []

    slices = [cv2.resize(np.array(each_slice.pixel_array), (IMG_PX_SIZE, IMG_PX_SIZE)) for each_slice in slices]

    # ******************************To Make Slices Devisable by X***************

    sizeOfSlices = len(slices)
    for i in range(sizeOfSlices):
        if (sizeOfSlices % HM_SLICES == 0):

            break;
        else:

            slices.pop()
            sizeOfSlices = len(slices)

    # **************************************************************************

    chunk_sizes = math.ceil(len(slices) / HM_SLICES)
    print(chunk_sizes)
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(med, zip(*slice_chunk)))

        new_slices.append(slice_chunk)

    if visualize:
        fig = plt.figure()
        for num, each_slice in enumerate(new_slices):
            y = fig.add_subplot(4, 5, num + 1)
            y.imshow(each_slice, cmap='gray')
        plt.show()

    # print('LABEL___: ',label)
    if label == 1:
        label = np.array([0, 1])
    elif label == 0:
        label = np.array([1, 0])

    return np.array(new_slices), label


def prepareImage():
    patients = os.listdir(data_dir)

    much_data = []

    for patient in tqdm(patients[0:20]):
        # if num % 10 == 0:
        #     print(num)
        try:
            img_data, label = process_data(patient, labels, img_px_size=IMG_PX_SIZE, hm_slices=HM_SLICES)
            # print(img_data.shape,label)
            much_data.append([img_data, label])
        except KeyError as e:

            print('This is unlabeled data!')

    np.save("much_data.npy", much_data);


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def maxpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
