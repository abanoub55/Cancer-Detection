import tensorflow as tf
import numpy as np
import cv2
import os
from tqdm import tqdm
from random import shuffle

image_size = 100

batch_size = 2

n_classes = 2

x = tf.placeholder('float')

y = tf.placeholder('float')




def get_training_data():
    NORMAL = os.listdir('D:/FCI_STUDY/Fourth year Fci_1/IT498 - Project/projects/lung bacteria/train_normal')
    PNEUMONIA = os.listdir('D:/FCI_STUDY/Fourth year Fci_1/IT498 - Project/projects/lung bacteria/train_pneumonia')

    imgsN = []
    imgsP = []

    for (img_n, img_p) in tqdm(zip(NORMAL[0:50], PNEUMONIA[0:50])):

        img_n = cv2.imread('D:/FCI_STUDY/Fourth year Fci_1/IT498 - Project/projects/lung bacteria/train_normal/' + img_n,cv2.IMREAD_GRAYSCALE)
        img_p = cv2.imread('D:/FCI_STUDY/Fourth year Fci_1/IT498 - Project/projects/lung bacteria/train_pneumonia/' + img_p,cv2.IMREAD_GRAYSCALE)

        if len(img_n.shape) == 2 and len(img_p.shape) == 2:
            img_n = np.array(cv2.resize(img_n, (image_size, image_size)))
            img_p = np.array(cv2.resize(img_p, (image_size, image_size)))



            imgsN.append([img_n, np.array([0, 1])])
            imgsP.append([img_p, np.array([1, 0])])

    for i in range(len(imgsP)) :
        imgsN.append(imgsP[i])



    shuffle(imgsN)

    return imgsN


def get_testing_data():
    NORMAL = os.listdir('D:/FCI_STUDY/Fourth year Fci_1/IT498 - Project/projects/lung bacteria/train_normal')
    PNEUMONIA = os.listdir('D:/FCI_STUDY/Fourth year Fci_1/IT498 - Project/projects/lung bacteria/train_pneumonia')

    imgsN = []
    imgsP = []

    for (img_n, img_p) in tqdm(zip(NORMAL[:50], PNEUMONIA[:50])):
        print(img_n, '  ' , img_p)

        img_n = cv2.imread('D:/FCI_STUDY/Fourth year Fci_1/IT498 - Project/projects/lung bacteria/train_normal/' + img_n,cv2.IMREAD_GRAYSCALE)
        img_p = cv2.imread('D:/FCI_STUDY/Fourth year Fci_1/IT498 - Project/projects/lung bacteria/train_pneumonia/' + img_p,cv2.IMREAD_GRAYSCALE)

        if len(img_n.shape) == 2 and len(img_p.shape) == 2:
            img_n = np.array(cv2.resize(img_n, (image_size, image_size)))
            img_p = np.array(cv2.resize(img_p, (image_size, image_size)))



            imgsN.append([img_n, np.array([0, 1])])
            imgsP.append([img_p, np.array([1, 0])])

    for i in range(len(imgsP)) :
        imgsN.append(imgsP[i])



    shuffle(imgsN)

    return imgsN


train_x= get_training_data()
test_x = get_testing_data()


X_Train = np.array([i[0] for i in train_x]).reshape(-1,image_size,image_size,1)
Y_Train = [i[1] for i in train_x]

X_Test = np.array([i[0] for i in test_x]).reshape(-1,image_size,image_size,1)
Y_Test = [i[1] for i in test_x]


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



def neural_network(x):

    weights = {
            'W1_conv':tf.Variable(tf.random_normal([5,  5, 1, 32])),
            'W2_conv':tf.Variable(tf.random_normal([5,  5,32, 64])),
            'W3_conv':tf.Variable(tf.random_normal([5, 5,64, 128])),
            'W_fc':tf.Variable(tf.random_normal([13*13*128, 256])),
            'out':tf.Variable(tf.random_normal([256,n_classes]))
            }

    biases = {
        'b1': tf.Variable(tf.random_normal([32])),
        'b2': tf.Variable(tf.random_normal([64])),
        'b3': tf.Variable(tf.random_normal([128])),
        'b_fc': tf.Variable(tf.random_normal([256])),
        'b_out': tf.Variable(tf.random_normal([n_classes]))
            }

    x = tf.reshape(x , shape = [-1, image_size, image_size, 1])

    convNet = conv2d(x,weights['W1_conv'])

    convNet = tf.nn.relu(convNet + biases['b1'])

    convNet = maxpool2d(convNet)



    convNet = conv2d(convNet, weights['W2_conv'])

    convNet = tf.nn.relu(convNet + biases['b2'])

    convNet = maxpool2d(convNet)



    convNet = conv2d(convNet, weights['W3_conv'])

    convNet = tf.nn.relu(convNet + biases['b3'])

    convNet = maxpool2d(convNet)


    convNet =  tf.reshape(convNet,shape = [-1,13*13*128])


    fc = tf.matmul(convNet, weights['W_fc'])+biases['b_fc']

    fc = tf.nn.relu(fc)

    out = tf.matmul(fc,weights['out']) + biases['b_out']

    return out


def train(x):
    prediction = neural_network(x)

    cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y)

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    numOfepochs = 5

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for counter in tqdm(range(numOfepochs)):
            loss = 0
            i = 0
            while i < len(X_Train):
                start = i
                end = i + batch_size
                batch_x = X_Train[start:end]
                batch_y = Y_Train[start:end]
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                loss += c
                i += batch_size

            print('Epoch', counter+1, 'completed out of', numOfepochs, 'loss', loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x: X_Test, y: Y_Test}))

train(x)