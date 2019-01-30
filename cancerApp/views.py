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
import scipy.ndimage
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



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
        weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,32])),
               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
               'W_conv2':tf.Variable(tf.random_normal([3,3,3,32,64])),
               #                                  64 features
               'W_conv3':tf.Variable(tf.random_normal([3,3,3,64,32])),
               
               'W_conv4':tf.Variable(tf.random_normal([3,3,3,32,128])),
               
               'W_fc':tf.Variable(tf.random_normal([9600,1024])),
               'W_fc2':tf.Variable(tf.random_normal([1024,512])),

               'out':tf.Variable(tf.random_normal([512, n_classes]))}

        biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
                   'b_conv2':tf.Variable(tf.random_normal([64])),
                   'b_conv3':tf.Variable(tf.random_normal([32])),
                   'b_conv4':tf.Variable(tf.random_normal([128])),

                   'b_fc':tf.Variable(tf.random_normal([1024])),
                   'b_fc2':tf.Variable(tf.random_normal([512])),
                   'out':tf.Variable(tf.random_normal([n_classes]))}
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
            conv3 = maxpool3d(conv3)

            conv4 = tf.nn.relu(conv3d(conv3, weights['W_conv4']) + biases['b_conv4'])
            conv4 = maxpool3d(conv4)


            fc = tf.reshape(conv4,[-1, 9600])
            fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])

            fc2 = tf.nn.relu(tf.matmul(fc, weights['W_fc2'])+biases['b_fc2'])
            #fc2 = tf.nn.dropout(fc2, keep_rate)

            output = tf.matmul(fc2, weights['out'])+biases['out']

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









def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)








def resample(image_o,slices):
    # Determine current pixel spacing
    
    thickness = [slices[0].SliceThickness]
    
    Pixelspacing = list(slices[0].PixelSpacing)
    
    print(type(thickness))
    print(type(Pixelspacing))
    
    
    result = thickness + Pixelspacing
    print(result)
    spacing = np.array(result, dtype=np.float32)

    image = scipy.ndimage.interpolation.zoom(image_o, spacing, mode='nearest')
    
    return image






def plot_3d(image, threshold=-300):
    
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)
    
    verts, faces, normals, values = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
        
    plt.savefig('templates/static/cancerApp/img/lungfig.jpg')

    





def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image




def visualizeFn(request):

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
        patient = os.listdir(data_dir)[0]
        seriries_id = os.listdir(data_dir+patient+'/')

        path = data_dir + patient +'/'+ seriries_id[0]
        print(path)
        
        slices = []
        
        slices = [pydicom.read_file(path+'/' + s) for s in os.listdir(path)]

        slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

        
        image_o = get_pixels_hu(slices);
        print('Done HU')
        pix_resampled = resample(image_o , slices)
        print('Done Resample')
        segmented_lungs = segment_lung_mask(pix_resampled, False)
        print('Done Segment')

        plot_3d(segmented_lungs, 0)
        print('Done Plot_3d')
        return HttpResponse("Done..")


def process_data(patient,labels_df,img_px_size=50, hm_slices=20, visualize=False):
    label = labels_df.loc[labels_df['PatientID'] == patient]
    
    label = label['Label']
    label = np.array(label);
    label = label[0]
        
    seriries_id = os.listdir(data_dir+patient+'/')
        
    path = data_dir + patient +'/'+ seriries_id[0]
        
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
        
    slices.sort(key = lambda x: int(x.SliceLocation))
        
    new_slices = []
       
    slices = [cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE)) for each_slice in slices]
        
        
    #******************************To Make Slices Devisable by X***************
        
    sizeOfSlices = len(slices)
    
    
    #**************************************************************************
    #------------------NEW---------------------------------
    subSample_factor = np.floor(sizeOfSlices  / HM_SLICES)
    slice_index = 0
    #------------------NEW---------------------------------
    
        
        
    for i in range(0,HM_SLICES):
        new_slices.append(slices[slice_index])
        slice_index += subSample_factor
        slice_index = int(slice_index)
        
    
    if visualize:
        fig = plt.figure()
        for num,each_slice in enumerate(new_slices):
            y = fig.add_subplot(4,5,num+1)
            y.imshow(each_slice, cmap='gray')
        plt.show()

    #print('LABEL___: ',label)
    if label == 1: label=np.array([0,1])
    elif label == 0: label=np.array([1,0])
        
    return np.array(new_slices),label






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
