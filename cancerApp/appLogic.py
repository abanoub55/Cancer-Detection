import scipy.ndimage
from django.http import HttpResponse
from .models import Statistics
import scipy.ndimage
import pandas as pd  # for some simple data analysis (right now, just to load in the labels data and quickly reference it)
from tqdm import tqdm
import cv2
import numpy as np
import tensorflow as tf
import os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.misc
import shutil
import zipfile
import pydicom  # for reading dicom files

home_dir = 'project_data/'
data_dir = home_dir + 'patients/'
labels = pd.read_csv('FinalLabels.csv')

IMG_PX_SIZE = 80

HM_SLICES = 40

n_classes = 2

batch_size = 10

keep_rate = 0.9


# this function is fired from ajax in front-end to get prediction results
# and print them on the screen

def prediction(request):
    if request.method == 'POST':
        f = request.FILES['file']
        pid = str(f).split('.')[0]
        if not os.path.isdir("project_data"):
            os.mkdir("project_data")
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
        much_data = np.load("much_data.npy", allow_pickle=True)
        pred_x = np.array(much_data[0][0])
        tf.reset_default_graph()
        pred_x = np.reshape(pred_x, [-1, IMG_PX_SIZE, IMG_PX_SIZE, HM_SLICES, 1])
        x = tf.placeholder('float')
        weights, biases = defineCnn()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, save_path='./savedModel')
            result = feedForward(x, weights, biases, pred_x, sess)
            if len(Statistics.get_stats(patient_id=pid, username=request.user.username)) == 0:
                stat = Statistics()
                stat.username = request.user.username
                stat.patient_id = pid
                if result == 0:
                    stat.label = "Nocancer"
                else:
                    stat.label = "Cancer"
                stat.save()

            if result == 0:
                return HttpResponse('patient is healthy')
            elif result == 1:

                return HttpResponse('patient is suspected to have cancer')
            else:
                return HttpResponse('unknown result')


#####################################################################
# preprocessing the incoming patient images before making prediction
#####################################################################

def process_data(patient, labels_df, img_px_size=50, hm_slices=20, visualize=False):
    label = labels_df.loc[labels_df['PatientID'] == patient]

    label = label['Label']
    label = np.array(label)
    label = label[0]

    seriries_id = os.listdir(data_dir + patient + '/')

    path = data_dir + patient + '/' + seriries_id[0]

    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key=lambda x: int(x.SliceLocation))

    new_slices = []

    slices = [cv2.resize(np.array(each_slice.pixel_array), (IMG_PX_SIZE, IMG_PX_SIZE)) for each_slice in slices]

    # ******************************To Make Slices Devisable by X***************

    sizeOfSlices = len(slices)

    # **************************************************************************
    # ------------------NEW---------------------------------
    subSample_factor = np.floor(sizeOfSlices / HM_SLICES)
    slice_index = 0
    # ------------------NEW---------------------------------

    for i in range(0, HM_SLICES):
        new_slices.append(slices[slice_index])
        slice_index += subSample_factor
        slice_index = int(slice_index)
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

    np.save("much_data.npy", much_data)


#############################
# model preparation functions
############################

def feedForward(x, weights, biases, pred_x,sess):

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

    fc = tf.reshape(conv4, [-1, 9600])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc2 = tf.nn.relu(tf.matmul(fc, weights['W_fc2']) + biases['b_fc2'])
    # fc2 = tf.nn.dropout(fc2, keep_rate)

    output = tf.matmul(fc2, weights['out']) + biases['out']

    result = sess.run(tf.argmax(output, 1)[0], feed_dict={x: pred_x})
    return result

def defineCnn():
    weights = {'W_conv1': tf.Variable(tf.random_normal([3, 3, 3, 1, 32])),
               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
               'W_conv2': tf.Variable(tf.random_normal([3, 3, 3, 32, 64])),
               #                                  64 features
               'W_conv3': tf.Variable(tf.random_normal([3, 3, 3, 64, 32])),

               'W_conv4': tf.Variable(tf.random_normal([3, 3, 3, 32, 128])),

               'W_fc': tf.Variable(tf.random_normal([9600, 1024])),
               'W_fc2': tf.Variable(tf.random_normal([1024, 512])),

               'out': tf.Variable(tf.random_normal([512, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_conv3': tf.Variable(tf.random_normal([32])),
              'b_conv4': tf.Variable(tf.random_normal([128])),

              'b_fc': tf.Variable(tf.random_normal([1024])),
              'b_fc2': tf.Variable(tf.random_normal([512])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    return weights,biases

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def maxpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


##############################################################
# visualization functions fired from front-end (rib structure-lung structure)
##############################################################


def ribVisualize(request):
    if request.method == 'POST':
        print("Rib")
        f = request.FILES['file']
        if not os.path.isdir("project_data"):
            os.mkdir("project_data")
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
        seriries_id = os.listdir(data_dir + patient + '/')

        path = data_dir + patient + '/' + seriries_id[0]
        print(path)

        slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]

        slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
        image_o = get_pixels_hu(slices)
        print('Done HU')
        pix_resampled = resample(image_o, slices)
        print('Done Resample')

        plot_3d(pix_resampled, 400)
        print('Done Plot_3d')
        return HttpResponse("Rib Structure")


def lungStructure(request):
    if request.method == 'POST':
        f = request.FILES['file']
        if not os.path.isdir("project_data"):
            os.mkdir("project_data")
        if os.path.isdir(data_dir):
            shutil.rmtree(data_dir)
        else:
            os.mkdir(data_dir)
        with zipfile.ZipFile(f, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        patient = os.listdir(data_dir)[0]
        seriries_id = os.listdir(data_dir + patient + '/')

        path = data_dir + patient + '/' + seriries_id[0]
        slices = []

        slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]

        slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

        image_o = get_pixels_hu(slices)
        print('Done HU')
        pix_resampled = resample(image_o, slices)
        print('Done Resample')
        segmented_lungs = segment_lung_mask(pix_resampled, True)
        print('Done Segment')

        plot_3d(segmented_lungs, 0)
        print('Done Plot_3d')
        return HttpResponse("Lung Structure")


def cancer_spread(request):
    if request.method == 'POST':
        f = request.FILES['file']
        if not os.path.isdir("project_data"):
            os.mkdir("project_data")
        if os.path.isdir(data_dir):
            shutil.rmtree(data_dir)
        else:
            os.mkdir(data_dir)
        with zipfile.ZipFile(f, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        patient = os.listdir(data_dir)[0]
        seriries_id = os.listdir(data_dir + patient + '/')
        patients = Statistics.get_stats(patient_id=patient, username=request.user.username)
        if len(patients)!= 0:
            print('# of patients',len(patients))
            print('patient found in stats')
            lbl = patients[0].label
            if lbl == 'Cancer':
                print('label = 1')
                path = data_dir + patient + '/' + seriries_id[0]
                slices = read_ct_scan(path)
                segmented_ct_scan = segment_lung_from_ct_scan(slices)
                segmented_ct_scan[segmented_ct_scan < 604] = 0
                # from morphology
                selem = ball(2)
                binary = binary_closing(segmented_ct_scan, selem)

                label_scan = label(binary)

                areas = [r.area for r in regionprops(label_scan)]
                areas.sort()

                for r in regionprops(label_scan):
                    max_x, max_y, max_z = 0, 0, 0
                    min_x, min_y, min_z = 1000, 1000, 1000

                    for c in r.coords:
                        max_z = max(c[0], max_z)
                        max_y = max(c[1], max_y)
                        max_x = max(c[2], max_x)

                        min_z = min(c[0], min_z)
                        min_y = min(c[1], min_y)
                        min_x = min(c[2], min_x)
                    if min_z == max_z or min_y == max_y or min_x == max_x or r.area > areas[-3]:
                        for c in r.coords:
                            segmented_ct_scan[c[0], c[1], c[2]] = 0
                    else:
                        index = (max((max_x - min_x), (max_y - min_y), (max_z - min_z))) / (
                            min((max_x - min_x), (max_y - min_y), (max_z - min_z)))

                    plot_3d(segmented_ct_scan, 604)
                    return HttpResponse('cancerSpread')
            else :
                print('label = 0')
                return HttpResponse('healthy')
        else:
            print('patient is not found in stat')
            prepareImage()
            much_data = np.load("much_data.npy", allow_pickle=True)
            pred_x = np.array(much_data[0][0])
            tf.reset_default_graph()
            pred_x = np.reshape(pred_x, [-1, IMG_PX_SIZE, IMG_PX_SIZE, HM_SLICES, 1])
            x = tf.placeholder('float')
            weights, biases = defineCnn()
            saver = tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(sess, save_path='./savedModel')
                result = feedForward(x, weights, biases, pred_x, sess)
                print('label = ',result)
                if result == 0:
                    return HttpResponse('healthy')
                else:
                    path = data_dir + patient + '/' + seriries_id[0]
                    slices = read_ct_scan(path)
                    segmented_ct_scan = segment_lung_from_ct_scan(slices)
                    segmented_ct_scan[segmented_ct_scan < 604] = 0
                    # from morphology
                    selem = ball(2)
                    binary = binary_closing(segmented_ct_scan, selem)

                    label_scan = label(binary)

                    areas = [r.area for r in regionprops(label_scan)]
                    areas.sort()

                    for r in regionprops(label_scan):
                        max_x, max_y, max_z = 0, 0, 0
                        min_x, min_y, min_z = 1000, 1000, 1000

                        for c in r.coords:
                            max_z = max(c[0], max_z)
                            max_y = max(c[1], max_y)
                            max_x = max(c[2], max_x)

                            min_z = min(c[0], min_z)
                            min_y = min(c[1], min_y)
                            min_x = min(c[2], min_x)
                        if min_z == max_z or min_y == max_y or min_x == max_x or r.area > areas[-3]:
                            for c in r.coords:
                                segmented_ct_scan[c[0], c[1], c[2]] = 0
                        else:
                            index = (max((max_x - min_x), (max_y - min_y), (max_z - min_z))) / (
                                min((max_x - min_x), (max_y - min_y), (max_z - min_z)))

                        plot_3d(segmented_ct_scan, 604)
                        return HttpResponse('cancerSpread')



##########################################################
# visualization preprocessing functions and plotting
##########################################################


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


def resample(image_o, slices):
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

    verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)

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
    plt.clf()



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
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image

def get_segmented_lungs(im, plot=False):
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < 604

    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    #from segmentation
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    #from morphology
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone)

    return im



def segment_lung_from_ct_scan(ct_scan):
    return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])


def read_ct_scan(folder_name):
    # Read the slices from the dicom file
    slices = [pydicom.read_file(folder_name + '/' + filename) for filename in os.listdir(folder_name)];

    # Sort the dicom slices in their respective order
    slices.sort(key=lambda x: int(x.InstanceNumber));
    slices = np.stack(
        [cv2.resize(np.array(each_slice.pixel_array), (IMG_PX_SIZE, IMG_PX_SIZE)) for each_slice in slices])

    # Get the pixel values for all the slices
    # slices = np.stack([s.pixel_array for s in slices]);
    slices[slices == -2000] = 0;
    return slices

def getIndex(pid):
    i = len(pid)-1
    str =''
    while i >= 0:
        if pid[i] != '0':
            str = pid[i] +str
        else:
            break
        i -= 1
    return int(str)-1

##################################################################
# statistics related functions (cancer-gender-age)
##################################################################

# stats for cancer
def cancerStats(request):
    labels = ['cancer', 'non-cancer']
    cancer = Statistics.get_stats(username=request.user.username,label='Cancer')
    nocancer = Statistics.get_stats(username=request.user.username, label='Nocancer')
    values = [len(cancer), len(nocancer)]
    if len(cancer) == 0 and len(nocancer) == 0:
        return HttpResponse("user has no activity yet")
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

    def func(pct, allvals):
        absolute = int(pct / 100. * np.sum(allvals))
        return "{:.1f}%".format(pct, absolute)

    wedges, texts, autotexts = ax.pie(values, autopct=lambda pct: func(pct, values),
                                      textprops=dict(color="w"))

    ax.legend(wedges, labels,
              title="labels",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=8, weight="bold")

    ax.set_title("Cancer stats")
    plt.savefig("templates/static/cancerApp/img/cancer_chart.jpg")
    plt.clf()
    return HttpResponse("stats are ready!")


# checks if an element is in a list
def isFound(el, listt):
    i = 0
    while i < len(listt):

        if el == listt[i]:
            return i
        i += 1
    return -1


# stats for gender
def genderStats(request):
    patients = Statistics.get_stats(username=request.user.username)
    p_ids = [i.patient_id for i in patients]  # select patient ids only from statistics objects
    global labels
    all_genders = labels['Gender']
    all_p_ids = labels['PatientID']
    genders = []
    i = 0
    for p_id in p_ids:
        print("im in!")
        print("p_id = " + p_id)
        ind = isFound(p_id, all_p_ids)
        if (ind != -1):
            print("found one!")
            print("all_genders[i] = " + all_genders[ind])
            genders.append(all_genders[ind])
        i += 1

    fig_labels = ['Male', 'Female']
    males = [i for i in genders if i == 'Male']
    females = [i for i in genders if i == 'Female']
    values = [len(males), len(females)]
    if len(males) == 0 and len(females) == 0:
        return HttpResponse("user has no activity yet")
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

    def func(pct, allvals):
        absolute = int(pct / 100. * np.sum(allvals))
        return "{:.1f}%".format(pct, absolute)

    theme = plt.get_cmap('copper')
    ax.set_prop_cycle("color", [theme(1. * i / 2) for i in range(3)])
    wedges, texts, autotexts = ax.pie(values, autopct=lambda pct: func(pct, values),
                                      textprops=dict(color="W"))

    ax.legend(wedges, fig_labels,
              title="labels",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title("Gender stats")
    plt.savefig("templates/static/cancerApp/img/gender_chart.jpg")
    plt.clf()
    return HttpResponse("gender Stats success!")


# stats for age
def ageStats(request):
    patients = Statistics.get_stats(username=request.user.username)
    p_ids = [i.patient_id for i in patients]  # select patient ids only from statistics objects
    global labels
    all_ages = labels['Age']
    all_p_ids = labels['PatientID']
    ages = []
    i = 0
    for p_id in p_ids:
        ind = isFound(p_id, all_p_ids)
        if ind != -1:
            ages.append(all_ages[ind])
        i += 1

    bins = range(0, 120, 20)
    if len(ages) == 0:
        return HttpResponse("user has no activity yet")
    plt.hist(ages, bins=bins)
    plt.ylabel('Frequency')
    plt.xlabel('Age')
    plt.title("Patients\' ages Histogram")
    plt.savefig("templates/static/cancerApp/img/age_chart.jpg")
    plt.clf()
    return HttpResponse("age Stats success!")


##################################################################
# statistics related functions (cancer-gender-age)
##################################################################

# stats for cancer
def cancerStats(request):
    labels = ['cancer', 'non-cancer']
    cancer = Statistics.get_stats(username=request.user.username, label="Cancer")
    nocancer = Statistics.get_stats(username=request.user.username, label="Nocancer")

    values = [len(cancer), len(nocancer)]
    if len(cancer) == 0 and len(nocancer) == 0:
        return HttpResponse("user has no activity yet")
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

    def func(pct, allvals):
        absolute = int(pct / 100. * np.sum(allvals))
        return "{:.1f}%".format(pct, absolute)

    wedges, texts, autotexts = ax.pie(values, autopct=lambda pct: func(pct, values),
                                      textprops=dict(color="w"))

    ax.legend(wedges, labels,
              title="labels",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=8, weight="bold")

    ax.set_title("Cancer stats")
    plt.savefig("templates/static/cancerApp/img/cancer_chart.jpg")
    plt.clf()
    return HttpResponse("stats are ready!")


# checks if an element is in a list
def isFound(el, listt):
    i = 0
    while i < len(listt):

        if el == listt[i]:
            return i
        i += 1
    return -1


# stats for gender
def genderStats(request):
    patients = Statistics.get_stats(username=request.user.username)
    p_ids = [i.patient_id for i in patients]  # select patient ids only from statistics objects
    global labels
    all_genders = labels['Gender']
    all_p_ids = labels['PatientID']
    genders = []
    i = 0
    for p_id in p_ids:
        print("im in!")
        print("p_id = " + p_id)
        ind = isFound(p_id, all_p_ids)
        if (ind != -1):
            print("found one!")
            print("all_genders[i] = " + all_genders[ind])
            genders.append(all_genders[ind])
        i += 1

    fig_labels = ['Male', 'Female']
    males = [i for i in genders if i == 'Male']
    females = [i for i in genders if i == 'Female']
    values = [len(males), len(females)]
    if len(males) == 0 and len(females) == 0:
        return HttpResponse("user has no activity yet")
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

    def func(pct, allvals):
        absolute = int(pct / 100. * np.sum(allvals))
        return "{:.1f}%".format(pct, absolute)

    theme = plt.get_cmap('copper')
    ax.set_prop_cycle("color", [theme(1. * i / 2) for i in range(3)])
    wedges, texts, autotexts = ax.pie(values, autopct=lambda pct: func(pct, values),
                                      textprops=dict(color="W"))

    ax.legend(wedges, fig_labels,
              title="labels",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title("Gender stats")
    plt.savefig("templates/static/cancerApp/img/gender_chart.jpg")
    plt.clf()
    return HttpResponse("gender Stats success!")


# stats for age
def ageStats(request):
    patients = Statistics.get_stats(username=request.user.username)
    p_ids = [i.patient_id for i in patients]  # select patient ids only from statistics objects
    global labels
    all_ages = labels['Age']
    all_p_ids = labels['PatientID']
    ages = []
    i = 0
    for p_id in p_ids:
        ind = isFound(p_id, all_p_ids)
        if ind != -1:
            ages.append(all_ages[ind])
        i += 1

    bins = range(0, 120, 20)
    if len(ages) == 0:
        return HttpResponse("user has no activity yet")
    plt.hist(ages, bins=bins)
    plt.ylabel('Frequency')
    plt.xlabel('Age')
    plt.title("Patients\' ages Histogram")
    plt.savefig("templates/static/cancerApp/img/age_chart.jpg")
    plt.clf()
    return HttpResponse("age Stats success!")


# clears all statistics records for the current user
def clearHistory(request):
    Statistics.get_stats(username=request.user.username).delete()
    return HttpResponse('History cleared!')