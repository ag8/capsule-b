import sys

sys.path.append("../")
from mmnistlib.input_utils import load_mnist, load_mmnist4
import numpy as np
from random import randint
import os
from matplotlib import pyplot as plt
from scipy import ndimage


def _mergeMNIST(img1, img2, shift_pix, rand_shift, rot_range, corot):
    assert shift_pix <= 8, "can only shift up to 8 with 32x32 MMNIST and 28x28 original MNIST"
    if corot:
        rot_deg = randint(rot_range[0], rot_range[1])
        img1 = ndimage.rotate(img1, rot_deg, reshape=False)
        img2 = ndimage.rotate(img2, rot_deg, reshape=False)
    else:
        rot_deg = randint(rot_range[0], rot_range[1])
        img1 = ndimage.rotate(img1, rot_deg, reshape=False)
        rot_deg = randint(rot_range[0], rot_range[1])
        img2 = ndimage.rotate(img2, rot_deg, reshape=False)

    if rand_shift:
        shift_pix = randint(0, shift_pix)
    # merge two images from MNIST into one
    canvas = np.zeros([36, 36])
    canvas[:28, :28] += img1
    canvas[shift_pix:shift_pix + 28, shift_pix:shift_pix + 28] += img2
    canvas = np.clip(canvas, 0, 1)
    return canvas


def _merge_many_MNIST(images, shift_pixels, rotation_range, use_shifting=True, rotate=True):
    """
    Merges an arbitrary number of MNIST images.

    :param images: a list of MNIST images
    :param shift_pixels: the (maximum) number of pixels to shift each image by
    :param rotation_range: a tuple containing the minimum and maximum rotation angle, in degrees, for each MNIST image
    :param use_shifting: whether to shift the images
    :param rotate: whether to rotate
    :return:
    """
    assert shift_pixels <= 8, "Each MNIST image can be shifted a maximum of 8 pixels, since the input images are 28x28 and the output images are 36x36. You are requesting a shift of " + (shift_pixels) + " pixels."

    if rotate:
        for i in range(0, images.__len__()):
            rot_deg = randint(rotation_range[0], rotation_range[1])
            images[i] = ndimage.rotate(images[i], rot_deg, reshape=False)

    if use_shifting:
        canvas = np.zeros([36, 36])

        for i in range(1, images.__len__()):
            shift_pixels = randint(0, shift_pixels)
            print(shift_pixels)
            canvas[shift_pixels:shift_pixels + 28, shift_pixels:shift_pixels + 28] += images[i]
            canvas = np.clip(canvas, 0, 1)
    else:
        canvas = np.zeros([36, 36])

        for i in range(1, images.__len__()):
            canvas[:28, :28] += images[i]
            canvas = np.clip(canvas, 0, 1)

    return canvas



def _MNIST2MMNIST(X, Y, num_samples, outpath, shift_pix, rand_shift=False, rot_range=[0, 0], corot=True):
    # saves MMNIST from MNIST backend
    assert X.shape[0] == Y.shape[0], "number of images and labels should be equal"
    n_MNIST = X.shape[0]

    # samples MNIST to make MMNIST
    X_MMNIST = []
    Y_MMNIST = []
    while len(Y_MMNIST) < num_samples:
        # pick two random train images with different labels
        idx1 = len(Y_MMNIST) % (n_MNIST - 1)
        idx2 = randint(0, n_MNIST - 1)
        print "debug:", Y[idx1], Y[idx2]
        if Y[idx1] != Y[idx2]:
            # merge two images together
            Y_MMNIST.append([Y[idx1], Y[idx2]])
            merged_img = _mergeMNIST(X[idx1, :, :, 0], X[idx2, :, :, 0], shift_pix=shift_pix, rand_shift=rand_shift,
                                     rot_range=rot_range, corot=corot)
            X_MMNIST.append(merged_img)
        print "next img", len(Y_MMNIST)

    # save imgMMNIST and lblMMNIST
    X_MMNIST = np.asarray(X_MMNIST)
    Y_MMNIST = np.asarray(Y_MMNIST, dtype=np.int32)  # convert label to int32
    # convert image to uint8
    X_MMNIST = 255. * X_MMNIST
    X_MMNIST = X_MMNIST.astype(np.uint8)
    # save images and labels
    np.ndarray.tofile(X_MMNIST, outpath + 'X')
    np.ndarray.tofile(Y_MMNIST, outpath + 'Y')
    return None


def _mnist_to_mmnist4(X, Y, num_samples, outpath, shift_pixels, use_shifting=False, rotation_range=None, rotate=False):
    # Initial checks
    assert X.shape[0] == Y.shape[0], "The number of images (" + X.shape[0] + ") and labels (" + Y.shape[0] + ") should be equal!"

    if rotation_range is None:
        rotation_range = [0, 0]

    num_digits = X.shape[0]


    # Sample the MNIST images to create the MMNIST dataset
    X_MMNIST = []
    Y_MMNIST = []

    while len(Y_MMNIST) < num_samples:
        # Pick four random training images with different labels
        idx1 = len(Y_MMNIST) % (num_digits - 1)
        idx2 = random(0, num_digits - 1, ignore=[idx1])
        idx3 = random(0, num_digits - 1, ignore=[idx1, idx2])
        idx4 = random(0, num_digits - 1, ignore=[idx1, idx2, idx3])  # TODO: This is, uh, suboptimal, to say the least
        print "debug:", Y[idx1], Y[idx2], Y[idx3], Y[idx4]

        assert idx1 != idx2 != idx3 != idx4

        # merge four images together
        Y_MMNIST.append([Y[idx1], Y[idx2], Y[idx3], Y[idx4]])
        images = [
            X[idx1, :, :, 0],
            X[idx2, :, :, 0],
            X[idx3, :, :, 0],
            X[idx4, :, :, 0]
        ]
        merged_img = _merge_many_MNIST(images, shift_pixels, rotation_range, use_shifting=use_shifting, rotate=rotate)
        X_MMNIST.append(merged_img)
        print "next img", len(Y_MMNIST)

    # save imgMMNIST and lblMMNIST
    X_MMNIST = np.asarray(X_MMNIST)
    Y_MMNIST = np.asarray(Y_MMNIST, dtype=np.int32)  # convert label to int32
    # convert image to uint8
    X_MMNIST = 255. * X_MMNIST
    X_MMNIST = X_MMNIST.astype(np.uint8)
    # save images and labels
    np.ndarray.tofile(X_MMNIST, outpath + 'X')
    np.ndarray.tofile(Y_MMNIST, outpath + 'Y')
    return None


def random(min, max, ignore):
    """
    Chooses a random number between min and max, that is not in the set of numbers to ignore.

    :param min: The minimum bound
    :param max: The maximum bound
    :param ignore: Numbers to ignore. Assumed to be in range [min, max]
    :return:
    """
    assert ignore.__len__() <= max - min, "I can't choose a number since you're disallowing too many!"

    while True:
        chosen = randint(min, max)
        if chosen not in ignore:
            return chosen



def genMMNIST(mnistpath, outpath, samples_tr=200000, samples_te=10000):
    trX, trY, teX, teY = load_mnist(mnistpath)
    # generate training samples
    _MNIST2MMNIST(trX, trY, samples_tr, os.path.join(outpath, 'tr'), shift_pix=8, rand_shift=True)
    # generate test samples
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'te0'), shift_pix=0)
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'te1'), shift_pix=1)
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'te2'), shift_pix=2)
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'te3'), shift_pix=3)
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'te4'), shift_pix=4)
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'te5'), shift_pix=5)
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'te6'), shift_pix=6)
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'te7'), shift_pix=7)
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'te8'), shift_pix=8)
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'teR30'), shift_pix=8, rot_range=[0, 30])
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'teR60'), shift_pix=8, rot_range=[30, 60])
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'teR30R'), shift_pix=8, rot_range=[0, 30], corot=False)
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'teR60R'), shift_pix=8, rot_range=[30, 60], corot=False)


def genMMNIST4(mnistpath, outpath, samples_tr=200000, samples_te=10000):
    trX, trY, teX, teY = load_mnist(mnistpath)
    # generate training samples
    _mnist_to_mmnist4(trX, trY, samples_tr, os.path.join(outpath, 'tr'), shift_pixels=8, use_shifting=True)
    # generate test samples
    _mnist_to_mmnist4(teX, teY, samples_te, os.path.join(outpath, 'te0'), shift_pixels=0, use_shifting=True)
    _mnist_to_mmnist4(teX, teY, samples_te, os.path.join(outpath, 'te1'), shift_pixels=1, use_shifting=True)
    _mnist_to_mmnist4(teX, teY, samples_te, os.path.join(outpath, 'te2'), shift_pixels=2, use_shifting=True)
    _mnist_to_mmnist4(teX, teY, samples_te, os.path.join(outpath, 'te3'), shift_pixels=3, use_shifting=True)
    _mnist_to_mmnist4(teX, teY, samples_te, os.path.join(outpath, 'te4'), shift_pixels=4, use_shifting=True)
    _mnist_to_mmnist4(teX, teY, samples_te, os.path.join(outpath, 'te5'), shift_pixels=5, use_shifting=True)
    _mnist_to_mmnist4(teX, teY, samples_te, os.path.join(outpath, 'te6'), shift_pixels=6, use_shifting=True)
    _mnist_to_mmnist4(teX, teY, samples_te, os.path.join(outpath, 'te7'), shift_pixels=7, use_shifting=True)
    _mnist_to_mmnist4(teX, teY, samples_te, os.path.join(outpath, 'te8'), shift_pixels=8, use_shifting=True)
    _mnist_to_mmnist4(teX, teY, samples_te, os.path.join(outpath, 'teR30R'), shift_pixels=8, rotation_range=[0, 30], use_shifting=True, rotate=True)
    _mnist_to_mmnist4(teX, teY, samples_te, os.path.join(outpath, 'teR60R'), shift_pixels=8, rotation_range=[30, 60], use_shifting=True, rotate=True)


genMMNIST4('/home/andrew/mnist', '/home/andrew/mmnist', samples_tr=2000, samples_te=10)
mmnist = load_mmnist4('/home/andrew/mmnist', samples_tr=2000, samples_te=10)

# trX, trY, teX, teY = load_mnist("/home/maksym/Projects/datasets/mnist")
# img = trX[randint(0,1000),:,:,0]
# rot_img = ndimage.interpolation.rotate(img,60.0,reshape=False)
#
# plt.imshow(rot_img)
# plt.show()

image = mmnist["teR30RX"][9][:, :, 0]
digits = mmnist["teR30RY"][9]
print "digits:", digits
plt.imshow(image)
plt.show()
# print "showitng digits:", mmnist[1][11,:]
# plt.show()
# plt.imshow(mmnist[2][11,:,:,0])
# print "showitng digits:", mmnist[3][11,:]
# plt.show()