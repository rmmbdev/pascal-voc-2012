import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from imgaug import augmenters as iaa
from skimage import io
from skimage import transform as trf
from sklearn.preprocessing import MinMaxScaler
from spectral import get_rgb, ndvi

import static_values as sv


def show(img_path):
    im = cv.imread(img_path, -1)
    im = np.asarray(im)
    size = im.shape
    fig = plt.figure()

    imgn = np.ndarray((size[0], size[1], 3), dtype=np.float)

    im2 = im[:, :, 0]
    im2 = cv.normalize(np.asarray(im2, dtype=np.float), None, 0, 1, cv.NORM_MINMAX)
    sb = fig.add_subplot(3, 2, 1)
    imgplot = plt.imshow(im2)
    sb.set_title('R')
    imgn[:, :, 0] = im2

    im2 = im[:, :, 1]
    im2 = cv.normalize(np.asarray(im2, dtype=np.float), None, 0, 1, cv.NORM_MINMAX)
    sb = fig.add_subplot(3, 2, 2)
    imgplot = plt.imshow(im2)
    sb.set_title('G')
    imgn[:, :, 1] = im2

    im2 = im[:, :, 2]
    im2 = cv.normalize(np.asarray(im2, dtype=np.float), None, 0, 1, cv.NORM_MINMAX)
    sb = fig.add_subplot(3, 2, 3)
    imgplot = plt.imshow(im2)
    sb.set_title('B')
    imgn[:, :, 2] = im2

    im2 = im[:, :, 3]
    im2 = cv.normalize(np.asarray(im2, dtype=np.float), None, 0, 1, cv.NORM_MINMAX)
    sb = fig.add_subplot(3, 2, 4)
    imgplot = plt.imshow(im2)
    sb.set_title('NIn')

    sb = fig.add_subplot(3, 2, 5)
    imgplot = plt.imshow(imgn)
    sb.set_title('RGB')

    plt.show()


def read(img_path):
    path = img_path.replace('\\', '/')
    if not os.path.exists(path):
        print('File not exist ', path)
    im = cv.imread(path, cv.IMREAD_UNCHANGED)
    im = np.asarray(im)
    return im


def read_raw_jpg(img_path):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def read_jpg(img_path):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    if (img.shape[0] == sv.STATIC_VALUES.image_size[0]) & (img.shape[1] == sv.STATIC_VALUES.image_size[1]):
        return img

    img = cv.resize(img, (sv.STATIC_VALUES.image_size[0], sv.STATIC_VALUES.image_size[1]))
    return img


def read_jpg2(img_path):
    path = img_path.replace('\\', '/')
    if not os.path.exists(path):
        print('File not exist ', path)
    im = cv.imread(path, cv.IMREAD_COLOR)
    im = cv.resize(im, (sv.STATIC_VALUES.image_size[0], sv.STATIC_VALUES.image_size[1]))
    # plt.imshow(im)

    # histogram equalization
    img_y_cr_cb = cv.cvtColor(im, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(img_y_cr_cb)

    # Applying equalize Hist operation on Y channel.
    y_eq = cv.equalizeHist(y)

    img_y_cr_cb_eq = cv.merge((y_eq, cr, cb))
    res = cv.cvtColor(img_y_cr_cb_eq, cv.COLOR_YCR_CB2BGR)
    # plt.imshow(res)

    vgg_mean = [103.939, 116.779, 123.68]
    res[:, :, 2] = res[:, :, 2] - vgg_mean[0]
    res[:, :, 1] = res[:, :, 1] - vgg_mean[1]
    res[:, :, 0] = res[:, :, 0] - vgg_mean[2]
    # plt.imshow(res)

    res = np.asarray(res)
    return res


def read_tif2(img_path):
    # image is bgrNIr
    img = io.imread(img_path)
    img = img.astype(np.float)

    # use rg NDVI
    img_new = np.zeros([256, 256, 3], np.float)
    img_new[:, :, 0] = img[:, :, 2]
    img_new[:, :, 1] = img[:, :, 1]
    img_new[:, :, 2] = (img[:, :, 3] - img[:, :, 2]) / (img[:, :, 3] + img[:, :, 2])

    """
    # replace gradients of image by g
    kernel = np.zeros([3, 3])
    kernel[1, 0] = -1
    kernel[1, 2] = 1
    img_new[:, :, 1] = cv.filter2D(img_new[:, :, 1], -1, kernel)
    """

    img2 = get_rgb(img_new, [0, 1, 2])  # RGB

    # rescaling to 0-255 range - uint8 for display
    rescaleIMG = np.reshape(img2, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 255))
    rescaleIMG = scaler.fit_transform(rescaleIMG)  # .astype(np.float32)
    img2_scaled = (np.reshape(rescaleIMG, img2.shape)).astype(np.uint8)
    res = cv.resize(img2_scaled, (sv.STATIC_VALUES.image_size[0], sv.STATIC_VALUES.image_size[0]))

    """
    vgg_mean = [103.939, 116.779, 123.68]
    res[:, :, 2] = res[:, :, 2] - vgg_mean[0]
    res[:, :, 1] = res[:, :, 1] - vgg_mean[1]
    res[:, :, 0] = res[:, :, 0] - vgg_mean[2]
    """
    return res


def read_tif(img_path):
    # image is BGRN
    img = io.imread(img_path)
    img2 = get_rgb(img, [3, 2, 1])  # NIR-R-G

    # spectral module ndvi
    img2[:, :, 0] = ndvi(img, 1, 0)

    # calculate ndvi
    # np.seterr(all='warn')
    # img2[:, :, 0] = (img2[:, :, 0] - img2[:, :, 1]) / (img2[:, :, 0] + img2[:, :, 1])

    # rescaling to 0-255 range - uint8 for display
    rescaleIMG = np.reshape(img2, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 255))
    rescaleIMG = scaler.fit_transform(rescaleIMG)  # .astype(np.float32)
    img2_scaled = (np.reshape(rescaleIMG, img2.shape)).astype(np.uint8)
    res = cv.resize(img2_scaled, (sv.STATIC_VALUES.image_size[0], sv.STATIC_VALUES.image_size[0]))
    return res


def im_show(im_mat):
    plt.figure('Img')
    plt.imshow(im_mat)
    plt.show()


def im_show_spectral(im_mat):
    plt.figure('Img')
    plt.imshow(im_mat, cmap=plt.get_cmap('nipy_spectral'))
    plt.show()


def generate_images_jpg(image_path, destination_path, file_name):
    path = image_path.replace('\\', '/')
    if not os.path.exists(path):
        print('File not exist ', path)
    image = cv.imread(path, cv.IMREAD_COLOR)

    results = []
    # results.append(image)
    rows, cols, channels = image.shape

    # rotate
    # 90
    m = cv.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    dst = cv.warpAffine(image, m, (cols, rows))
    results.append(dst)

    # 180
    m = cv.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
    dst = cv.warpAffine(image, m, (cols, rows))
    results.append(dst)

    # 270
    m = cv.getRotationMatrix2D((cols / 2, rows / 2), 270, 1)
    dst = cv.warpAffine(image, m, (cols, rows))
    results.append(dst)

    # flip
    # vertical
    flipped = cv.flip(image, 1)
    results.append(flipped)

    # horizontal
    flipped = cv.flip(image, 0)
    results.append(flipped)

    for idx, im in enumerate(results):
        path = '{}\\{}-{}.jpg'.format(destination_path, file_name, idx)
        cv.imwrite(path, im, [int(cv.IMWRITE_JPEG_QUALITY), 90])


def generate_images_tif(image_path, destination_path, file_name):
    path = image_path.replace('\\', '/')
    if not os.path.exists(path):
        print('File not exist ', path)
    image = io.imread(path)

    results = []
    # results.append(image)
    rows, cols, channels = image.shape

    # rotate
    # 90
    dst = trf.rotate(image, 90, preserve_range=True)
    results.append(dst)

    # 180
    dst = trf.rotate(image, 180, preserve_range=True)
    results.append(dst)

    # 270
    dst = trf.rotate(image, 270, preserve_range=True)
    results.append(dst)

    # flip
    # vertical
    dst = image
    dst[:, :, 0] = np.fliplr(dst[:, :, 0])
    dst[:, :, 1] = np.fliplr(dst[:, :, 1])
    dst[:, :, 2] = np.fliplr(dst[:, :, 2])
    dst[:, :, 3] = np.fliplr(dst[:, :, 3])
    results.append(dst)

    # horizontal
    dst = image
    dst[:, :, 0] = np.flipud(dst[:, :, 0])
    dst[:, :, 1] = np.flipud(dst[:, :, 1])
    dst[:, :, 2] = np.flipud(dst[:, :, 2])
    dst[:, :, 3] = np.flipud(dst[:, :, 3])
    results.append(dst)

    for idx, im in enumerate(results):
        path = '{}\\{}-{}.tif'.format(destination_path, file_name, idx)
        io.imsave(path, np.uint16(im), 'tifffile')


# TODO implement this
def calc_correlation(img_path):
    # image is BGRN
    img = io.imread(img_path)
    b = get_rgb(img, [0])
    img2 = get_rgb(img, [3, 2, 1])  # NIR-R-G

    # spectral module ndvi
    img2[:, :, 0] = ndvi(img, 1, 0)

    # rescaling to 0-255 range - uint8 for display
    rescaleIMG = np.reshape(img2, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 255))
    rescaleIMG = scaler.fit_transform(rescaleIMG)  # .astype(np.float32)
    img2_scaled = (np.reshape(rescaleIMG, img2.shape)).astype(np.uint8)
    res = cv.resize(img2_scaled, (sv.STATIC_VALUES.image_size[0], sv.STATIC_VALUES.image_size[0]))

    # rescaling to 0-255 range - uint8 for display
    rescaleIMG = np.reshape(b, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 255))
    rescaleIMG = scaler.fit_transform(rescaleIMG)  # .astype(np.float32)
    img2_scaled = (np.reshape(rescaleIMG, img2.shape)).astype(np.uint8)
    res2 = cv.resize(img2_scaled, (sv.STATIC_VALUES.image_size[0], sv.STATIC_VALUES.image_size[0]))

    cor_mat = np.corrcoef(res, res2)
    print(cor_mat)


def transformations(src, choice):
    if choice == 0:
        seq = iaa.Sequential([
            iaa.Crop(px=(0, 128)),  # crop images from each side by 0 to 16px (randomly chosen)
        ])
        src = seq.augment_images([src])[0]
    if choice == 1:
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        ])
        src = seq.augment_images([src])[0]
    if choice == 2:
        seq = iaa.Sequential([
            iaa.MedianBlur(k=3)  # blur image using local medians with kernel sizes between 2 and 7
        ])
        src = seq.augment_images([src])[0]
    if choice == 3:
        seq = iaa.Sequential([
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))  # sharpen images
        ])
        src = seq.augment_images([src])[0]
    if choice == 4:
        seq = iaa.Sequential([
            iaa.Affine(shear=(-30, 30))  # shear by -16 to +16 degrees
        ])
        src = seq.augment_images([src])[0]
    if choice == 5:
        src = src
    if choice == 6:
        src = src
    return src


if __name__ == '__main__':
    # generate_images_tif('D:\\Data\\train-tif-v2\\train_1.tif', 'D:\\Data\\aug-tif', 'train_1')
    # im = read_tif('{}train-tif-v2\\train_10.tif'.format(sv.STATIC_VALUES.base_dir))
    # im_show_spectral(im)

    im = read_jpg(sv.STATIC_VALUES.base_dir + 'images/{}.jpg'.format('2007_000027'))
    im_show(im)
    im = transformations(im, 4)
    im_show(im)

    # im_show(read_tif('D:\\Data\\aug-tif\\train_1-0.tif'))
