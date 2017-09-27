import cv2
import numpy as np
import os


def scaling(arr):
    dif_min = 0 - arr.min()
    arr = arr + dif_min
    arr = arr * 255 / arr.max()
    return arr


def cov(x):
    """
    compute eigenvalue & eigenvector

    :param x: flatten array
    :return: covariance matrix
    """
    x = x.T                     # observation, variable = (pixels, images)
    ob, var = x.shape
    mean_x = np.mean(x, axis=0)
    x_ = x - mean_x
    cov_x = np.dot(x_.T, x_) / (ob - 1)

    return cov_x


def save_images(y, shape, name='default'):
    y = np.reshape(y, shape)
    #y = y.astype(np.uint8)
    for idx in range(shape[0]):
        cv2.imwrite(str(name)+'_image'+str(idx)+'.png', y[idx, :, :])


def main():
    img = np.array([cv2.imread(idx,cv2.IMREAD_GRAYSCALE)
                    for idx in ['images/Fig1138_a.tif',
                                'images/Fig1138_b.tif',
                                'images/Fig1138_c.tif',
                                'images/Fig1138_d.tif',
                                'images/Fig1138_e.tif',
                                'images/Fig1138_f.tif']],
                 dtype=np.float)
    shape = img.shape
    x = np.reshape(img, (6,-1))
    # cov_x = np.cov(x, rowvar=True)
    cov_x = cov(x)
    e, e_vec = np.linalg.eig(cov_x)        # decending order
    trans = e_vec.T

    trans[0] = -trans[0]
    trans[2] = -trans[2]
    trans[3] = -trans[3]
    trans[4] = -trans[4]

    x_m = x.copy()
    mean_x = np.mean(x, axis=1)
    for idx in range(mean_x.shape[0]):
        x_m[idx, :] = x_m[idx, :] - mean_x[idx]

    # Principle Component images
    y = np.dot(trans, x_m)
    y_ = scaling(y)
    save_images(y_, shape, name='PCI')

    # Reconstruction with 2 PCI
    num = 2
    x_ = np.dot(trans[0:num].T, y[0:num])
    for idx in range(mean_x.shape[0]):
        x_[idx, :] = x_[idx, :] + mean_x[idx]
    save_images(x_, shape, name='Rec'+str(num))

    # Difference between origin and reconstructed images
    x_ = scaling(x_)
    d_ = x - x_
    save_images(d_, shape, name='Diff')

if __name__=='__main__':
    main()