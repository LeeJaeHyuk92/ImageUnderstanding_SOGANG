import cv2
import numpy as np
import os


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


def save_images(y, shape):
    y = np.reshape(y, shape)
    for idx in range(shape[0]):
        cv2.imwrite('PCimage'+str(idx)+'.png', y[idx, :, :])
        cv2.imwrite('reverse_PCimage' + str(idx) + '.png', np.absolute(y[idx, :, :]))


def main():
    x = np.array([cv2.imread(idx, cv2.IMREAD_GRAYSCALE) for idx in ['images/Fig1138_a.tif',
                                                                   'images/Fig1138_b.tif',
                                                                   'images/Fig1138_c.tif',
                                                                   'images/Fig1138_d.tif',
                                                                   'images/Fig1138_e.tif',
                                                                   'images/Fig1138_f.tif']],
                 dtype=np.float)
    shape = x.shape
    x = np.reshape(x, (6,-1))
    # cov_x = np.cov(x, rowvar=True)
    cov_x = cov(x)
    e, e_vec = np.linalg.eigh(cov_x)        # ascending order
    trans = e_vec[::-1]

    x = x.T                     # observation, variable = (pixels, images)
    mean_x = np.mean(x, axis=0)
    x_ = x - mean_x

    y = np.dot(trans, x_)

    save_images(y, shape)


if __name__=='__main__':
    main()