import cv2
import numpy as np
import os


def main():
    X = np.array([cv2.imread(idx, cv2.IMREAD_GRAYSCALE) for idx in ['images/Fig1138_a.tif',
                                                                   'images/Fig1138_b.tif',
                                                                   'images/Fig1138_c.tif',
                                                                   'images/Fig1138_d.tif',
                                                                   'images/Fig1138_e.tif',
                                                                   'images/Fig1138_f.tif']])
    X = np.reshape(X, (6,-1))
    mean_X = np.mean(X, axis=0)
    X = X - mean_X
    cov_X = np.cov(X, rowvar=True)
    e, EV = np.linalg.eigh(cov_X)
    print(e)




if __name__=='__main__':
    main()