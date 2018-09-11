#!/usr/bin/env python3

import os
from os.path import basename

import SimpleITK as sitk
import numpy as np
import glob
from skimage import measure, draw
from PIL import Image

import matplotlib.pyplot as plt
plt.style.use('bmh') # plot style

import matplotlib as mpl
mpl.use('Agg') # non-interactive backend

from matplotlib import rcParams
rcParams.update({'figure.autolayout':True})

def plot_contours(imagePath, predPath, expFileName, expDir):

    sitk_image = sitk.ReadImage(imagePath)
    image = np.transpose(sitk.GetArrayFromImage(sitk_image), [2, 1, 0]) # image.shape 320, 320, 130
    sitk_gt = sitk.ReadImage(gtPath)
    gt = np.transpose(sitk.GetArrayFromImage(sitk_gt), [2, 1, 0]) # gt.shape 320, 320, 130
    sitk_pred = sitk.ReadImage(predPath)
    pred = np.transpose(sitk.GetArrayFromImage(sitk_pred), [2, 1, 0]) # pred.shape ???
    print('\nimage shape:{}\ngt shape:{}\npred shape:{}'.format(image.shape, gt.shape, pred.shape))

    plt.figure(figsize=(20,40)) # need to custom for how many images to show
    gtPositions = [i for i in range(image.shape[2]) if np.max(gt[...,i])==1]
    maskPositions = [i for i in range(image.shape[2]) if np.max(pred[...,i])==1]
    if gtPositions[0] in maskPositions:
        start = gtPositions[0]
    elif maskPositions[0] in gtPositions:
        start = maskPositions[0]

    if start:
        for i in range(start, start + 2):
            plt.subplot(2, 1, i + 1 - start)
            gt_contour = measure.find_contours(gt[..., i], 0.5)[0]
            pred_contour = measure.find_contours(pred[..., i], 0.5)[0]
            plt.imshow(image[..., i], cmap="gray")
            plt.plot(gt_contour[:, 1], gt_contour[:, 0], c='r', linewidth=4)
            plt.plot(pred_contour[:, 1], pred_contour[:, 0], c='g', linewidth=4)

            plt.axis('off')
        fname = os.path.join(expDir, expFileName)
        plt.savefig(fname)  # need to be above plt.show()
        plt.show()

        Image.open(fname).rotate(270, expand=True).save(
            fname)  # rotate the final image with 90 degree to get a normal view
        print('Created contoured image for {}'.format(expFileName))
    else:
        print('OMG!!: Ground truth and predicted mask are not overlapped for {}'.format(expFileName))


####################################################################
resultDir = '/Users/messi/Downloads/results/vnet.base.promise12.20180901_1623_RMSprop/'
cases = [basename(i).split('_')[0] for i in glob.glob(resultDir+'*test*.mhd')]

for case in cases:
    imagePath = '/Users/messi/PycharmProjects/promise12/dataset/imagesTr/{}.mhd'.format(case)
    gtPath = '/Users/messi/PycharmProjects/promise12/dataset/labelsTr/{}_segmentation.mhd'.format(case)
    predPath = '/Users/messi/Downloads/results/vnet.base.promise12.20180901_1623_RMSprop/{}_tested.mhd'.format(case)
    expDir = '/Users/messi/Downloads/'
    expFileName = '{}_adam.png'.format(case)

    plot_contours(imagePath, gtPath, predPath, expFileName, expDir)


##################################################################
##################################################################
# for test data which has no gt
def plot_infer_contours(imagePath, predPath, expFileName, expDir):

    sitk_image = sitk.ReadImage(imagePath)
    image = np.transpose(sitk.GetArrayFromImage(sitk_image), [2, 1, 0]) # image.shape 320, 320, 130
    sitk_pred = sitk.ReadImage(predPath)
    pred = np.transpose(sitk.GetArrayFromImage(sitk_pred), [2, 1, 0]) # pred.shape ???
    print('\nimage shape:{}\npred shape:{}'.format(image.shape, pred.shape))

    plt.figure(figsize=(20,40)) # need to custom for how many images to show
    maskPositions = [i for i in range(image.shape[2]) if np.max(pred[...,i])==1]
    start = maskPositions[0]

    if start:
        for i in range(start, start + 2):
            plt.subplot(2, 1, i + 1 - start)
            pred_contour = measure.find_contours(pred[..., i], 0.5)[0]
            plt.imshow(image[..., i], cmap="gray")
            plt.plot(pred_contour[:, 1], pred_contour[:, 0], c='g', linewidth=4)

            plt.axis('off')
        fname = os.path.join(expDir, expFileName)
        plt.savefig(fname)  # need to be above plt.show()
        plt.show()

        Image.open(fname).rotate(270, expand=True).save(
            fname)  # rotate the final image with 90 degree to get a normal view
        print('Created contoured image for {}'.format(expFileName))
    else:
        print('OMG!!: Ground truth and predicted mask are not overlapped for {}'.format(expFileName))


####################################################################
resultDir = '/Users/messi/Downloads/results/vnet.base.promise12.20180906_1419/'
cases = [basename(i).split('_')[0] for i in glob.glob(resultDir+'*infer*.mhd')]

for case in cases:
    imagePath = os.path.join('/Users/messi/PycharmProjects/promise12/dataset/imagesTs/', '{}.mhd'.format(case))
    predPath = os.path.join(resultDir, '{}_inferred.mhd'.format(case))
    expDir = '/Users/messi/Downloads/results/'
    expFileName = '{}_adam.png'.format(case)

    plot_infer_contours(imagePath, predPath, expFileName, expDir)
