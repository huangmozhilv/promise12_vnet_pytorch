import os
from os import listdir
from os.path import isfile, join, splitext, basename
import csv

import numpy as np
import SimpleITK as sitk
from glob import glob

msdPath = '/Volumes/17816861965/msd'

# taskF = 'Task02_Heart'
meta = dict()

meta['task'] = list()
for taskF in listdir(msdPath):
    for f in glob(join(msdPath, taskF, 'imagesTr', '*.nii.gz')):
        meta['task'].append(taskF)

for taskF in listdir(msdPath):
    for f in glob(join(msdPath, taskF, 'imagesTr', '*.nii.gz')):
        # f = '/Volumes/17816861965/msd/Task02_Heart/imagesTr/la_003.nii.gz'
        sitk_image = sitk.ReadImage(f)
        meta['task'] = taskF
        for key in ['bitpix', 'datatype', 'dim[0]', 'dim[1]', 'dim[2]', 'dim[3]', 'dim[4]', 'dim[5]', 'dim[6]', 'dim[7]', 'dim_info', 'pixdim[0]', 'pixdim[1]', 'pixdim[2]', 'pixdim[3]', 'pixdim[4]', 'pixdim[5]', 'pixdim[6]', 'pixdim[7]', 'scl_inter', 'scl_slope', 'srow_x', 'srow_y', 'srow_z']: # part of sitk_image.GetMetaDataKeys()
            meta[key] = sitk_image.GetMetaData(key)
        # stats = sitk.StatisticsImageFilter()
        # stats.Execute(sitk_image) # sitk::ERROR: Pixel type: 32-bit float is not supported in 4D by N3itk6simple21StatisticsImageFilterE or SimpleITK compiled with SimpleITK_4D_IMAGES set to OFF.
        # meta['intensity_max'] = stats.GetMaximum()
        # meta['intensity_min'] = stats.GetMinimum()


def get_size_spacing(imageFolder, resultFolder, resultTag):
    imagesTrList = glob(imageFolder + '*.mhd')
    trainF = open(os.path.join(resultFolder, resultTag + '_size_spacing.csv'), 'w')
    trainF.write('{}, {}, {}, {}, {}, {}, {}\n'.format('id', 'width', 'height', 'depth', 'pixel0', 'pixel1', 'pixel2'))
    for i in imagesTrList:
        sitk_image = sitk.ReadImage(i)
        width, height, depth = sitk_image.GetSize()
        pixel0, pixel1, pixel2 = sitk_image.GetSpacing()
        trainF.write('{}, {}, {}, {}, {}, {}, {}\n'.format(basename(i).split('.')[0], width, height, depth, pixel0, pixel1, pixel2))
    trainF.close()


get_size_spacing('/Users/messi/PycharmProjects/promise12/dataset/imagesTr/', '/Users/messi/PycharmProjects/promise12/dataset/', 'imagesTr')
get_size_spacing('/Users/messi/PycharmProjects/promise12/dataset/imagesTs/', '/Users/messi/PycharmProjects/promise12/dataset/', 'imagesTs')