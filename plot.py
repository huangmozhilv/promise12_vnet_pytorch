#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
from matplotlib import rcParams
rcParams.update({'figure.autolayout':True})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('xLabel', type=str)
    parser.add_argument('expDir', type=str)
    args = parser.parse_args()

    xLabel = args.xLabel
    expDir = args.expDir

    trainP = os.path.join(expDir, 'train.csv')
    trainData = pd.read_csv(trainP, header=None)

    trainI, trainDice, trainErr = trainData.iloc[:,[0]], trainData.iloc[:,[1]], trainData.iloc[:,[2]]

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.plot(trainI, trainDice, label='Train')
    # plt.plot(range(len(testI)), testDice, label='Test')
    plt.xlabel(xLabel)
    plt.ylabel('Dice coefficient')
    plt.legend()
    ax.set_yscale('linear')
    dice_fname = os.path.join(expDir, 'dice.png')
    plt.savefig(dice_fname)
    print('Created {}'.format(dice_fname))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.plot(trainI, trainErr, label='Train')
    # plt.plot(range(len(testI)), testErr, label='Test')
    plt.xlabel(xLabel)
    plt.ylabel('Error')
    ax.set_yscale('linear')
    plt.legend()
    err_fname = os.path.join(expDir, 'error.png')
    plt.savefig(err_fname)
    print('Created {}'.format(err_fname))

    dice_err_fname = os.path.join(expDir, 'dice-error.png')
    os.system('convert +append {} {} {}'.format(dice_fname, err_fname, dice_err_fname))
    print('Created {}'.format(dice_err_fname))

if __name__ == '__main__':
    main()
