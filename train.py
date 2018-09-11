#!/usr/bin/env python3
import os
from os.path import splitext
from random import sample
import time
from multiprocessing import Process, Queue
import pdb

import shutil
import setproctitle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import lossFuncs
import utils as utils

import vnet
import DataManager as DM
import customDataset
import make_graph

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def save_checkpoint(state, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)

def train_test_split(images, labels, test_proportion):
    # images and labels are both dict().
    # pdb.set_trace()
    keys = list(images.keys())
    size = len(keys)
    test_keys = sample(keys, int(test_proportion*size))
    test_images = {i: images[i] for i in keys if i in test_keys}
    test_labels = {i+'_segmentation': labels[i+'_segmentation'] for i in keys if i in test_keys} # require customization
    train_images = {i: images[i] for i in keys if i not in test_keys}
    train_labels = {i+'_segmentation': labels[i+'_segmentation'] for i in keys if i not in test_keys} # require customization
    return train_images, train_labels, test_images, test_labels

def dataAugmentation(params, args, dataQueue, numpyImages, numpyGT):

    nr_iter = args.numIterations # params['ModelParams']['numIterations']
    batchsize = args.batchsize # params['ModelParams']['batchsize']

    # pdb.set_trace()
    keysIMG = list(numpyImages.keys())

    nr_iter_dataAug = nr_iter*batchsize
    np.random.seed(1)
    whichDataList = np.random.randint(len(keysIMG), size=int(nr_iter_dataAug/params['ModelParams']['nProc']))
    np.random.seed(11)
    whichDataForMatchingList = np.random.randint(len(keysIMG), size=int(nr_iter_dataAug/params['ModelParams']['nProc']))

    for whichData,whichDataForMatching in zip(whichDataList,whichDataForMatchingList):

        currImgKey = keysIMG[whichData]
        currGtKey = keysIMG[whichData] + '_segmentation' # require customization. This is for PROMISE12 data.
        # print("keysIMG type:{}\nkeysIMG:{}".format(type(keysIMG),str(keysIMG)))
        # print("whichData:{}".format(whichData))
        # pdb.set_trace()
        # currImgKey = keysIMG[whichData]
        # currGtKey = keysIMG[whichData] # for MSD data.

        # data agugumentation through hist matching across different examples...
        ImgKeyMatching = keysIMG[whichDataForMatching]

        defImg = numpyImages[currImgKey]
        defLab = numpyGT[currGtKey]

        defImg = utils.hist_match(defImg, numpyImages[ImgKeyMatching]) # why do histogram matching for all images? By Chao.

        if(np.random.rand(1)[0]>0.5): #do not apply deformations always, just sometimes
            defImg, defLab = utils.produceRandomlyDeformedImage(defImg, defLab, args.numcontrolpoints, params['ModelParams']['sigma'])

        dataQueue.put(tuple((defImg, defLab)))

def adjust_opt(optAlg, optimizer, iteration):
    if optAlg == 'sgd':
        if epoch < 150:
            lr = 1e-1
        elif epoch == 150:
            lr = 1e-2
        elif epoch == 225:
            lr = 1e-3
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train_dice(args, epoch, iteration, model, trainLoader, optimizer, trainF):
    model.train()
    nProcessed = 0
    batch_size = len(trainLoader.dataset)
    for batch_idx, output in enumerate(trainLoader):
        data, target = output # data shape [batch_size, channels, z, y, x], output shape [batch_size, z, y, x]
        # pdb.set_trace()
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data = Variable(data)
        target = Variable(target)
        # data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data) # output shape[batch_size, 2, z*y*x]
        # print("data shape:{}\noutput shape:{}\ntarget shape:{}".format(data.shape, output.shape, target.shape))
        loss = lossFuncs.dice_loss(output, target)
        # make_graph.make_dot(os.path.join(resultDir, 'promise_net_graph.dot'), loss)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        diceOvBatch = loss.data[0]/batch_size # loss.data[0] is sum of dice coefficient over a mini-batch. By Chao.
        err = 100.*(1. - diceOvBatch)

    if np.mod(iteration, 10) == 0:
        print('\nFor trainning: epoch: {} iteration: {} \tdice_coefficient over batch: {:.4f}\tError: {:.4f}\n'.format(epoch, iteration, diceOvBatch, err))

    return diceOvBatch, err


def test_dice(dataManager, args, epoch, model, testLoader, testF, resultDir):
    '''
    :param dataManager: contains self.sitkImages which is a dict of test sitk images or all sitk images including test sitk images.
    :param args:
    :param epoch:
    :param model:
    :param testLoader:
    :param testF: path to file recording test results.
    :return:
    '''
    model.eval()
    test_dice = 0
    incorrect = 0
    # assume single GPU/batch_size =1
    # pdb.set_trace()
    for batch_idx, data in enumerate(testLoader):
        data, target, id = data
        # print("testing with {}".format(id[0]))
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = Variable(data)
        target = Variable(target)
        output = model(data)
        dice = lossFuncs.dice_loss(output, target).data[0]
        test_dice += dice
        incorrect += (1. - dice)

        # pdb.set_trace()
        _, _, z, y, x = data.shape  # need to squeeze to shape of 3-d. by Chao.
        output = output[0,...] # assume batch_size = 1
        _, output = output.max(0)
        output = output.view(z, y, x)
        output = output.cpu()
        # In numpy, an array is indexed in the opposite order (z,y,x)  while sitk will generate the sitk image in (x,y,z). (refer: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/01_Image_Basics.html)
        output = output.numpy()
        output = np.transpose(output, [2,1,0]) # change to simpleITK order (x, y, z)
        # pdb.set_trace()
        print("save predicted label for test{}".format(id[0]))
        dataManager.writeResultsFromNumpyLabel(output, id[0], '_tested_epoch{}'.format(epoch), '.mhd', resultDir) # require customization
        testF.write('{},{},{},{}\n'.format(epoch, id[0], dice, 1-dice))

    nTotal = len(testLoader)
    test_dice /= nTotal  # loss function already averages over batch size
    err = 100.*incorrect/nTotal
    # if np.mod(iteration, 10) == 0:
    #     print('\nFor testing: iteration:{}\tAverage Dice Coeff: {:.4f}\tError:{:.4f}\n'.format(iteration, test_dice, err))

    # testF.write('{},{},{}\n'.format('avarage', test_dice, err))
    testF.flush()


def inference(dataManager, args, loader, model, resultDir):
    model.eval()
    # assume single GPU / batch size 1
    # pdb.set_trace()
    for batch_idx, data in enumerate(loader):
        data, id = data
        # pdb.set_trace()
        # convert names to batch tensor
        if args.cuda:
            data.pin_memory()
            data = data.cuda()
        with torch.no_grad():
            data = Variable(data)
        output = model(data)

        _, _, z, y, x = data.shape  # need to subset shape of 3-d. by Chao.
        output = output[0,...] # assume batch_size=1
        _, output = output.max(0)
        output = output.view(z, y, x) 
        output = output.cpu()
        # In numpy, an array is indexed in the opposite order (z,y,x)  while sitk will generate the sitk image in (x,y,z). (refer: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/01_Image_Basics.html)
        output = output.numpy()
        output = np.transpose(output, [2,1,0]) # change to simpleITK order (x, y, z)
        # pdb.set_trace()
        print("save predicted label for inference {}".format(id[0]))
        dataManager.writeResultsFromNumpyLabel(output, id[0], '_inferred', '.mhd', resultDir) # require customization


## main method
def main(params, args):
    best_prec1 = 100. # accuracy? by Chao
    epochs = args.nEpochs
    nr_iter = args.numIterations # params['ModelParams']['numIterations']
    batch_size = args.batchsize # params['ModelParams']['batchsize']
    resultDir = 'results/vnet.base.{}.{}'.format(params['ModelParams']['task'], datestr())

    weight_decay = args.weight_decay
    setproctitle.setproctitle(resultDir)
    if os.path.exists(resultDir):
        shutil.rmtree(resultDir)
    os.makedirs(resultDir, exist_ok=True)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print("build vnet")
    model = vnet.VNet(elu=False, nll=False)
    gpu_ids = args.gpu_ids
    # torch.cuda.set_device(gpu_ids) # why do I have to add this line? It seems the below line is useless to apply GPU devices. By Chao.
    # model = nn.parallel.DataParallel(model, device_ids=[gpu_ids])
    model = nn.parallel.DataParallel(model)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        model.apply(weights_init)


    train = train_dice
    test = test_dice

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    if args.cuda:
        model = model.cuda()

    # transform
    trainTransform = transforms.Compose([
        transforms.ToTensor()
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor()
    ])

    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.baseLR,
                              momentum=args.momentum, weight_decay=weight_decay) # params['ModelParams']['baseLR']
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), weight_decay=weight_decay)


    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # pdb.set_trace()
    DataManagerParams = {'dstRes':np.asarray(eval(args.dstRes), dtype=float), 'VolSize':np.asarray(eval(args.VolSize), dtype=int), 'normDir':params['DataManagerParams']['normDir']}

    if params['ModelParams']['dirTestImage']:  # if exists, means test files are given.
        print("\nloading training set")
        dataManagerTrain = DM.DataManager(params['ModelParams']['dirTrainImage'],
                                          params['ModelParams']['dirTrainLabel'],
                                          params['ModelParams']['dirResult'],
                                          DataManagerParams)
        dataManagerTrain.loadTrainingData()  # required
        train_images = dataManagerTrain.getNumpyImages()
        train_labels = dataManagerTrain.getNumpyGT()

        print("\nloading test set")
        dataManagerTest = DM.DataManager(params['ModelParams']['dirTestImage'], params['ModelParams']['dirTestLabel'],
                                         params['ModelParams']['dirResult'],
                                         DataManagerParams)
        dataManagerTest.loadTestingData()  # required
        test_images = dataManagerTest.getNumpyImages()
        test_labels = dataManagerTest.getNumpyGT()

        testSet = customDataset.customDataset(mode='test', images=test_images, GT=test_labels, transform=testTransform)
        testLoader = DataLoader(testSet, batch_size=1, shuffle=True, **kwargs)

    elif args.testProp:  # if 'dirTestImage' is not given but 'testProp' is given, means only one data set is given. need to perform train_test_split.
        print('\n loading dataset, will split into train and test')
        dataManager = DM.DataManager(params['ModelParams']['dirTrainImage'],
                                     params['ModelParams']['dirTrainLabel'],
                                     params['ModelParams']['dirResult'],
                                     DataManagerParams)
        dataManager.loadTrainingData()  # required
        numpyImages = dataManager.getNumpyImages()
        numpyGT = dataManager.getNumpyGT()
        # pdb.set_trace()
        
        train_images, train_labels, test_images, test_labels = train_test_split(numpyImages, numpyGT, args.testProp)
        testSet = customDataset.customDataset(mode='test', images=test_images, GT=test_labels, transform=testTransform)
        testLoader = DataLoader(testSet, batch_size=1, shuffle=True, **kwargs)

    else: # if both 'dirTestImage' and 'testProp' are not given, means the only one dataset provided is used as train set.
        print('\n loading only train dataset')
        dataManager = DM.DataManager(params['ModelParams']['dirTrainImage'],
                                     params['ModelParams']['dirTrainLabel'],
                                     params['ModelParams']['dirResult'],
                                     DataManagerParams)
        dataManager.loadTrainingData()  # required
        train_images = dataManager.getNumpyImages()
        train_labels = dataManager.getNumpyGT()

        test_images = None
        test_labels = None
        testSet = None
        testLoader = None

    if params['ModelParams']['dirTestImage']:
        dataManager_toTestFunc = dataManagerTest
    else:
        dataManager_toTestFunc = dataManager

    ### For train_images and train_labels, starting data augmentation and loading augmented data with multiprocessing
    dataQueue = Queue(30)  # max 30 images in queue?
    dataPreparation = [None] * params['ModelParams']['nProc']

    # processes creation
    for proc in range(0, params['ModelParams']['nProc']):
        dataPreparation[proc] = Process(target=dataAugmentation,
                                        args=(params, args, dataQueue, train_images, train_labels))
        dataPreparation[proc].daemon = True
        dataPreparation[proc].start()

    batchData = np.zeros((batch_size, DataManagerParams['VolSize'][0],
                                 DataManagerParams['VolSize'][1],
                                 DataManagerParams['VolSize'][2]), dtype=float)
    batchLabel = np.zeros((batch_size, DataManagerParams['VolSize'][0],
                                  DataManagerParams['VolSize'][1],
                                  DataManagerParams['VolSize'][2]), dtype=float)

    trainF = open(os.path.join(resultDir, 'train.csv'), 'w')
    testF = open(os.path.join(resultDir, 'test.csv'), 'w')

    for epoch in range(1, epochs+1):
        dataQueue_tmp = dataQueue # not working from epoch = 2 and so on. why??? By Chao.
        diceOvBatch = 0
        err = 0
        for iteration in range(1, nr_iter + 1):
            # adjust_opt(args.opt, optimizer, iteration+)
            if args.opt == 'sgd':
                if np.mod(iteration, args.stepsize) == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= args.gamma

            for i in range(batch_size):
                [defImg, defLab] = dataQueue_tmp.get()

                batchData[i, :, :, :] = defImg.astype(dtype=np.float32)
                batchLabel[i, :, :, :] = (defLab > 0.5).astype(dtype=np.float32)

            trainSet = customDataset.customDataset(mode='train', images=batchData, GT=batchLabel,
                                                   transform=trainTransform)
            trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, **kwargs)

            diceOvBatch_tmp, err_tmp = train(args, epoch, iteration, model, trainLoader, optimizer, trainF)

            if args.xLabel == 'Iteration':
                trainF.write('{},{},{}\n'.format(iteration, diceOvBatch_tmp, err_tmp))
                trainF.flush()
            elif args.xLabel == 'Epoch':
                diceOvBatch += diceOvBatch_tmp
                err += err_tmp
        if args.xLabel == 'Epoch':
            trainF.write('{},{},{}\n'.format(epoch, diceOvBatch/nr_iter, err/nr_iter))
            trainF.flush()

        if np.mod(epoch, epochs) == 0: # default to set last epoch to save checkpoint
            save_checkpoint({'epoch': epoch,
                             'state_dict': model.state_dict(),
                             'best_prec1': best_prec1}, path=resultDir, prefix="vnet_epoch{}".format(epoch))
        if epoch == epochs and testLoader:
            test(dataManager_toTestFunc, args, epoch, model, testLoader, testF, resultDir)  # by Chao.

    os.system('./plot.py {} {} &'.format(args.xLabel, resultDir))

    trainF.close()
    testF.close()

    # inference, i.e. output predicted mask for test data in .mhd
    if params['ModelParams']['dirInferImage'] != '':
        print("loading inference data")
        dataManagerInfer = DM.DataManager(params['ModelParams']['dirInferImage'], None,
                                          params['ModelParams']['dirResult'],
                                          DataManagerParams)
        dataManagerInfer.loadInferData()  # required.  Create .loadInferData??? by Chao.
        numpyImages = dataManagerInfer.getNumpyImages()

        inferSet = customDataset.customDataset(mode='infer', images=numpyImages, GT=None, transform=testTransform)
        inferLoader = DataLoader(inferSet, batch_size=1, shuffle=True, **kwargs)
        inference(dataManagerInfer, args, inferLoader, model, resultDir)

