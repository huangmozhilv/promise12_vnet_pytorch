import pdb

import torch
from torch.autograd import Function
from itertools import repeat
import numpy as np

# Intersection = dot(A, B)
# Union = dot(A, A) + dot(B, B)
# The Dice loss function is defined as
# 1/2 * intersection / union
#
# The derivative is 2[(union * target - 2 * intersect * input) / union^2]

class DiceLoss(Function):
    '''
    Compute energy based on dice coefficient.
    Aims to maximize dice coefficient.
    '''
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def forward(self, input, target, save=True): # it seems official v-net sum up the dice coefficients over a minibatch. but not sure how it does with backward gradients? by Chao. In this case, mean is used both for forward and backward for each minibatch.
        # input shape: softmax output. shape is [batch_size, 2 (background and foreground), z*y*x]??? by Chao.
        # target shape: [batch_size, z, y, x]?? by Chao.
        # print("Loss forward:\ninput shape:{}; target shape:{}".format(input.shape, target.shape))

        eps = 0.00001

        # reshape target
        # pdb.set_trace()
        b, z, y, x = target.shape  # b:batch_size, z:depth, y:height, w:width
        target_ = target.view(b, -1)

        # result_ = torch.zeros(input.shape[0], input.shape[2])
        # target_ = torch.zeros(input.shape[0], input.shape[2])

        #     _, result_ = input.max(1)
        # for i in range(input.shape[0]):
        #     result_[i, :] = input[i, :, :].argmax(0) # by Chao
        result_ = input.argmax(1) # dim 2 is of length 2. Reduce the length to 1 and label it with the class with highest probability. by Chao.

        # result_ = torch.squeeze(result_) # will do harm when batch_size=1

        # if input.is_cuda:
        #     result = torch.cuda.FloatTensor(result_.size())
        #     target = torch.cuda.FloatTensor(target_.size())
        # else:
        #     result = torch.FloatTensor(result_.size())
        #     target = torch.FloatTensor(target_.size())
        # result.copy_(result_)
        # self.target_.copy_(target)
        # target = self.target_
        if input.is_cuda: # by Chao.
            result = result_.type(torch.cuda.FloatTensor)
            target = target_.type(torch.cuda.FloatTensor)
        else:
            result = result_.type(torch.FloatTensor)
            target = target_.type(torch.FloatTensor)

        if save:
            self.save_for_backward(result, target)

        self.intersect = torch.zeros(input.shape[0])
        self.union = torch.zeros(input.shape[0])
        dice = torch.zeros(input.shape[0])
        if input.is_cuda:
            self.intersect = self.intersect.cuda()
            self.union = self.union.cuda()
            dice = dice.cuda()
        for i in range(input.shape[0]):
            self.intersect[i] = torch.dot(result[i, :], target[i, :])
            # binary values so sum the same as sum of squares
            result_sum = torch.sum(result[i, :])
            target_sum = torch.sum(target[i, :])
            self.union[i] = result_sum + target_sum

            # the target volume can be empty - so we still want to
            # end up with a score of 1 if the result is 0/0
            dice[i] = 2*self.intersect[i] / (self.union[i] + eps)
            # print('union: {}\t intersect: {}\t dice_coefficient: {:.7f}'.format(str(self.union[i]), str(self.intersect[i]), dice[i])) # target_sum: {:.0f} pred_sum: {:.0f}; target_sum, result_sum,

            # intersect = torch.dot(result, target)
            # # binary values so sum the same as sum of squares
            # result_sum = torch.sum(result)
            # target_sum = torch.sum(target)
            # union = result_sum + target_sum
            #
            # # the target volume can be empty - so we still want to
            # # end up with a score of 1 if the result is 0/0
            # dice = 2*intersect / (union + eps)

        # batch mean dice
        sumDice = torch.sum(dice)

        out = torch.FloatTensor(1).fill_(sumDice)
        if input.is_cuda:
            out = out.cuda() # added by Chao.
        return out

    @staticmethod
    def backward(self, grad_output): # Update the weights of the network, typically using a simple update rule: weight = weight - learning_rate * gradient (refer: https://seba-1511.github.io/tutorials/beginner/blitz/neural_networks_tutorial.html )
        # print("grad_output:{}".format(grad_output))
        # why fix grad_output:tensor([1.])??? By Chao.
        input, target = self.saved_tensors
        intersect, union = self.intersect, self.union

        grad_input = torch.zeros(target.shape[0], 2, target.shape[1])
        if input.is_cuda:
            grad_input = grad_input.cuda()
        # pdb.set_trace()
        for i in range(input.shape[0]):
            part1 = torch.div(target[i,:], union[i])
            part2_2 = intersect[i] / (union[i] * union[i])
            part2 = torch.mul(input[i,:], part2_2)
            dDice = torch.add(torch.mul(part1, 2), torch.mul(part2, -4))
            if input.is_cuda:
                dDice = dDice.cuda()
            grad_input[i,0,:] = torch.mul(dDice, grad_output[0])
            grad_input[i,1,:] = torch.mul(dDice, -grad_output[0])

        return grad_input, None # Return None for the gradient of values that don’t actually need gradients

def dice_loss(input, target):
    #return DiceLoss()(input, target)
    return DiceLoss.apply(input, target)

def dice_error(input, target):
    eps = 0.00001
    _, result_ = input.max(1)
    result_ = torch.squeeze(result_)
    if input.is_cuda:
        result = torch.cuda.FloatTensor(result_.size())
        target_ = torch.cuda.FloatTensor(target.size())
    else:
        result = torch.FloatTensor(result_.size())
        target_ = torch.FloatTensor(target.size())
    result.copy_(result_.data)
    target_.copy_(target.data)
    target = target_
    intersect = torch.dot(result, target)

    result_sum = torch.sum(result)
    target_sum = torch.sum(target)
    union = result_sum + target_sum
    intersect = np.max([eps, intersect])
    # the target volume can be empty - so we still want to
    # end up with a score of 1 if the result is 0/0
    dice = 2*intersect / (union + eps)
#    print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
#        union, intersect, target_sum, result_sum, 2*IoU))
    return dice
