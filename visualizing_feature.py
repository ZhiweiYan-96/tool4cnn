from graphviz import Digraph
import re
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Variable
import torchvision.models as models
import numpy  as np
import os
import math
import matplotlib.pyplot as plt

def plot_channel_statistics(feature,together='True'):
    b,c,h,w = feature.size()
    channel_max = feature.view(b,c,-1).max(2)[0].view(-1)
    channel_mean = feature.view(b,c,-1).mean(2).view(-1)
    if together:
        plt.plot(np.arange(0,b*c),channel_mean.data.cpu().numpy(),label='mean')
        plt.plot(np.arange(0,b*c),channel_max.data.cpu().numpy(),label='max')
        plt.legend()
    else:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.plot(np.arange(0,b*c),channel_mean.data.cpu().numpy(),label='mean',color='blue')
        ax1.legend()
        ax2 = fig.add_subplot(122)
        ax2.plot(np.arange(0,b*c),channel_max.data.cpu().numpy(),label='max',color='orange')
        ax2.legend()
    plt.show()

def plot_features(features,total_num,num,dir_name,fig_size):
    '''
    :param features: Variable B*C*H*W
    :param total_num: Channel Number
    :param num: plot per figure
    :param dir_name: save_dir
    :param fig_size:
    :return:
    '''
    if not os.path.exists(dir_name+'/'):
        os.makedirs(dir_name + '/')
    index = 0
    for iteration in range(0,math.ceil(total_num/num)):
        plt.figure( figsize = fig_size )
        start_index = index
        for i in range(0,num):
            if index >= total_num -1:
                continue
            ax = plt.subplot( int(math.sqrt(num)), int(math.sqrt(num)), i+1)
            ax.set_title('Sample #{}'.format(index))
            ax.axis('off')
            plt.imshow(features[0,index].cpu().data.numpy(),cmap='jet')
            index += 1
        end_index = index
        plt.savefig(dir_name+'/'+'_'+str(start_index)+'_'+str(end_index)+'.png')
        plt.close()

activation = {}
def get_activation(name):
    def hook(model,input,output):
        # print(output)
        activation[name] = output.detach()
    return hook