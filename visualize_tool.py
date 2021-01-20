import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.lines import *
import matplotlib.pyplot as plt



def plot_grad_flow_1(named_parameters,file_name):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        # print(n)
        if (p.requires_grad) and ("bias" not in n) and ('downsample' not in n):
            if p.grad is not None:
                name_list = n.split('.')
                n = '.'.join(name_list[:-1])
                layers.append(n)
                ave_grads.extend(p.grad.abs().mean().data.cpu().numpy())
                max_grads.extend(p.grad.abs().max().data.cpu().numpy())
    # print(np.array(ave_grads))
    # plt.figure(figsize=(100,100))
    plt.bar(np.arange(len(max_grads)), np.array(max_grads), alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), np.array(ave_grads), alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    ax = plt.gca()
    if len(layers)>20:
        labels = ax.get_xticklabels()
        for i in range(len(layers)):
            if i%3 != 0:
                labels[i].set_visible(False)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=100)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name,dpi=300)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def plot_grad_flow(named_parameters,file_name):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        # print(n)
        if (p.requires_grad) and ("bias" not in n) and ('downsample' not in n):
            if p.grad is not None:
                name_list = n.split('.')
                n = '.'.join(name_list[:-1])
                layers.append(n)
                ave_grads.extend(p.grad.abs().mean().data.cpu().numpy())
                max_grads.extend(p.grad.abs().max().data.cpu().numpy())
    # print(np.array(ave_grads))
    # plt.figure(figsize=(100,100))
    plt.figure(figsize=(10,10))
    for i in range(4):
        plt.subplot(2,2,i+1)
        if i==0:
            plt.bar(np.arange(len(max_grads)), np.array(max_grads), alpha=0.1, lw=1, color="c")
            plt.title('Max Gradient')
            plt.legend([Line2D([0], [0], color="c", lw=4)], ['max-gradient'])
            plt.ylabel("Max gradient")
        elif i==1:
            plt.bar(np.arange(len(max_grads)), np.array(ave_grads), alpha=0.1, lw=1, color="b")
            plt.title('Avg Gradient')
            plt.legend([Line2D([0], [0], color="b", lw=4)], ['mean-gradient'])
            plt.ylabel("average gradient")
        elif i==2:
            plt.bar(np.arange(len(max_grads)), np.array(max_grads), alpha=0.1, lw=1, color="c")
            plt.bar(np.arange(len(max_grads)), np.array(ave_grads), alpha=0.1, lw=1, color="b")
            plt.title('Mix')
            plt.legend([Line2D([0], [0], color="c", lw=4),
                        Line2D([0], [0], color="b", lw=4),
                        Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        elif i==3:
            plt.bar(np.arange(len(max_grads)), np.array(max_grads), alpha=0.1, lw=1, color="c")
            plt.bar(np.arange(len(max_grads)), np.array(ave_grads), alpha=0.1, lw=1, color="b")
            plt.title('Mix')
            plt.legend([Line2D([0], [0], color="c", lw=4),
                        Line2D([0], [0], color="b", lw=4),
                        Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
            plt.ylim(bottom=-0.001, top=0.02)
            plt.ylabel("average gradient")
        ax = plt.gca()
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        if len(layers) > 20:
            labels = ax.get_xticklabels()
            for j in range(len(layers)):
                if j % 3 != 0:
                    labels[j].set_visible(False)
        plt.xlim(left=0, right=len(ave_grads))
        plt.grid(True)
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        ax = plt.gca()
        if len(layers) > 20:
            labels = ax.get_xticklabels()
            for i in range(len(layers)):
                if i % 3 != 0:
                    labels[i].set_visible(False)
        plt.xlim(left=0, right=len(ave_grads))
        # plt.ylim(bottom=-0.001, top=100)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        # plt.title("Gradient flow")
    plt.tight_layout()
    plt.savefig(file_name,dpi=300)
    # plt.show(block=False)
    # plt.pause(1)
    plt.close()
