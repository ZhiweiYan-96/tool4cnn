from data import *
from history.ssd_res50_pseudo import build_ssd
import torch
from torch.autograd import Variable
from layers.modules import MultiBoxLoss
from matplotlib import pyplot as plt
plt.switch_backend('TKAgg')
import os
import math


dataset_root =  VOC_ROOT
cfg = voc
testset = VOCDetection(root=dataset_root,
                               transform=BaseTransform(300, MEANS))

num_classes = 21
weights_file = 'weights/pseudo/ssd300_COCO_135000.pth'
net = build_ssd('test',300,num_classes=num_classes)

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


state_dict = torch.load(weights_file, map_location=lambda storage, loc: storage)
# create new OrderedDict that does not contain `module.`
#
# from collections import OrderedDict
#
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     head = k[:7]
#     if head == 'module.':
#         name = k[7:]  # remove `module.`
#     else:
#         name = k
#     new_state_dict[name] = v
# # net.load_state_dict(new_state_dict)
# net.resnet.load_state_dict(state_dict)
# net.conv_init()
net.load_state_dict(state_dict)
net.eval()
net.cuda()
# print(net.resnet.state_dict().keys())
criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False,True)



top_k =300
# detector = Detect(num_classes,0,cfg)
rgb_means = (104,117,123)
transform = BaseTransform(net.size, rgb_means)

img_id = 150
img, gt, h, w= testset.pull_item(img_id)
scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
x = Variable(img.unsqueeze(0), volatile=True)
x = x.cuda()
scale = scale.cuda()



activation = {}
def get_activation(name):
    def hook(model,input,output):
        # print(output)
        activation[name] = output.detach()
    return hook

net.resnet.layer1.register_forward_hook(get_activation('layer1'))
net.resnet.layer2.register_forward_hook(get_activation('layer2'))
net.resnet.layer3.register_forward_hook(get_activation('layer3'))
net.resnet.layer4.register_forward_hook(get_activation('layer4'))
net.resnet.layer1[1].conv2.register_forward_hook(get_activation('dilated'))
net.resnet.layer1[1].conv2.register_forward_hook(get_activation('no_dilated'))
net(x)
# print(activation['test'])

fig_size =(10,10)
# plot_features(activation['layer1'],256,16,'visualize/layer1',fig_size)
# plot_features(activation['layer2'],512,16,'visualize/layer2',fig_size)
# plot_features(activation['layer3'],1024,16,'visualize/layer3',fig_size)
# plot_features(activation['layer4'],2048,16,'visualize/layer4',fig_size)
plot_features(activation['dilated'],64,16,'visualize/dilated',fig_size)
plot_features(activation['no_dilated'],64,16,'visualize/no_dilated',fig_size)
# print(x.size())
# print(x)
# fig_size = (10,10)
# detail_branch = net.add_conv(x)
# plot_features(detail_branch,64,4,'visualize/detail_branch/%d'%(img_id),fig_size)
#
# temp = net.resnet.conv1(x)
# temp = net.resnet.bn1(temp)
# res_conv1 = net.resnet.relu(temp)
# plot_features(res_conv1,64,4,'visualize/re_conv1/%d'%(img_id),fig_size)
#
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(10,10))
# plt.imshow(img)
# plt.savefig('%d.jpg'%(img_id))



# import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
# max = 7
# plt.switch_backend('TKAgg')
# i = 1
# for module in net.conv1._modules.values():
#     if isinstance(module,nn.Conv2d):
#         weight = module.weight.data
#         print(module.weight)
#         c_out,c_in,h,w  = module.weight.size()
#         for j in range(0,int(c_out/16)):
#             if c_in == 3:
#                 k_max=3
#             else:
#                 k_max = int(c_in/16)
#             for k in range(0,k_max):
#                 kernel = weight[j][k]
#                 plt.subplot(max,max,i)
#                 i +=1
#                 plt.matshow(kernel,fignum=False)
#                 plt.title(module.)
#
# plt.show()
