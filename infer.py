import pandas as pd
import  torchvision.models as models
import dogdataset
import numpy as np
import sys
import os
import vgg
import cv2
import time
import torch
from torch.autograd import Variable
from dogdataset import center_crop


def main():
    model_para = sys.argv[1]
    files = os.listdir('./data/test/')
    train_dataset = dogdataset.TestDataset('./data/train/', './data/labels.csv')
    pre_dict = {}
    # net = vgg.MyVGG().cuda()
    net = models.resnet50(num_classes=120)
    net = torch.nn.DataParallel(net, device_ids=[0,1]).cuda()
    net.load_state_dict(torch.load('./checkpoints/res50/model-%s0000'%model_para))
    net.eval()
    df = pd.DataFrame(columns=[id_.split('.')[0] for id_ in (['id'] + sorted(train_dataset.class_mapping.keys()))])
    idx = 0
    for file_name in files:
        idx += 1
        print '%s image, name: %s'%(idx, file_name)
        img = cv2.imread('./data/test/'+file_name, -1)
        img = center_crop(img).astype(np.float32).transpose((2,0,1)).reshape(1, 3, 224, 224)
        img = Variable(torch.from_numpy(img).cuda())
        probability = torch.nn.functional.softmax(net(img), dim=1)
        # probability = net(img)
        probability = list(probability.data.cpu().numpy().reshape([120]))
        id_name = file_name.split('.')[0]
        probability.insert(0, id_name)

        temp_df = pd.DataFrame([probability], columns=[id_.split('.')[0] for id_ in (['id'] + sorted(train_dataset.class_mapping.keys()))])
        df = df.append(temp_df)
    
    df.to_csv('./submit/submit%s.csv'%model_para, index=False, index_label=False)





if __name__ == '__main__':
    main()
