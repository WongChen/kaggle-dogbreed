import dogdataset
import vgg
import sys
# import models
import  torchvision.models as models
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable
from time import time
import numpy as np
import torch
import resnet
import torch.nn.functional as F



def main():
    lr = 0.0001
    batch_size = 30
    test_batch_size = 20
    model_save_dir = sys.argv[1]
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    net = models.resnet50(num_classes=120, pretrained=True)
    net = torch.nn.DataParallel(net, device_ids=[0, 1])
    # net = vgg.MyVGG()
    net = net.cuda()
    train_dataset = dogdataset.TrainDataSet('./data_gen/train/')
    test_dataset = dogdataset.TestDataset('./data/train/', './data/test.csv')


    train_loader = DataLoader(train_dataset, batch_size=batch_size, \
            shuffle=True, num_workers=10, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=10, \
            shuffle=False, num_workers=10, pin_memory=True, drop_last=True)
    ce_weight = test_dataset.ce_weight
    criterion = nn.CrossEntropyLoss(torch.from_numpy(np.array(ce_weight))).cuda()
    # criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    
    print 'start to run'
    step = 0
    net.load_state_dict(torch.load('./checkpoints/dense_weightedloss/model-4000'))
    for epoch in range(300):
        net.train()
        for i, data in enumerate(train_loader):
            step += 1
            imgs = Variable(data[0].cuda())
            labels = Variable(data[1].cuda())
            predicted = net(imgs)

            optimizer.zero_grad()
            loss = criterion(predicted, labels)
            if step % 50 == 0:
                print 'at step %s loss: '%step, loss.data[0], 'lr: %s'%lr, 'acc: ', predicted.data.max(1)[1].eq(labels.data).cpu().sum()/float(batch_size)
                # acc = (s_pos.round()==train_target_constant).type(torch.FloatTensor)
                # acc2 = (s_neg.round()==train_target_constant-train_target_constant).type(torch.FloatTensor)
                # acc = (acc+acc2)/2
                # print 'acc: ', acc.mean().data[0]
                
            loss.backward()
            optimizer.step()
            lr = lr*0.9**(1e-4)
            if step % 600 == 0:
                net.eval()
                correct = 0
                loss_ = 0
                for i_test, data_test in enumerate(test_loader):
                    imgs_test = Variable(data_test[0].cuda())
                    labels_test = Variable(data_test[1].cuda())
                    results = net(imgs_test)
                    loss = criterion(results, labels_test)
                    loss_ += loss.data[0]
                    predict = results.data.max(1)[1]
                    correct += predict.eq(labels_test.data).cpu().sum()
                print 'test acc is :', float(correct) / 990, 'loss is : ', float(loss_) / i_test
                net.train()

            if step % 2000 == 0:
                torch.save(net.state_dict(), os.path.join(model_save_dir, 'model-%s'%step))

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

if __name__ == '__main__':
    main()
