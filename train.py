import dogdataset
import sys
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
    lr = 0.0005
    batch_size = 20
    test_batch_size = 20
    model_save_dir = sys.argv[1]
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    net = resnet.ResNet18()
    train_dataset = dogdataset.DogDataset('./data/train/', './data/labels.csv')


    train_loader = DataLoader(train_dataset, batch_size=batch_size, \
            shuffle=True, num_workers=10, pin_memory=True, drop_last=True)
    net.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True)
    
    print 'start to run'
    step = 0
    train_target_constant = Variable(torch.ones(batch_size).cuda())
    for epoch in range(100):
        net.train()
        for i, data in enumerate(train_loader):
            step += 1
            imgs = Variable(data[0].cuda())
            labels = Variable(data[1].cuda())
            predicted = net(imgs)

            optimizer.zero_grad()
            loss = criterion(predicted, labels)
            if step % 50 == 0:
                print 'at step %s loss: '%step, loss.data[0]
                # acc = (s_pos.round()==train_target_constant).type(torch.FloatTensor)
                # acc2 = (s_neg.round()==train_target_constant-train_target_constant).type(torch.FloatTensor)
                # acc = (acc+acc2)/2
                # print 'acc: ', acc.mean().data[0]
                
            loss.backward()
            optimizer.step()
            lr = lr*0.86**(1e-4)
            # if step % 1000 == 0:
                # acc_ = 0
                # for i_test, data_test in enumerate(test_loader):
                    # net.eval()
                    # x_b_test = data_test[:, 0]
                    # x_p_test = data_test[:, 1]
                    # x_n_test = data_test[:, 2]
                    # x_b_test = Variable(x_b_test.cuda())
                    # x_p_test = Variable(x_p_test.cuda())
                    # x_n_test = Variable(x_n_test.cuda())

                    # y_b = net(x_b_test)
                    # y_p = net(x_p_test)
                    # y_n = net(x_n_test)
                    # s_pos = cos_similarity(y_b, y_p)
                    # s_neg = cos_similarity(y_b, y_n)
                    # acc = (s_pos.round()==test_target_constant).type(torch.FloatTensor)
                    # acc2 = (s_neg.round()==test_target_constant-test_target_constant).type(torch.FloatTensor)
                    # acc = (acc+acc2)/2
                    # acc_ += acc.mean().data[0]
                    # del x_b_test, x_p_test, x_n_test, s_pos, s_neg, acc, acc2, y_b, y_p, y_n
                    
                # # print 'test acc is :', acc_ / len(test_loader)
                # print 'test acc is :', acc_ / len(test_loader)
                # net.train()

            if step % 10000 == 0:
                torch.save(net.state_dict(), os.path.join(model_save_dir, 'model-%s'%step))

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

if __name__ == '__main__':
    main()
