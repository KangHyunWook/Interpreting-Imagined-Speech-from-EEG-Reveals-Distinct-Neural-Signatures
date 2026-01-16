from data_loader import get_loader
from config import get_config
from solver import Solver
from sklearn.metrics import accuracy_score

import torch
import os
import scipy.io as sio
import numpy as np
from random import random


random_name = str(random())
random_seed = 336
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)


acc_list=[]
train_config = get_config(mode='train')
n_total_sub=train_config.n_subjects

for i in range(n_total_sub):
    train_config = get_config(mode='train')
    test_config = get_config(mode='test')
    train_config.subject=test_config.subject=i+1

    train_data_loader = get_loader(train_config, shuffle=True)
    test_data_loader = get_loader(test_config, shuffle=False)


    device= "cuda:0" if torch.cuda.is_available() else "cpu"

    #todo: train model
    solver = Solver
    solver = solver(train_config, test_config, train_data_loader, test_data_loader, device, is_train=True, model=None)

    solver.build()
    solver.train()

    test_loss, test_true, test_pred = solver.eval(mode="test", to_print=True)

    acc = accuracy_score(test_pred, test_true) *100
    acc_list.append(acc)

    print('acc:',  round(acc,2))

with open(train_config.save_file_name, 'w') as f:
    for i in range(len(acc_list)):
        f.write('sub %d, %.2f\n'%(i+1,acc_list[i]))


std=np.std(acc_list)
print('avg acc: %.2f std: %.2f'%(np.mean(acc_list), np.std(acc_list)))
print('variance:', np.var(acc_list))
print('variance:', std**2)