from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import random
import os
import numpy as np

label_int_map={'f':0, 'k':1}

def load_data(file_path):
    f = open(file_path)
    line = f.readline()

    # exit()
    data_list=[]
    label_list = []

    for line in f:
        splits = line.split(',')
        epoch = splits[1]
        label = splits[-3]
        
        if label not in ['f', 'k']:
            continue

        data = splits[2:-3]
        data = np.asarray(data).astype(float)
        # print('d:', data.shape)
        data_list.append(data)
        label_list.append(label_int_map[label])


    data_list=np.asarray(data_list)
    label_list=np.asarray(label_list)

    return data_list, label_list

model = SVC(C=1)

print('svm')

dataset_dir=r'/home/Hyunwook/codes/BrainCon-revision/FEIS-v1.1/scottwellington-FEIS-7e726fd/experiments'

subs=sorted([sub for sub in os.listdir(dataset_dir) if 'chinese' not in sub])

train=[]
random.seed(123)
accs=[]
for sub in subs:
    file_path = os.path.join(dataset_dir, sub,'thinking','thinking.csv')           
    data, labels = load_data(file_path)
    
    for i in range(data.shape[0]):
        train.append([data[i], labels[i]])
    random.shuffle(train)
    train_len = int(0.7*len(train))

    test=train[train_len:]
    train=train[:train_len]
    train_X, train_y = [], []
    test_X, test_y = [], []
    for feature, label in train:
        train_X.append(feature)
        train_y.append(label)
    for feature, label in test:
        test_X.append(feature)
        test_y.append(label)

    model.fit(train_X, train_y)

    preds = model.predict(test_X)
    acc=accuracy_score(test_y, preds)*100
    accs.append(acc)
    print('Sub%s acc: %.2f'%(sub, acc))

avg = np.mean(accs)
std = np.std(accs)
print(avg, std)
    



