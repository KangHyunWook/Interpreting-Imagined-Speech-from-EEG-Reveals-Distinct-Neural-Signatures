import random
import os
import scipy.io as sio
import numpy as np
import pandas as pd
import mat73
import pickle

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(data, pkl_path):
    with open(pkl_path, 'wb') as f:
        pickle.dump(data,f)

class BCIComp:
    def __init__(self, config):
        root_dir=config.dataset_dir
        target_subject='0'+str(config.subject) if config.subject<10 else str(config.subject)

        t_len=159 #segment size
        if config.exper_setting=='indep':
            train = []
            for i in range(1,16):
                if i==config.subject:
                    continue
                current_subject='0'+str(i) if i<10 else str(i)
                train.extend(self.get_train(root_dir, current_subject, t_len))
            
            test = self.get_test(root_dir,target_subject, t_len)
            self.train=train
            self.test=test

        elif config.exper_setting=='dep':
            self.train=self.get_train(root_dir, subject, t_len)            
            random.shuffle(self.train)

            self.test = self.get_test(root_dir,subject, t_len)

    def get_test(self,root_dir, subject, t_len):
        test_folder_path = os.path.join(root_dir, 'Test set')                        
        test_data_path = os.path.join(test_folder_path, 'Data_Sample'+subject+'.mat')
        #get test data
        splits = test_data_path.split('.')

        index = splits[-2].index('ple')
        current_test_label_idx = int(splits[-2][index+3:])*2

        test_groundtruth_fpath=os.path.join(root_dir,'Test set', 'Track3_Answer Sheet_Test.xlsx')        
        df = pd.read_excel(test_groundtruth_fpath)
        
        test_labels = df.iloc[:, current_test_label_idx]
        test_labels= test_labels.to_numpy()[2:]
        
        data = mat73.loadmat(test_data_path)
        
        data = data['epo_test']['x']
        data = data.transpose(2,1,0) 
        
        test = []
        for i in range(data.shape[0]):
            current_trial = data[i]
            label = test_labels[i]-1
            for j in range(current_trial.shape[1]//t_len):
                seg = current_trial[:, j*t_len:(j+1)*t_len]
                test.append((seg, label))

        return test

    def get_train(self, root_dir, subject, t_len):
        train_data_path = os.path.join(root_dir,'Training set','Data_Sample'+subject+'.mat')
        #load data
        data = sio.loadmat(train_data_path)

        raw_EEG_samples = data['epo_train']['x'][0][0]
        raw_EEG_samples = raw_EEG_samples.transpose()
        labels = np.argmax(data['epo_train']['y'][0][0]==1,axis=0)

        train = []
        l_cnt=0

        for trial in raw_EEG_samples:
            iterations = trial.shape[1]//t_len
            for i in range(iterations):
                seg = trial[:, i*t_len:(i+1)*t_len]
                train.append((seg,labels[l_cnt]))

            l_cnt+=1

        return train

    def get_data(self, mode):
        if mode=='train':
            return self.train
        elif mode=='test':
            return self.test
        else:
            print('wrong mode')
            exit()

class FEIS:
    def __init__(self, config):
        PKL_ROOT='./pkls'
        self.config=config
        self.pkl_train_path=pkl_train_path=os.path.join(PKL_ROOT,'sub'+str(config.subject)+'_train.pkl')
        self.pkl_test_path=pkl_test_path= os.path.join(PKL_ROOT,'sub'+str(config.subject)+'_test.pkl') 
        # self.label_int_map={'f':0, 'fleece':1, 'goose':2, 'k':3, 'm':4, 'n':5, 'ng':6, 'p':7, 
        #                                     's':8, 'sh':9, 't':10, 'thought':11, 'trap':12, 'v':13, 'z':14, 'zh':15}

        dep_pkl_train_path = os.path.join(PKL_ROOT,'sub'+str(config.subject)+'_train_dep.pkl')
        dep_pkl_test_path = os.path.join(PKL_ROOT,'sub'+str(config.subject)+'_test_dep.pkl') 

        self.label_int_map={'f':0, 'k':1}

        # self.label_int_map={'f':1, 'fleece':0, 'goose':0, 'k':0, 'm':0, 'n':0, 'ng':0, 'p':0, 
        #                                      's':0, 'sh':0, 't':0, 'thought':0, 'trap':0, 'v':0, 'z':0, 'zh':0}
        try:
            if config.exper_setting=='indep':
                self.train = load_pkl(pkl_train_path)
                self.test = load_pkl(pkl_test_path)
            elif config.exper_setting=='dep':
                
                self.train = load_pkl(dep_pkl_train_path)
                self.test = load_pkl(dep_pkl_test_path)
        except:
            if config.exper_setting=='indep':
                self.train, self.test = self.split_train_test(config.subject)
                save_pkl(self.train, self.pkl_train_path)
                save_pkl(self.test, self.pkl_test_path)
            elif config.exper_setting=='dep':
                subs =sorted([sub for sub in os.listdir(self.config.dataset_dir) if 'chinese' not in sub])
                train=[]
                sub='0'+str(config.subject) if config.subject<10 else str(config.subject)
                file_path = os.path.join(self.config.dataset_dir, sub,'thinking','thinking.csv')           
                data, labels = self.load_data(file_path)
                for i in range(data.shape[0]):
                    train.append([data[i], labels[i]])
                random.shuffle(train)
                train_len = int(0.7*len(train))
                
                test=train[train_len:]
                train=train[:train_len]

                self.train=train
                self.test=test
            
                save_pkl(train, dep_pkl_train_path)
                save_pkl(test, dep_pkl_test_path)



    def split_train_test(self, target_sub):
        subs =sorted([sub for sub in  os.listdir(self.config.dataset_dir) if 'chinese' not in sub])
        train, test=[],[]
        for sub in subs:
            file_path = os.path.join(self.config.dataset_dir, sub,'thinking','thinking.csv')           
            data, labels = self.load_data(file_path)
            
            if int(sub)==self.config.subject:
                print('target sub')
                for i in range(data.shape[0]):
                    test.append([data[i], labels[i]])
            else:
                print('sub:', sub)
                for i in range(data.shape[0]):
                    train.append([data[i], labels[i]])

        
        return train, test

    def load_data(self, file_path):
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
            label_list.append(self.label_int_map[label])


        data_list=np.asarray(data_list)
        label_list=np.asarray(label_list)

        return data_list, label_list

    def get_data(self, mode):
        if mode =='train':
            return self.train
        elif mode=='test':
            return self.test
        else:
            print('wrong mode, choice: train, test')
            exit(1)










