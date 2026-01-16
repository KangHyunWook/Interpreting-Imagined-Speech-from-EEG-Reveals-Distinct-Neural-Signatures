from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix

import torch
import models
import numpy as np
import os

import torch.nn as nn

#todo: define following method.
def write_losses(epoch_incon_losses,epoch_invar_losses, epoch_recon_losses, file_name):
    f = open(file_name, 'w')

    f.write('incon_loss,')
    for i in range(len(epoch_incon_losses)):
        f.write(str(epoch_incon_losses[i])+',')
    
    f.write('\ninvar_loss,')
    for i in range(len(epoch_invar_losses)):
        f.write(str(epoch_invar_losses[i])+',')

    f.write('\nrecon_loss,')            
    for i in range(len(epoch_recon_losses)):
        f.write(str(epoch_recon_losses[i])+',')
    
    f.close()


class Solver(object):
    def __init__(self, train_config, test_config, train_data_loader, test_data_loader, device, is_train=True, model=None):
        self.train_config = train_config
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.model=model
        self.is_train=is_train
        self.device = device
        
    def build(self, cuda=True):
        
        if self.model is None:
            self.model = getattr(models, self.train_config.model_name)(self.train_config)

        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        if torch.cuda.is_available() and cuda:
            self.model.to(self.device)
            
        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate)
                
    def train(self):
        curr_patience = patience = self.train_config.patience
        
        num_trials = 1
        
        criterion = self.criterion

        best_valid_loss = float('inf')
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        
        valid_losses = []

        for e in range(self.train_config.num_epochs):
            self.model.train()
            
            train_loss = []
            
            for batch in self.train_data_loader:
                self.model.zero_grad()
                features, labels = batch
                
                if self.train_config.model_name=='Conformer' and 'bcicomp' in self.train_config.dataset_dir.lower():
                    features = torch.mean(features, dim=2)    
                    features = torch.unsqueeze(features, axis=1)
                labels = labels.type(torch.LongTensor)
                
                batch_size = features.size(0)
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                y_tilde = self.model(features)

                cls_loss = criterion(y_tilde, labels)
                
                loss=cls_loss
                
                loss.backward()

                torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], self.train_config.clip)

                self.optimizer.step()
                
                
                train_loss.append(loss.item())
            

            print(f"Training loss: {round(np.mean(train_loss), 4)}")

            
            valid_loss, valid_true, valid_pred = self.eval(mode="dev", to_print=True)
            
            print(f"Current patience: {curr_patience}, current trial: {num_trials}.")
            if valid_loss <= best_valid_loss:
                best_valid_loss = valid_loss
                print("Found new best model on dev set!")
                if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
         
                torch.save(self.model.state_dict(), f'./checkpoints/model_{self.train_config.subject}.std')
                torch.save(self.optimizer.state_dict(), f'./checkpoints/optim_{self.train_config.subject}.std')
                curr_patience = patience
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    print("Running out of patience, loading previous best model.")
                    num_trials -= 1
                    curr_patience = patience
                    self.model.load_state_dict(torch.load(f'./checkpoints/model_{self.train_config.subject}.std'))
                    self.optimizer.load_state_dict(torch.load(f'./checkpoints/optim_{self.train_config.subject}.std'))
                    lr_scheduler.step()
                    print(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")

            if num_trials <= 0:
                print("Running out of patience, early stopping.")
                break

       

    def eval(self,mode=None, to_print=False, model_weight_path=None):
        assert(mode is not None)
        self.model.eval()

        y_true, y_pred = [], []
        eval_loss, eval_loss_diff = [], []

        if mode == "dev":
            dataloader = self.test_data_loader
        elif mode == "test":
            dataloader = self.test_data_loader

            if to_print:
                if model_weight_path==None:
                    self.model.load_state_dict(torch.load(
                        f'checkpoints/model_{self.train_config.subject}.std'))
                else:
                    self.model.load_state_dict(torch.load(
                            model_weight_path))


        valid_incon_losses=[]
        valid_invar_losses=[]
        valid_recon_losses=[]
        
        with torch.no_grad():
            for batch in dataloader:
                self.model.zero_grad()
                features, labels = batch
                if self.train_config.model_name=='Conformer' and 'bcicomp' in self.train_config.dataset_dir.lower():
                    features = torch.mean(features, dim=2)
                    features = torch.unsqueeze(features, axis=1)

                labels = labels.type(torch.LongTensor)

                features = features.to(self.device)
                labels = labels.to(self.device)

                y_tilde = self.model(features)

                
                cls_loss = self.criterion(y_tilde, labels)
                loss = cls_loss
                
                eval_loss.append(loss.item())

                y_pred.append(y_tilde.detach().cpu().numpy())
                y_true.append(labels.detach().cpu().numpy())

        eval_loss = np.mean(eval_loss)

        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()
        #todo: if exper_setting=='indep' return y_true and y_pred
        
        y_pred = np.argmax(y_pred, 1)


        # accuracy = self.calc_metrics(y_true, y_pred, mode, to_print)

        return eval_loss, y_true, y_pred#, round(np.mean(valid_diff_losses),4), round(np.mean(valid_sim_losses),4), round(np.mean(valid_recon_losses),4)
        
        

    
    
        