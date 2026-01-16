from datetime import datetime

import argparse

import torch.optim as optim
import torch.nn as nn

optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {'elu': nn.ELU, 'hardshrink': nn.Hardshrink, 'hardtanh': nn.Hardtanh, 'leakyrelu': nn.LeakyReLU, 'prelu': nn.PReLU, 'relu': nn.ReLU, 'rrelu': nn.RReLU, 'tanh': nn.Tanh}

def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Config(object):
    def __init__(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                if key =='optimizer':
                    value = optimizer_dict[value]        
                if key=='activation':
                    value = activation_dict[value]
                setattr(self, key, value)
                

def get_config(**optional_kwargs):
    parser = argparse.ArgumentParser()
    time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
   
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--model_weight_path', type=str, default=None)
    parser.add_argument('--reduction_ratio', type=int, default=2)
    parser.add_argument('--n_subjects', type=int, default=16, required=True)
    parser.add_argument('--exper-setting', type=str, default='dep')
    parser.add_argument('--model_name', type=str, default='EEGNet')
    parser.add_argument('--dataset_dir', default='/mnt/data/members/fusion/SEED/ExtractedFeatures')
    parser.add_argument('--subject', type=str, default='1')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lstm', action='store_true')
    parser.add_argument('--w_mode', type=str, default='w')
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--modulator', action='store_true')
    parser.add_argument('--save_file_name', type=str, default='results.csv')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--b1', default=0.5)
    parser.add_argument('--b2', default=0.999)
    parser.add_argument('--data_choice', type=str, default='seed')
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--num_epochs',type=int, default=500)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--hidden_size', type=int, default=68)
    parser.add_argument('--use_cmd_sim', type=str2bool, default=True)
    parser.add_argument('--reverse_grad_weight', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--incon_weight', type=float, default=0.2)
    parser.add_argument('--invar_weight', type=float, default=0.1)
    parser.add_argument('--recon_weight', type=float, default=0.3)

    parser.add_argument('--name', type=str, default=f"{time_now}")
    parser.add_argument('--label_type', type=str, default='discrete')
    parser.add_argument('--num_electrodes', type=int, default=62)
    parser.add_argument('--in_channels', type=int, default=5)
    parser.add_argument('--lstm_hidden_size', type=int, default=14)

    
    kwargs = parser.parse_args()
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)
    
    return Config(**kwargs)