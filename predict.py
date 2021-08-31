#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author: charles
# @Date:   2021-02-12 12:02:11
# @Last modified by:   charles
# @Last modified time: 2021-02-12 14:02:24

# Prédit les images segmentées à partir du fichier weights.pt à placer dans le même répertoire
# Faire ressortir un fichier excel (dans le dossier model_id_predict) avec le iou score moyen et la deviation 
# pour chaque valeur de threshold spécifiée en début de boucle


import pickle
from glob import glob
import datetime
import time
import getpass
import os
from pathlib import Path
import matplotlib.pyplot as plt
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import csv
import numpy as np
import torch
from torch.functional import atleast_1d
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from torchvision import transforms
import pandas as pd
from detector.unet import UNet
from detector.utils import train_model, predict
from detector.utils import count_parameters
from detector.utils import SegmentationDataSet
from detector.utils import get_mean_std, get_class_weights
from detector.metrics import iou_not_empty, iou_not_empty_alt
from detector.utils import mIoULoss
from detector.utils import DiceLoss
from detector.utils import preds_to_png, RandomSampler
from detector.plot import plot_results, learning_curve, compare_learning_curve

t0 = time.time()

home = os.path.expanduser("~")

model_id = datetime.datetime.now().replace(microsecond=0)
model_id = model_id.isoformat().replace(':', '-')
model_id += '_predict'
print('\nUsername: {}'.format(getpass.getuser()))
print('Model ID: {}'.format(model_id))

batch_size = 32 #taille du vecteur contenant les images 
# num_workers = 8
# epochs = 50 #nombre d'itérations de batch
# class_weights = [1, 112]
# pos_weight = 112*torch.ones(256)
# learning_rate = 1e-3
# batch_norm = True
# droprate = 0

# transform_X = transforms.Normalize((33.6657,), (9.0867,))  # from get_mean_std(train_loader)
def transform_X(X):
    stats = X.mean(), X.std()
    return transforms.Normalize(*stats)(X)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# input_path = f'{home}/Notebooks/test_input/*.png'
# output_path = f'{home}/Notebooks/test_output/*.png'
#train_dataset = SegmentationDataSet(inputs=glob(input_path),
                                    #targets=glob(output_path),
                                    #transform_X=transform_X)


# test_input_path = r'C:\Users\Ange\Notebooks\test_input\*.png'
# test_output_path = r'C:\Users\Ange\Notebooks\test_output\*.png'

test_input_path, test_output_path = RandomSampler(
    r'C:\Users\Ange\Notebooks\test_input\*.png', 
    r'C:\Users\Ange\Notebooks\test_output\*.png', 
    0.5)

test_dataset = SegmentationDataSet(inputs=test_input_path,
                                   targets=test_output_path,
                                   transform_X=transform_X)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# train_dataset_size = len(train_dataset)
# indices = list(range(train_dataset_size))
# split = int(np.floor(0.2 * train_dataset_size))
# np.random.shuffle(indices)
# train_indices, val_indices = indices[split:], indices[:split]
# train_sampler = SubsetRandomSampler(train_indices)
# val_sampler = SubsetRandomSampler(val_indices)
# train_loader = DataLoader(train_dataset, batch_size=batch_size,
#                           sampler=train_sampler, num_workers=num_workers)
# val_loader = DataLoader(train_dataset, batch_size=batch_size,
#                         sampler=val_sampler, num_workers=num_workers)
# print("Number of training/validation patches:",
#       (len(train_indices),
#        len(val_indices)))

dataloaders = {
               'test': test_loader,
               }

# get_mean_std(train_loader)
# get_class_weights(train_loader)


# Define model
print('CUDA:', torch.cuda.is_available())

unet_params = {'n_classes': 1,
               'in_channels': 1,
               'depth': 5,
               'padding': True,
               'wf': 6,
               'up_mode': 'upconv',
               'batch_norm': False, #normalisation des poids
               'droprate': 0} 

model = UNet(**unet_params).to(device)

model = nn.DataParallel(model) 

model.load_state_dict(torch.load(r'weights.pt', map_location=torch.device('cpu')))

n_parameters = count_parameters(model)

print(f'Number of model parameters: {n_parameters}')

# Define optimizer and loss function
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# class_weights = torch.Tensor(class_weights).to(device)
#pos_weight = pos_weight.to(device)
# criterion = nn.CrossEntropyLoss(class_weights)
# criterion = nn.BCEWithLogitsLoss(pos_weight)
#criterion = nn.BCEWithLogitsLoss()
# criterion = mIoULoss()
# criterion = DiceLoss()

# model, history = train_model(model, dataloaders, epochs,
                            #  criterion, optimizer, return_history=True)

exports_dir = './{}/'.format(model_id)
os.makedirs(exports_dir, exist_ok=True)

# model_filename = exports_dir + 'weights.pt'
# torch.save(model.state_dict(), model_filename)
# print('Saved model weights as : {}'.format(model_filename))

# history_filename = exports_dir + 'history.pickle'
# with open(history_filename, 'wb') as handle:
#     pickle.dump(history, handle, protocol=-1)
# print('Saved training history as: {}'.format(history_filename))

# learning_curve(history=history,
#                name='loss',
#                outfile=exports_dir+'loss_learning_curves.png')

# compare_learning_curve(history=history,
#                        name='accuracy',
#                        outfile=exports_dir+'accuracy_learning_curves.png')

print('Predicting over the test dataset')
print(f'Dataset length : {len(test_input_path)}')


th_score_filename = exports_dir + 'th_score.csv'

with open(th_score_filename, mode='w', newline='') as file:

    for th in np.round(np.arange(0.21,0.26,0.01),3):
        file_writer = csv.writer(file)
        print(f'\nThreshold : {th}')
        resultat = predict(model, dataloaders['test'], th=th)
        
        # plot_results(inputs=resultat['inputs'].cpu().numpy(),
        #             masks=resultat['targets'].cpu().numpy(),
        #             preds=resultat['preds'].cpu().numpy(),
        #             probs=resultat['probs'].cpu().numpy(),
        #             n_plot=5,
        #             save_path=exports_dir+'example_tiles.png')

        #png_path = r'\Users\Ange\Desktop\preds_output'

        #preds_to_png(resultat['preds'], test_input_path, png_path)

        iou_scores = {'avg': resultat['iou_avg'],
                    'std': resultat['iou_std'],
                    }
        
        file_writer.writerow([th, resultat['iou_avg'], resultat['iou_std']])

        # test_scores_filename = exports_dir + 'test_scores_th' + str(th) + '.pickle'
        # with open(test_scores_filename, 'wb') as handle:
        #     pickle.dump(iou_scores, handle, protocol=-1)
        # print('Saved test iou scores as: {}'.format(test_scores_filename))
    

now = datetime.datetime.now()
print ("\nFinished at : " + now.strftime("%Y-%m-%d %H:%M:%S"))
print("Elapsed time : %s seconds" % (time.time() - t0))




