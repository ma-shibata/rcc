#!/usr/bin/env python

import PIL
import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data

# This script make DenseNet and inception trained with ccRCC data predict grades of composite patches.

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, label_path, labels,transform=None):
        x = label_path
        y = labels
        self.x = x
        self.y = torch.from_numpy(np.array(y,dtype='f4')).clone()
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = PIL.Image.open('img/{}.jpeg'.format(self.x[i]))
        if self.transform is not None:
            img = self.transform(img)
        return self.x[i] ,img, self.y[i]

def prepdata(modelname,fold):
    imgsize = 224 if modelname == 'densenet' else 299
    bs = 64
    transform = transforms.Compose([
    transforms.Resize(imgsize),
    transforms.ToTensor()])
    df = pd.read_csv(f'imglist/fold{fold}.csv')
    df['wsigrade'] += 1
    val_samples = list(df['imgname'])
    val_labels  = list(df['wsigrade'])
    val_set = MyDataset(val_samples,val_labels,transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=bs,shuffle=False)
    return val_loader,val_samples

def main(alg,modelname,val_loader,df,fold):
    ep = {'regression':{'d':48,'i':47},'classification':{'d':48,'i':44}}[alg][modelname[0]]
    model = torch.jit.load(f'../g14_{alg}/{modelname}/fold{fold}_ep{ep}.pth',map_location="cuda")
    with torch.no_grad():
        for data in val_loader:
            names,inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.eval()
            output = model(inputs)
            if modelname == 'inception':
                output = output.logits
            if alg == 'regression':
                for i in range(len(names)):
                    df.loc[names[i],modelname] = float(output[i,:].to('cpu').detach().numpy().copy())
            else:
                output = torch.nn.functional.softmax(output,dim=1)
                for i in range(len(output)):
                    prob = output[i,:].to('cpu').detach().numpy().copy()
                    expectedval = np.sum(np.array([1,2,3,4])*prob)
                    df.loc[names[i],modelname] = float(expectedval) 
    return df

allimgs = []; allgrade = []
for i in range(5):
    allimgs += list(pd.read_csv(f'imglist/fold{i}.csv')['imgname'])
    allgrade += list(pd.read_csv(f'imglist/fold{i}.csv')['wsigrade'])
df = pd.DataFrame(columns=['densenet','inception'],index=allimgs)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for alg in ['regression','classification']:
    for modelname in ['densenet','inception']:
        for fold in range(5):
            val_loader,imgnames = prepdata(modelname,fold)
            df = main(alg,modelname,val_loader,df,fold)
    df['label'] = allgrade
    df['label'] += 1
    df.to_csv(f'pred.{alg}.csv',float_format='%1.4f')

