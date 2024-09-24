#!/usr/bin/env python

# This script fine-tunes the pre-trained DenseNet with ccRCC patches.
# Regression approach.

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
        img = PIL.Image.open('../data/img/{}.jpeg'.format(self.x[i]))
        if self.transform is not None:
            img = self.transform(img)
        return self.x[i] ,img, self.y[i]

def prepdata(nfold):
    bs = 64
    transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()])
    # splitdata: note that labels are converted to [1,2,3,4]
    df = pd.read_csv(f'../data/splitdata/fold{n}.csv')
    df['wsilabel'] += 1
    train_samples = list(df[df['class']=='train']['imgname'])
    train_labels  = list(df[df['class']=='train']['wsilabel'])
    val_samples   = list(df[df['class']=='test']['imgname'])
    val_labels    = list(df[df['class']=='test']['wsilabel'])
    train_set = MyDataset(train_samples,train_labels,transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=bs,shuffle=True)
    val_set = MyDataset(val_samples,val_labels,transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=bs,shuffle=False)
    return train_loader,val_loader

def prepmodel(device):
    densenet = models.densenet121(pretrained=True)
    num_ftrs = densenet.classifier.in_features
    densenet.classifier = nn.Linear(num_ftrs,1)
    for param in densenet.parameters():
        param.requires_grad = True
    criterion = nn.MSELoss()
    optimizer = optim.Adam(densenet.parameters())
    densenet = densenet.to(device)
    return densenet,criterion,optimizer

def sharedprocess(dataloader,densenet,criterion,phase):
    eploss = 0.0; epacc = 0.0; correct = 0; preddf = []
    for data in dataloader:
        names,inputs,labels = data
        inputs = inputs.to(device)
        labels = labels.to(device).view(len(inputs),1) #data
        optimizer.zero_grad()
        outputs = densenet(inputs)
        loss = criterion(outputs, labels)
        eploss += loss.item()*inputs.size(0)
        if phase == 'train':
            loss.backward()
            optimizer.step()
        else:
            for i in range(len(outputs)):
                name  = names[i].split('/')[-1].replace('.jpeg','')
                pred  = float(outputs[i,:].to('cpu').detach().numpy().copy())
                label = int(float(labels[i].to('cpu').detach().numpy().copy()))
                preddf.append([name,label,pred])
    eploss = eploss/len(dataloader.dataset)
    if phase == 'train':
        return [eploss]
    else:
        preddf = pd.DataFrame(preddf,columns=['imgname','wsilabel','pred'])
        return [eploss],preddf

def main(n,densenet,device,criterion,optimizer,train_loader,val_loader):
#    print('----- original paramters -----')
#    print(list(densenet.parameters()))
    num_epochs = 50
    # initial performance
    densenet.eval()
    with torch.no_grad():
        lossacc2,preddf = sharedprocess(val_loader,densenet,criterion,'validation')
        lossaccs= [[0]+lossacc2]
    for epoch in range(1,num_epochs+1):
        # training
        densenet.train()
        with torch.set_grad_enabled(True):
            lossacc1 = sharedprocess(train_loader,densenet,criterion,'train')
        # validation
        densenet.eval()
        with torch.no_grad():
            lossacc2,preddf = sharedprocess(val_loader,densenet,criterion,'validation')
        lossaccs.append(lossacc1+lossacc2)
        preddf.to_csv(f'densenet/fold{n}_ep{epoch}.csv',index=False,float_format='%.4f')
        model_scripted = torch.jit.script(densenet)
        model_scripted.save(f'densenet/fold{n}_ep{epoch}.pth')
    # export lossacc
    lossacclabels = ['train_loss','val_loss']
    lossaccdf = pd.DataFrame(lossaccs,columns=lossacclabels,index=[i for i in range(num_epochs+1)])
    lossaccdf.to_csv(f'densenet/fold{n}.csv',float_format='%.4f')
#    print('----- updated paramters -----')
#    print(list(densenet.parameters()))
    return

# main
nfold = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for n in range(nfold):
    train_loader,val_loader = prepdata(n)
    densenet,criterion,optimizer = prepmodel(device)
    main(n,densenet,device,criterion,optimizer,train_loader,val_loader) 

