#!/usr/bin/env python

from PIL import Image
import random
import pandas as pd

# This script generates composite image patches for ITH predictions.

def miximg(fold):
    l = []
    df = pd.read_csv(f'pairs/fold{fold}.csv')
    for i in range(df.shape[0]):
        im1 = Image.open(f'../data/img/{df.iloc[i,0]}.jpeg')
        im2 = Image.open(f'../data/img/{df.iloc[i,1]}.jpeg')
        grades = [int(s) for s in df.iloc[i,-1].split('-')]
        for proportion in [25,50,75]:
            xlimpx = {25:109,50:218,75:327}[proportion]
            im1_crop = im1.crop((0,0,xlimpx-1,439)) 
            im2_crop = im2.crop((xlimpx-1,0,435,439))
            newim = Image.new('RGB',(im1_crop.width+im2_crop.width,im1_crop.height))
            newim.paste(im1_crop,(0,0))
            newim.paste(im2_crop, (im1_crop.width,0))
            newim.save(f'img/{proportion}={df.iloc[i,0]}={df.iloc[i,1]}.jpeg')
            newim.close()
            newgrade = (grades[0]*proportion + grades[1]*(100-proportion))*0.01 
            l.append([f'{proportion}={df.iloc[i,0]}={df.iloc[i,1]}',newgrade])
        im1.close(); im2.close()
    outdf = pd.DataFrame(l,columns=['imgname','wsigrade'])
    outdf.to_csv(f'imglist/fold{fold}.csv',index=False)
    return

def matchimgs(fold):
    outlist = []; labels = []
    df = pd.read_csv(f'../data/splitdata/fold{fold}.csv')
    testimgs = df[df['class']=='test']
    gradeimgs = [sorted(list(testimgs[testimgs['wsilabel']==i]['imgname'])) for i in range(4)]
    for i in range(4):
        for j in range(i,i+2):
            if i == 3 and j == 4:
                break
            else:
                if i == j:
                    imgpairs = [random.sample(gradeimgs[i],2) for p in range(250)]
                else:
                    imgpairs = [random.sample(gradeimgs[i],1)+random.sample(gradeimgs[j],1) for p in range(250)]
                outlist += imgpairs
                labels += [f'{i}-{j}']*250
    outdf = pd.DataFrame(outlist,columns=['img1','img2'])
    outdf['wsilabel'] = labels
    outdf.to_csv(f'pairs/fold{fold}.csv',index=False)
    return outdf

for fold in range(5):
    imgpairs = matchimgs(fold)
    miximg(fold) 

