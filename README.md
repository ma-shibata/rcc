# rcc
# This repository contains main scripts used in the submitted manuscript entitled "Exploring the applicability of the regression on histological multi-class grading in clear cell renal cell carcinoma" by Shibata et al. 
1. regression_models: scripts for developing CNN models  
    1-1. `densenet.py`: fine-tunes the pre-trained DenseNet-121 with ccRCC image patches  
    1-2. `inception.py`: fine-tunes the pre-trained inception-v3 with ccRCC image patches  
    1-3. `pred2categorical.py`: performs k-means clustering to convert predictions of the regression models to four classes  

2. classification_models: scripts for developing CNN models  
    2-1. `densenet.py`: fine-tunes the pre-trained DenseNet-121 with ccRCC image patches  
    2-2. `inception.py`: fine-tunes the pre-trained inception-v3 with ccRCC image patches  

3. miximg: scripts for generating composite image patches and predicting their grades using the fine-tuned regression CNN models  
    3-1. `prepimg.py`: prepares composite image patches  
    3-2. `prediction.py`: predicts the grades of the composite patches using the fine-tuned regression CNN models  
