CONTENTS OF THIS FILE
---------------------

 * Introduction
 * Prerequirsites
 * Folder structure
 * Dataset
 * Methods
 * Results and discussion
 * References



INTRODUCTION
---------------------
Our 'Rice Leaf' Project detects rice leaf disease by classifying different leaf disease
species. The purpose of this project is to limit crop damage and further allows agriculture 
industry to produce more rice for the whole society.



PREREQUISITES
---------------------

The following open source packages are used in this project:
 * torch
 * numpy
 * matplotlib
 * PIL
 * sklearn
 * OpenCV



FOLDER STRUTURE
---------------------
We developed our project in Jupyter notebook. Then we converted each notebook file into .py file. So, in our submission codes folder, there are ipynb files along with corresponding .py files.

data
.
│
├── train_data_original.csv
├── val_data_original.csv
├── test_data_original.csv
├── train_data_preprocessed.csv
├── val_data_preprocessed.csv
├── test_data_preprocessed.csv

codes
.
│
├── StructureData
|   ├──COMP9417_convertData.ipynb # Split the whole dataset into train, validation, and testing. Then convert the unstructured data into structured csv files.
|
├── preprocessing
|   ├──COMP9417_preprocessing.ipynb # preprocess data
|   ├──demo.ipynb
│   ├──mmsegementation
|   ├──output
|
├── DT_KNN
│   ├──COMP9417_feature_extraction_and_DecisionTree.ipynb # extract all features and implement decision trees 
│   ├──COMP9417_feature_extraction_for_knn.ipynb # Crop the image according to the lesion area for KNN                    
│   ├──COMP9417_Knn_riceleaf.ipynb # implement KNN
|
├── SimpleCNN
│   ├──COMP9417_CNN.ipynb # Used a simple non-pretrained CNN model for the classification task.
│
├── TransferLearning_Densenet
│   ├── COMP9417_model_select.ipynb # Select and fine tune model
│   ├── COMP9417_dense161_svm.ipynb # Used CNN as feature extractor and SVM as classifier
│   ├── COMP9417_dense201_processed.ipynb # Testing result



DATASET
---------------------

The dataset used is the "Rice Leafs - An image collection four rice diseases" at Kaggle, which aims to classify different leaf diseases and have real world significance. In our experimentation, all the 3355 images are used for training, validation and testing.

You can download the orignal dataset from:
https://www.kaggle.com/datasets/shayanriyaz/riceleafs
After downloading the original dataset from Kaggle, you need to manually combine the train and validation sets. Then you can run the COMP9417_convertData.ipynb under StructureData folder to randomly split the dataste into training set, validation set and testing set.

If you want to download the original dataset after split or if you want to download the preprocessed dataset after split, you can use the Google Drive link below:
https://drive.google.com/drive/folders/1HersHqd5UXeMqFbOWV5WypnlVNzUQJ5r?usp=sharing



METHODS
---------------------

Multiple classification of detailed images is always challenging. In this project, we tried Decision Trees, self-built CNN networks, and transfer learning. We preprocessed the images with identical backgrounds, and split the dataset into 8:1:1 training, validation, and test sets. We compared the various methods and tried to get better performance.



RUN
---------------------

Download the dataset from the Google Drive link provided above and the six .csv files in our submission folder and put the csv files in the data directory.
For each .ipynb file, change the path and run cells one by one.
Google Colab GPU can be used to accelerate training process.

*****************************
Preprocessing
*****************************
For the preprocessing codes, you can download the latest.pth files from:
https://drive.google.com/file/d/1UOfdd6X83Jl9Sex-7fA9gmzCjqDqJUFz/view?usp=sharing


preprocessing.ipynb can preprocess the original image and generate preprocessed images. 

After download the data and lastest.pth, you can change the data_dir and out_dir to generate image by it.

demo.ipynb is a demo of the preprocessing effect, you can run it to test the effect.
There are a few pictures are kept in preprocessing/mmsegmentation/data 
you can simply test the preprocessing effect by them.

*****************************
SimpleCNN
*****************************
For the SimpleCNN codes, just simply change the path of DIR_DATA, DIR_CSV, and DIR_MODEL. Then change the input_data to 'original' or 'preprocessed'.
The trained model parameters that used to generate the results in the report can be downloaded at:
https://drive.google.com/drive/folders/1RImSYAE7MbUc9KTDRD5v4z_huBJu9gZy?usp=sharing



RESULTS
---------------------

Our final model selection is Pretrained CNN model as it fits this dataset the most as it gives a highest accuacy of 0.86 for testing dataset and a highest f1 score of 0.8564 among all the models we used. 



REFERENCE
---------------------

https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html                                                  https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html                                          https://www.kaggle.com/datasets/shayanriyaz/riceleafs
https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/5?u=kuzand
https://www.kaggle.com/code/mihirpaghdal/intel-image-classification-with-pytorch
https://medium.com/bitgrit-data-science-publication/building-an-image-classification-model-with-pytorch-from-scratch-f10452073212
https://appsilon.com/visualize-pytorch-neural-networks/
https://www.kaggle.com/code/mehmetlaudatekman/rice-leaf-pytorch-transfer-learning 
https://github.com/open-mmlab/mmsegmentation
