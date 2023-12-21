# FPtand（pairwise molecular embedding）  
## Introduction
FPtand is a model using Text-CNN to generate molecular embedding based on donor and acceptor pair data in organic solar cell for predicting power conversion efficiency(PCE). 
<br>The model is stored in **model8_fp**, and the files **fa_fp.npy** and **fd_fp.npy** are training data of acceptor and donor, respectively.
<br>**fp_Y.npy** is the file containing target PCE values, and **fp1.py** is the main program code.
## Programs and Files:
**FPpara** is a comparative experimental model that manipulates the data in parallel to make a final prediction.
<br>The **MLemb** includes experiments with four other baseline models: RF、SVR、Xgboost、 and gcforest. The inputs for the training and test models are featuremap_train.csv and featuremap_test.csv, respectively.
<br>The **fingerprint** includes experiments to embed 4 different fingerprint tests, namely APfp、 CDKfp、GRAPHfp and MACCS. The model selected the best embedding effect for different fingerprint embedding, and then selected Morgan fingerprint to train the following model.
<br>The **visualization** consists of embedding、 t-sne and shap interpretability. The first two parts are the embedding of the model using t-SNE and PCA to visualize the scatter plot. shap reconstructs the model on the basis of the original data and adds some new data sources to explain the model's predictive ability. In the process of accurately predicting the results of the model, which fingerprints contribute more.
## Environment
Configuring the environment according to **fptand.yml** before running the code.
