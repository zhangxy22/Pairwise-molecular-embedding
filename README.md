#Pairwise molecular embedding  
The following programs are designed for generating pairwise molecular embedding and modeling for power conversion efficiency(PCE) of small organic molecular solar cell.
 
## Details of programs
**FPtand** is a folder containing the FPtand model that is using Text-CNN to generate molecular embedding based on donor and acceptor pair data with Mogan fingerprint in organic solar cell for predicting PCE. 
<br>The model is stored in **model8_fp**, and the files **fa_fp.npy** and **fd_fp.npy** are training data of acceptor and donor, respectively.
<br>**fp_Y.npy** is the file containing target PCE values, and **fp1.py** is the main program code.

**FPpara** is a folder containing a similar model as FPtand except that manipulates the donor and acceptor data in parallel to make a final prediction.

**MLemb** includes  four baseline manchine learning models: RF、SVR、Xgboost、 and gcforest. The inputs for the training and test of models are featuremap_train.csv and featuremap_test.csv, respectively.

**fingerprint** includes FPtand models with embedding from four types of fingerprints, namely APfp、 CDKfp、GRAPHfp and MACCS. 

**visualization** consists of  t-SNE and Shap plot program for embedding. 

## Environment
Configuring the environment according to **FPtand.yml** before running the code.
