Extended-Eigenclassifiers
=========================
NOTE: There can be redundant and extra codes inside the scripts for test purposes.
Dependency: drtoolbox, matlab toolbox for dimensionality reduction
/// These scripts prepares (divides to folds, test, val, train) the raw 
/// dataset for later use by classifier algorithms
-prepareDataFusion.m
-prepareDataFusionEA.m
-prepareEA.m

/// This script contains Eigenclassifiers, Kernelized Eigenclassifiers, SVMs
/// Neural Networks with dropout
-applyMethodsEA.m
	-BNN.m
	-BNNDropout.m
	-FNN.m
	-FNNDropout.m
	-onlineLearningVal.m
	-SVMoptimalLin.m
	-trainNN

/// This script also tests SVM classifiers
SVM_Cases.m
	
/// This scripts implements Extended Eigenclassifiers
EigenClassifiersMulti3Modal.m

/// These scripts prepares and implements Kernelized Extended Eigenclassifiers
XMEC_Kernel.m
XMEC_test.m

/// This script compares variances of Eigenclassifiers and Extended Eigenclassifiers
EC_XMEC_VAR_COMPARE.m

/// These scripts extracts features and implements the decision tree to select a fusion method
/// The argument "precMat2" given to methods is the table 1 in the paper.
FUSMETHODSELPYTHON.m
dec_treep.y