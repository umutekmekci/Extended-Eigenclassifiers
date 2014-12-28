function [ output ] = EC_XMEC_VAR_COMPARE()

datasets{1} = 'australian';  datasets{2} = 'balance';  datasets{3} = 'breast';
datasets{4} = 'bupa';  datasets{5} = 'car';  datasets{6} = 'cmc';
datasets{7} = 'credit';  datasets{8} = 'cylinder';  datasets{9} = 'dermatology';
datasets{10} = 'ecoli';  datasets{11} = 'flags';  datasets{12} = 'flare';
datasets{13} = 'glass';  datasets{14} = 'haberman';  datasets{15} = 'heart'; 
datasets{16} = 'hepatitis';  datasets{17} = 'horse';  datasets{18} = 'ionosphere';
datasets{19} = 'iris';  datasets{20} = 'monks';  datasets{21} = 'mushroom';
datasets{22} = 'nursery';  datasets{23} = 'optdigits';  datasets{24} = 'pageblock';
datasets{25} = 'pendigits';  datasets{26} = 'pima';  datasets{27} = 'ringnorm';
datasets{28} = 'segment';  datasets{29} = 'spambase';  datasets{30} = 'tae';
datasets{31} = 'thyroid';  datasets{32} = 'tictactoe';  datasets{33} = 'titanic';
datasets{34} = 'twonorm';  datasets{35} = 'vote';  datasets{36} = 'wine';
datasets{37} = 'yeast';  datasets{38} = 'zoo';


for mm  = 1:length(datasets)
dataset = datasets{mm};
name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEA\',dataset];
matobj = matfile(name);
trainCCTemp = matobj.trainCC;
clear matobj;

name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfiles\',dataset];
matobj = matfile(name);
testCC = matobj.testCC;
clear matobj;

name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEASVD\',dataset,'multipcamappingmodal2'];
matobj = matfile(name,'Writable',true);
U_map = matobj.U;
mean_K = matobj.mean_K;
clear matobj;    

dimsize = 6;
for cl = 1: 10 
    trainCC = trainCCTemp{cl};
    trainPCA = zeros(dimsize,size(trainCC,3));
    for i = 1:size(trainCC,3)
        temp = trainCC(:,:,i);
        Temp = zeros(size(temp,1)*size(temp,2),size(temp,2));
        satbas=1; aralik = size(temp,1); satson = aralik;
        for j = 1:size(temp,2)
            Temp(satbas:satson,j) = temp(:,j);
            satbas = satson + 1;
            satson = satson + aralik;
        end
        temp = Temp;
        [U,S,~] = svd(temp);
        S = diag(S);  [S,ind] = sort(S, 'descend');  U = U(:,ind);
        K = U(:,1)*S(1) - mean_K;
        trainPCA(:,i) = U_map(:,1:dimsize)'*K;
    end
    vartraindata(cl) = sum(sum(abs(cov(trainPCA'))));
    
    testData = zeros(dimsize,size(testCC,3));
    for i = 1:size(testCC,3)
        temp = testCC(:,:,i);
        Temp = zeros(size(temp,1)*size(temp,2),size(temp,2));
        satbas=1; aralik = size(temp,1); satson = aralik;
        for j = 1:size(temp,2)
            Temp(satbas:satson,j) = temp(:,j);
            satbas = satson + 1;
            satson = satson + aralik;
        end
        temp = Temp;
        [U,S,~] = svd(temp);
        S = diag(S);  [S,ind] = sort(S, 'descend'); U = U(:,ind);
        K = U(:,1)*S(1) - mean_K;
        testData(:,i) = U_map(:,1:dimsize)'*K;
    end
    vartestdata(cl) = sum(sum(abs(cov(testData'))));
end
Xvartrain(mm) = mean(vartraindata);
Xvartest(mm) = mean(vartestdata);
end

datasets{1} = 'australian';  datasets{2} = 'balance';  datasets{3} = 'breast';
datasets{4} = 'bupa';  datasets{5} = 'car';  datasets{6} = 'cmc';
datasets{7} = 'credit';  datasets{8} = 'cylinder';  datasets{9} = 'dermatology';
datasets{10} = 'ecoli';  datasets{11} = 'flags';  datasets{12} = 'flare';
datasets{13} = 'glass';  datasets{14} = 'haberman';  datasets{15} = 'heart'; 
datasets{16} = 'hepatitis';  datasets{17} = 'horse';  datasets{18} = 'ionosphere';
datasets{19} = 'iris';  datasets{20} = 'monks';  datasets{21} = 'mushroom';
datasets{22} = 'nursery';  datasets{23} = 'optdigits';  datasets{24} = 'pageblock';
datasets{25} = 'pendigits';  datasets{26} = 'pima';  datasets{27} = 'ringnorm';
datasets{28} = 'segment';  datasets{29} = 'spambase';  datasets{30} = 'tae';
datasets{31} = 'thyroid';  datasets{32} = 'tictactoe';  datasets{33} = 'titanic';
datasets{34} = 'twonorm';  datasets{35} = 'vote';  datasets{36} = 'wine';
datasets{37} = 'yeast';  datasets{38} = 'zoo';

for mm = 1:length(datasets)
dataset = datasets{mm};
name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEA\',dataset];
matobj = matfile(name);
trainCCTemp = matobj.trainCC;
clear matobj;


name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfiles\',dataset];
matobj = matfile(name);
testCC = matobj.testCC;
clear matobj;

name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEA\',dataset,'pcamappingorg'];
matobj = matfile(name);
mapping = matobj.mapping;
clear matobj;

for cl = 1: 10 
    trainCC = trainCCTemp{cl};
    dimsize = length(mapping.lambda);
    trainPCA = zeros(dimsize*size(trainCC,2),size(trainCC,3));
    for i = 1:size(trainCC,3)
        temp = trainCC(:,:,i);
        temp = out_of_sample(temp',mapping);  temp = temp';
        trainPCA(:,i) = temp(:);
    end
    vartraindata(cl) = sum(sum(abs(cov(trainPCA'))));
 
    testData = zeros(dimsize*size(testCC,2),size(testCC,3));
    for i = 1:size(testCC,3)
        temp = testCC(:,:,i);
        temp = out_of_sample(temp',mapping);  temp = temp';
        testData(:,i) = temp(:);
    end
    vartestdata(cl) = sum(sum(abs(cov(testData'))));
end
Evartrain(mm) = mean(vartraindata);
Evartest(mm) = mean(vartestdata);
end

plot(1:38,Xvartest,'g', 1:38,Evartest,'r');


output = 1;
end

