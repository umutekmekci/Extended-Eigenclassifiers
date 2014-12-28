function [ output ] = applyMethodsEA

%{
dataset = 'pageblock';
name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEA\',dataset,'Var'];
matobj = matfile(name);
vv = matobj.var;
clear matobj;

name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEA\',dataset,'VarEA'];
matobj = matfile(name);
vvea = matobj.var;
clear matobj;

bar([vv',vvea'])
%}

%% PCA + linear classifier on classifier outputs

clc
%rng(150);
dataset = 'pageblock';
name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEA\',dataset];
matobj = matfile(name);
trainCCTemp = matobj.trainCC;
trainPCATemp = matobj.trainPCA;
targetCCTemp = matobj.targetCC;
trainCCVal2 = matobj.trainCCVal2;
targetCCVal2 = matobj.targetCCVal2;
clear matobj;

mmax = 1;
for cl = 1:10
    trainTarget = targetCCTemp{cl};
    maxi = max(trainTarget);
    mmax = max(maxi,mmax);
end

name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfiles\',dataset];
matobj = matfile(name);
testCC = matobj.testCC;
testTarget = matobj.testTarget;
clear matobj;

name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEA\',dataset,'pcamapping'];
matobj = matfile(name);
mapping = matobj.mapping;
clear matobj;

weights = cell(1,10);
bias = cell(1,10);
DogruTest = 0;
DogruTrain = 0;
DogruAvarage = 0;
    c = 1;
    g = 1;
for cl = 1: 10 
    trainPCA = trainPCATemp{cl};
    trainCC = trainCCTemp{cl};
    trainTarget = targetCCTemp{cl};
    valset = trainCCVal2{cl};
    valsetTarget = targetCCVal2{cl};
    trainTarget = full(ind2vec(trainTarget));
    valsetTarget = full(ind2vec(valsetTarget));
    if size(trainTarget,1) ~= mmax
        trainTarget = [trainTarget; zeros(mmax-size(trainTarget-1),size(trainTarget,2))];
    end
    if size(valsetTarget,1) ~= mmax
        valsetTarget = [valsetTarget; zeros(mmax-size(valsetTarget-1),size(valsetTarget,2))];
    end
    
    %W = applyPCA(trainPCA,95);
    %dimsize = size(W,1);
    dimsize = length(mapping.lambda);
    trainPCA = zeros(dimsize*size(trainCC,2),size(trainCC,3));
    avaragePreTrain = zeros(1,size(trainCC,3));
    SVMtrain = zeros(size(trainCC,1)*size(trainCC,2),size(trainCC,3));
    for i = 1:size(trainCC,3)
        temp = trainCC(:,:,i);
        SVMtrain(:,i) = temp(:);
        [~,avaragePreTrain(i)] = max(mean(temp));
        %temp = W*temp;
        temp = out_of_sample(temp',mapping);  temp = temp';
        trainPCA(:,i) = temp(:);
    end
    testData = zeros(dimsize*size(testCC,2),size(testCC,3));
    avaragePreTest = zeros(1,size(testCC,3));
    SVMtest = zeros(size(testCC,1)*size(testCC,2),size(testCC,3));
    for i = 1:size(testCC,3)
        temp = testCC(:,:,i);
        SVMtest(:,i) = temp(:);
        [~,avaragePreTest(i)] = max(mean(temp));
        %temp = W*temp;
        temp = out_of_sample(temp',mapping);  temp = temp';
        testData(:,i) = temp(:);
    end
    valData = zeros(dimsize*size(valset,2),size(valset,3));
    SVMval = zeros(size(valset,1)*size(valset,2),size(valset,3));
    for i = 1:size(valset,3)
        temp = valset(:,:,i);
        SVMval(:,i) = temp(:);
        %temp = W*temp;
        temp = out_of_sample(temp',mapping);  temp = temp';
        valData(:,i) = temp(:);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%SVM
%{
    if size(trainTarget,1) == 2
        multi = false;
    else
        multi = true;
    end
    %if cl == 1
        [c,dogruval] = SVMoptimalLin(SVMtrain', vec2ind(trainTarget)', SVMval', vec2ind(valsetTarget)',c, multi);
        %[c,g,dogruval] = SVMoptimal(SVMtrain', vec2ind(trainTarget)', SVMval', vec2ind(valsetTarget)',c,g, multi);
        %[c,dogruval] = SVMoptimalLin(trainPCA', vec2ind(trainTarget)', valData', vec2ind(valsetTarget)',c, multi);
        %[c,g,dogruval] = SVMoptimal(trainPCA', vec2ind(trainTarget)', valData', vec2ind(valsetTarget)',c,g, multi);
        dogruvec(cl) = dogruval;
        cvec(cl) = c;
    %end
    tex = sprintf('-t 0 -c %f -b 1 -q',c);
    %tex = sprintf('-c %f -g %f -b 1 -q',c,g);
    dogrutrain(cl) = asist(SVMtrain', vec2ind(trainTarget)', SVMtrain',vec2ind(trainTarget)' , tex,multi);
    %dogrutrain(cl) = asist(trainPCA', vec2ind(trainTarget)', trainPCA',vec2ind(trainTarget)' , tex,multi);
    DogruTrain = DogruTrain + dogrutrain(cl);
    %if cl == 10
    %    aaaa = 10;
    %end
    [dogrutest(cl),TFP(:,:,cl)] = asist(SVMtrain', vec2ind(trainTarget)', SVMtest',vec2ind(testTarget)' , tex,multi);
    %[dogrutest(cl), TFP(:,:,cl)] = asist(trainPCA', vec2ind(trainTarget)', testData',vec2ind(testTarget)' , tex,multi);
    DogruTest = DogruTest + dogrutest(cl);
end
DogruTest = round(DogruTest/10)
DogruTrain = round(DogruTrain/10)
[~,ind] = max(dogruvec);
dogrutest(ind)
dogrutrain(ind)
[micro, macro] = micmac(mean(TFP,3))
%%%%%%%%%%%%%%%%%%%%%%%%%%
    %}
    
    %CCC = abs(cov(trainPCA'));
    %(sum(sum(CCC))-sum(diag(CCC)))/numel(CCC)
    % prepare linear classifier
    funcList{1} = 'purelin';
    arc = [size(trainPCA,1),size(trainTarget,1)];
    EC = 'mse';
    maxIter = 10000; minError = 1e-5;
    isnorm = true;

    % train the network
    
    [weights{cl},bias{cl},~,~, valerror(cl)] = trainNN( trainPCA,trainTarget,[],[],funcList,arc,EC,maxIter,minError,isnorm,valData, valsetTarget);
    %plot(1:length(valerror),valerror);
    %hold on; plot(1:length(errorVec),errorVec,'g'); hold on;
    output = FNN( trainPCA, weights{cl},bias{cl}, funcList);
    [~,output] = max(output);
    ddtrain(cl) = sum(avaragePreTrain == vec2ind(trainTarget));
    DogruAvarage = DogruAvarage + ddtrain(cl);
    DogruTrain = DogruTrain + sum(output == vec2ind(trainTarget));
    fprintf('Toplam = %d, avarage = %d,  Dogru = %d\n',size(output,2),sum(avaragePreTrain == vec2ind(trainTarget)), sum(vec2ind(trainTarget) == output));

% test
    output = FNN( testData, weights{cl},bias{cl}, funcList);  %output1 = output; save output1;
    [~,output] = max(output);
    ddtest(cl) = sum(output == vec2ind(testTarget));
    DogruTest = DogruTest + ddtest(cl);
    fprintf('Toplam = %d, avarage = %d,  Dogru = %d\n',size(output,2), sum(avaragePreTest == vec2ind(testTarget)), sum(output == vec2ind(testTarget)));
    [TFP(:,:,cl)] = hesapet(output, testTarget, true);
end
DogruTest = round(DogruTest/10)
DogruTrain = round(DogruTrain/10)
DogruAvarage = round(DogruAvarage/10)
[~,ind] = min(valerror);
ddtest(ind)
ddtrain(ind);
%[micro, macro] = micmac(mean(TFP,3))
%tempW = [];
%for cl = 1:10
%    ww = weights{cl};
%    bb = bias{cl};
%    temp = [];
%    for i = 1:length(ww)
%        www = ww{i};
%        bbb = bb{i};
%        temp = [temp;www(:);bbb(:)];
%    end
%    tempW(:,cl) = temp;
%end
%vv = [];
%for i = 1:size(tempW,1)
%    vv = [vv, var(tempW(i,:))];
%end
%Dogru = round(Dogru/10)
%name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEA\',dataset,'Var'];
%matobj = matfile(name);
%matobj.var = vv;
%clear matobj;
%}

%% PCA + dropout linear classifier on classifier outputs

clear;
dataset = 'cylinder';
name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfiles\',dataset];
matobj = matfile(name);
testCC = matobj.testCC;
testTarget = matobj.testTarget;
clear matobj;
name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEA\',dataset];
matobj = matfile(name);
trainCCTemp = matobj.trainCC;
trainPCATemp = matobj.trainPCA;
targetCCTemp = matobj.targetCC;
trainCCVal2 = matobj.trainCCVal2;
targetCCVal2 = matobj.targetCCVal2;
clear matobj;

weights = cell(1,10);
bias = cell(1,10);
DogruTest = 0;
DogruTrain = 0;
DogruAvarage = 0;
for cl = 1: 10 
    trainPCA = trainPCATemp{cl};
    trainCC = trainCCTemp{cl};
    trainTarget = targetCCTemp{cl};
    valset = trainCCVal2{cl};
    valsetTarget = targetCCVal2{cl};
    trainTarget = full(ind2vec(trainTarget));
    valsetTarget = full(ind2vec(valsetTarget));
    
    W = applyPCA(trainPCA,99);
    trainPCA = zeros(size(W,1)*size(trainCC,2),size(trainCC,3));
    avaragePreTrain = zeros(1,size(testCC,3));
    for i = 1:size(trainCC,3)
        temp = trainCC(:,:,i);
        [~,avaragePreTrain(i)] = max(mean(temp));
        temp = W*temp;
        trainPCA(:,i) = temp(:);
    end
    testData = zeros(size(W,1)*size(testCC,2),size(testCC,3));
    avaragePreTest = zeros(1,size(testCC,3));
    for i = 1:size(testCC,3)
        temp = testCC(:,:,i);
        [~,avaragePreTest(i)] = max(mean(temp));
        temp = W*temp;
        testData(:,i) = temp(:);
    end
    
    valData = zeros(size(W,1)*size(valset,2),size(valset,3));
    for i = 1:size(valset,3)
        temp = valset(:,:,i);
        temp = W*temp;
        valData(:,i) = temp(:);
    end

% prepare linear classifier
    funcList{1} = 'purelin';
    arc = [size(trainPCA,1),size(trainTarget,1)];
    EC = 'mse';
    isnorm = true;
    epoch = 5000;
    dropProb = [0.1,0];

    [weights{cl},bias{cl},~,~,~,valerror] = onlineLearning( trainPCA,trainTarget,[],[],funcList,arc, EC,isnorm,0,epoch, dropProb,30,25,0.01,0.7,0.0005, valData, valsetTarget);
    %plot(1:length(valerror),valerror);
    output = FNNDropout( trainPCA, weights{cl},bias{cl}, funcList, [0 0]);
    [~,output] = max(output);
    DogruAvarage = DogruAvarage + sum(avaragePreTrain == vec2ind(trainTarget));
    DogruTrain = DogruTrain + sum(output == vec2ind(trainTarget));
    fprintf('Toplam = %d, avarage = %d,  Dogru = %d\n',size(output,2),sum(avaragePreTrain == vec2ind(trainTarget)), sum(vec2ind(trainTarget) == output));

% test
    output = FNN( testData, weights{cl},bias{cl}, funcList);  %output2 = output; save output2;
    [~,output] = max(output);
    DogruTest = DogruTest + sum(output == vec2ind(testTarget));
    fprintf('Toplam = %d, avarage = %d,  Dogru = %d\n',size(output,2), sum(avaragePreTest == vec2ind(testTarget)), sum(output == vec2ind(testTarget)));
end
DogruTest = round(DogruTest/10)
DogruTrain = round(DogruTrain/10)
DogruAvarage = round(DogruAvarage/10)


%% Kernel PCA + linear classifier on classifier outputs

clear;
dataset = 'pageblock';
name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfiles\',dataset];
matobj = matfile(name);
testCC = matobj.testCC;
testTarget = matobj.testTarget;
clear matobj;
name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEA\',dataset];
matobj = matfile(name);
trainCCTemp = matobj.trainCC;
trainPCATemp = matobj.trainPCA;
targetCCTemp = matobj.targetCC;
trainCCVal2 = matobj.trainCCVal2;
targetCCVal2 = matobj.targetCCVal2;
clear matobj;

mmax = 1;
for cl = 1:10
    trainTarget = targetCCTemp{cl};
    maxi = max(trainTarget);
    mmax = max(maxi,mmax);
end

weights = cell(1,10);
bias = cell(1,10);
DogruTest = 0;
DogruTrain = 0;

name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEA\',dataset,'kernelmapping'];
    matobj = matfile(name);
    mapping = matobj.mapping;
    clear matobj;

for cl = 1:10
    trainPCA = trainPCATemp{cl};
    trainCC = trainCCTemp{cl};
    trainTarget = targetCCTemp{cl};
    valset = trainCCVal2{cl};
    valsetTarget = targetCCVal2{cl};
    trainTarget = full(ind2vec(trainTarget));
    valsetTarget = full(ind2vec(valsetTarget));
    if size(trainTarget,1) ~= mmax
        trainTarget = [trainTarget; zeros(mmax-size(trainTarget-1),size(trainTarget,2))];
    end
    if size(valsetTarget,1) ~= mmax
        valsetTarget = [valsetTarget; zeros(mmax-size(valsetTarget-1),size(valsetTarget,2))];
    end
%tic
%    [~,U, Ks, M] = kernelPCA(trainPCA, 95);
    %dimsize = 65;
    %[~, mapping] = kernel_pca(trainPCA',dimsize, 'gauss', 0.1);
    %mapping.name = 'KernelPCA';
    
    
%toc
    %name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEA\',dataset,'kernel',num2str(cl)];
    %matobj = matfile(name);
    %matobj.U = U;
    %matobj.Ks = Ks;
    %matobj.M = M;
    %clear matobj;
    dimsize = size(mapping.V,2);
    trainData = zeros(dimsize*size(trainCC,2),size(trainCC,3));
    avaragePreTrain = zeros(1,size(testCC,3));
%tic;
    for i = 1:size(trainCC,3)
        temp = trainCC(:,:,i);
        [~,avaragePreTrain(i)] = max(mean(temp));
        %temp = applyKernelPCA( temp, trainPCA, U, Ks, M);
        temp = out_of_sample(temp',mapping);  temp = temp';
        trainData(:,i) = temp(:);
        disp(i);
    end
%toc
    %name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEA\',dataset,'kernelTrain',num2str(cl)];
    %matobj = matfile(name);
    %trainData = matobj.trainData;
    %clear matobj;

    testData = zeros(dimsize*size(testCC,2),size(testCC,3));
    avaragePreTest = zeros(1,size(testCC,3));
    for i = 1:size(testCC,3)
        temp = testCC(:,:,i);
        [~,avaragePreTest(i)] = max(mean(temp));
        %temp = applyKernelPCA( temp, trainPCA, U, Ks, M);
        temp = out_of_sample(temp',mapping);  temp = temp';
        testData(:,i) = temp(:);
    end
    valData = zeros(dimsize*size(valset,2),size(valset,3));
    for i = 1:size(valset,3)
        temp = valset(:,:,i);
        temp = out_of_sample(temp',mapping);  temp = temp';
        valData(:,i) = temp(:);
    end
    
    %name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEA\',dataset,'kernelTest',num2str(cl)];
    %matobj = matfile(name);
    %testData = matobj.testData;
    %clear matobj;
%CCC = abs(cov(trainData'));
%(sum(sum(CCC))-sum(diag(CCC)))/numel(CCC)
% prepare linear classifier
    funcList{1} = 'purelin';
    arc = [size(trainData,1),size(trainTarget,1)];
    EC = 'mse';
    maxIter = 10000; minError = 1e-5;
    isnorm = true;

% train the network

    %[weights{cl},bias{cl},~] = trainNN( trainData,trainTarget,[],[],funcList,arc,EC,maxIter,minError,isnorm);
    [weights{cl},bias{cl},~,~, valerror(cl)] = trainNN( trainData,trainTarget,[],[],funcList,arc,EC,maxIter,minError,isnorm,valData, valsetTarget);
    output = FNN( trainData, weights{cl},bias{cl}, funcList);
    [~,output] = max(output);
    ddtrain(cl) = sum(output == vec2ind(trainTarget));
    DogruTrain = DogruTrain + ddtrain(cl);
    fprintf('Toplam = %d, avarage = %d,  Dogru = %d\n',size(output,2),sum(avaragePreTrain == vec2ind(trainTarget)), sum(vec2ind(trainTarget) == output));

% test
    output = FNN( testData, weights{cl},bias{cl}, funcList);  %output3 = output;  save output3;
    [~,output] = max(output);
    ddtest(cl) = sum(output == vec2ind(testTarget));
    DogruTest = DogruTest + ddtest(cl);
    fprintf('Toplam = %d, avarage = %d,  Dogru = %d\n',size(output,2), sum(avaragePreTest == vec2ind(testTarget)), sum(output == vec2ind(testTarget)));
    [TFP(:,:,cl)] = hesapet(output, testTarget, true);
end
DogruTest = round(DogruTest/10)
DogruTrain = round(DogruTrain/10)
[~,ind] = min(valerror);
ddtrain(ind)
ddtest(ind)
[micro, macro] = micmac(mean(TFP,3))


%% Kernel PCA + dropout linear classifier on classifier outputs 

clear;
dataset = 'pageblock';
name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfiles\',dataset];
matobj = matfile(name);
testCC = matobj.testCC;
testTarget = matobj.testTarget;
clear matobj;
name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEA\',dataset];
matobj = matfile(name);
trainCCTemp = matobj.trainCC;
targetCCTemp = matobj.targetCC;
trainCCVal2 = matobj.trainCCVal2;
targetCCVal2 = matobj.targetCCVal2;
clear matobj;

mmax = 1;
for cl = 1:10
    trainTarget = targetCCTemp{cl};
    maxi = max(trainTarget);
    mmax = max(maxi,mmax);
end

name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEA\',dataset,'kernelmapping'];
    matobj = matfile(name);
    mapping = matobj.mapping;
    clear matobj;

weights = cell(1,10);
bias = cell(1,10);
DogruTest = 0;
DogruTrain = 0;
for cl = 1:10
    trainCC = trainCCTemp{cl};
    trainTarget = targetCCTemp{cl};
    valset = trainCCVal2{cl};
    valsetTarget = targetCCVal2{cl};
    trainTarget = full(ind2vec(trainTarget));
    valsetTarget = full(ind2vec(valsetTarget));
    if size(trainTarget,1) ~= mmax
        trainTarget = [trainTarget; zeros(mmax-size(trainTarget-1),size(trainTarget,2))];
    end
    if size(valsetTarget,1) ~= mmax
        valsetTarget = [valsetTarget; zeros(mmax-size(valsetTarget-1),size(valsetTarget,2))];
    end
    
    dimsize = size(mapping.V,2);
    trainData = zeros(dimsize*size(trainCC,2),size(trainCC,3));
    avaragePreTrain = zeros(1,size(testCC,3));
    for i = 1:size(trainCC,3)
        temp = trainCC(:,:,i);
        [~,avaragePreTrain(i)] = max(mean(temp));
        temp = out_of_sample(temp',mapping);  temp = temp';
        trainData(:,i) = temp(:);
    end
    %name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEA\',dataset,'kernelTrain',num2str(cl)];
    %matobj = matfile(name);
    %trainData = matobj.trainData;
    %clear matobj;
    
    testData = zeros(dimsize*size(testCC,2),size(testCC,3));
    avaragePreTest = zeros(1,size(testCC,3));
    for i = 1:size(testCC,3)
        temp = testCC(:,:,i);
        [~,avaragePreTest(i)] = max(mean(temp));
        temp = out_of_sample(temp',mapping);  temp = temp';
        testData(:,i) = temp(:);
    end
    %name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEA\',dataset,'kernelTest',num2str(cl)];
    %matobj = matfile(name);
    %testData = matobj.testData;
    %clear matobj;
    
    valData = zeros(dimsize*size(valset,2),size(valset,3));
    for i = 1:size(valset,3)
        temp = valset(:,:,i);
        temp = out_of_sample(temp',mapping);  temp = temp';
        valData(:,i) = temp(:);
    end
    
% prepare linear classifier
    funcList{1} = 'purelin';
    arc = [size(trainData,1),size(trainTarget,1)];
    EC = 'mse';
    isnorm = true;
    epoch = 100000;
    dropProb = [0.1,0,0];
    lrbas = 0.05;
    mmbas = 0.7;
    wd = 0.000001;

    %[weights{cl},bias{cl}] = onlineLearning( trainData,trainTarget,[],[],funcList,arc, EC,isnorm,0,epoch, dropProb,30);
    [weights{cl}, bias{cl},~,~,~,valerror(cl)] = onlineLearningVal(trainData,trainTarget,[],[],funcList,arc, EC,isnorm,0,epoch, dropProb,80,25,lrbas,mmbas,wd, valData, valsetTarget);
    output = FNNDropout( trainData, weights{cl},bias{cl}, funcList, dropProb);
    [~,output] = max(output);
    ddtrain(cl) = sum(output == vec2ind(trainTarget));
    DogruTrain = DogruTrain + ddtrain(cl);
    fprintf('Toplam = %d, avarage = %d,  Dogru = %d\n',size(output,2),sum(avaragePreTrain == vec2ind(trainTarget)), sum(vec2ind(trainTarget) == output));

% test
    output = FNNDropout( testData, weights{cl},bias{cl}, funcList, dropProb);  %output4 = output;  save output4;
    [~,output] = max(output);
    ddtest(cl) = sum(output == vec2ind(testTarget));
    DogruTest = DogruTest + ddtest(cl);
    fprintf('Toplam = %d, avarage = %d,  Dogru = %d\n',size(output,2), sum(avaragePreTest == vec2ind(testTarget)), sum(output == vec2ind(testTarget)));
end
DogruTest = round(DogruTest/10)
DogruTrain = round(DogruTrain/10)
[~,ind] = min(valerror);
ddtest(ind)
ddtrain(ind)


%% Dropout directly on row data

%{
australian:  lrbas:0.05,  mmbas:0.7  mb:80,  nn:25,  ns:60, wd:0.000001, p=[0.2 0.5 0], lrbas:0.05,  mmbas:0.7  mb:80,  nn:25,  ns:120, wd:0.000001, p=[0.2 0.5 0]
balance: lrbas:0.05,  mmbas:0.7  mb:80,  nn:25,  ns:60, wd:0.000001, p=[0 0.3 0], lrbas:0.05,  mmbas:0.7  mb:80,  nn:25,  ns:60, wd:0.000001, p=[0 0.3 0]
breast: lrbas:0.05,  mmbas:0.7  mb:80,  nn:25,  ns:60, wd:0.000001, p=[0.2 0.5 0], lrbas:0.05,  mmbas:0.7  mb:80,  nn:25,  ns:120, wd:0.000001, p=[0.2 0.5 0]
bupa:lrbas:0.05,  mmbas:0.7  mb:50,  nn:25,  ns:60, wd:0.000001, p=[0.2 0.5 0], lrbas:0.05,  mmbas:0.7  mb:50,  nn:25,  ns:120, wd:0.000001, p=[0.2 0.5 0]
car:lrbas:0.05,  mmbas:0.7  mb:100,  nn:25,  ns:150, wd:0.000001, prob=[0 0.3 0], lrbas:0.05,  mmbas:0.7  mb:100,  nn:25,  ns:150, wd:0.000001, prob=[0 0.3 0]
cmc:lrbas:0.05,  mmbas:0.7  mb:100,  nn:25,  ns:120, wd:0.000001, pr = [0.2 0.3 0], lrbas:0.05,  mmbas:0.7  mb:100,  nn:25,  ns:260, wd:0.000001, prob=[0 0.2 0]
credit:lrbas:0.05,  mmbas:0.7  mb:100,  nn:25,  ns:120, wd:0.000001, p=[0.2 0.3 0], lrbas:0.05,  mmbas:0.7  mb:100,  nn:25,  ns:120, wd:0.000001, p=[0.2 0.3 0]
cylinder:lrbas:0.01,  mmbas:0.7  mb:30,  nn:25,  ns:150, wd:0.000001, pr = [0.2 0.5 0],  lrbas:0.01,  mmbas:0.7  mb:30,  nn:35,  ns:100, wd:0.000001, pr = [0 0.5 0],
dermatology:lrbas:0.05,  mmbas:0.7  mb:50,  nn:25,  ns:60, wd:0.000001, p=[0.2 0.5 0], same
ecoli:lrbas:0.05,  mmbas:0.7  mb:50,  nn:25,  ns:150, wd:0.000001, P=[0.2 0.5 0], same
flags:lrbas:0.05,  mmbas:0.7  mb:20,  nn:25,  ns:60, wd:0.000001, pr = [0.2 0.5 0], same
flare:lrbas:0.05,  mmbas:0.7  mb:50,  nn:25,  ns:60, wd:0.000001, p=[0.2 0.5 0], same
glass:lrbas:0.01,  mmbas:0.7  mb:20,  nn:25,  ns:60, wd:0.00001,p = [0 0.2 0], lrbas:0.01,  mmbas:0.7  mb:20,  nn:25,  ns:60, wd:0.00001 iter:150000, prob = [0 0.2 0]
haberman:lrbas:0.01,  mmbas:0.7  mb:20,  nn:25,  ns:60, wd:0.00001, p =[0.2 0.5 0], lrbas:0.01,  mmbas:0.7  mb:20,  nn:25,  ns:60, wd:0.00001 iter:150000, prob = [0 0.2 0]
heart:lrbas:0.01,  mmbas:0.7  mb:50,  nn:25,  ns:60, wd:0.000001,p=[0.2 0.5 0], same
hepatitis:lrbas:0.01,  mmbas:0.7  mb:20,  nn:25,  ns:60, wd:0.00001,p=[0.2 0.5 0], same
horse:lrbas:0.01,  mmbas:0.7  mb:20,  nn:25,  ns:60, wd:0.00001, p=[0.2 0.5 0], same
ionosphere:lrbas:0.01,  mmbas:0.7  mb:20,  nn:25,  ns:60, wd:0.00001,p=[0.2 0.5 0], same
iris:lrbas:0.05,  mmbas:0.7  mb:20,  nn:25,  ns:60, wd:0.000001, p=[0.2 0.5 0], same
monks:lrbas:0.05,  mmbas:0.7  mb:50,  nn:25,  ns:120, wd:0.000001, p=[0 0.3 0], same
mushroom:lrbas:0.05,  mmbas:0.7  mb:150,  nn:25,  ns:60, wd:0.000001, p=[0.2 0.5 0], same
nursery:lrbas:0.01,  mmbas:0.7  mb:100,  nn:25,  ns:60, wd:0.00001, p=[0 0.2 0], same
optdigits:lrbas:0.05,  mmbas:0.7  mb:150,  nn:25,  ns:120, wd:0.000001, p=[0.1 0.2 0], lrbas:0.05,  mmbas:0.7  mb:150,  nn:25,  ns:120, wd:0.000001, p=[0.2 0.3 0]
pageblock:lrbas:0.05,  mmbas:0.7  mb:150,  nn:25,  ns:120, wd:0.00001, p=[0 0.2 0], same
pendigits:lrbas:0.05,  mmbas:0.7  mb:150,  nn:25,  ns:160, wd:0.000001, p=[0 0.2 0], lrbas:0.05,  mmbas:0.7  mb:150,  nn:25,  ns:200, wd:0.000001, p=[0.1 0.2 0], iter = 20..
pendigits:lrbas:0.05,  mmbas:0.7  mb:50,  nn:25,  ns:60, wd:0.000001, p=[0 0.5 0], lrbas:0.05,  mmbas:0.7  mb:100,  nn:35,  ns:200, wd:0.000001, p=[0.1 0.2 0], iter = 20..
ringnorm:0.01,  mmbas:0.7  mb:150,  nn:25,  ns:200, wd:0.000001, p=[0.1 0.2 0], lrbas:0.01,  mmbas:0.7  mb:100,  nn:25,  ns:200, wd:0.000001, p=[0 0.2 0], iter = 20..
segment:0.05,  mmbas:0.7  mb:100,  nn:25,  ns:120, wd:0.000001,p=[0 0.2 0], lrbas:0.05,  mmbas:0.7  mb:100,  nn:25,  ns:200, wd:0.000001, p=[0 0.2 0], iter = 20..
spambase:0.01,  mmbas:0.7  mb:150,  nn:25,  ns:120, wd:0.000001,p=[0.2 0.3 0], lrbas0.01,  mmbas:0.7  mb:150,  nn:25,  ns:120, wd:0.000001,p=[0.2 0.3 0]
tae:0.05,  mmbas:0.7  mb:20,  nn:25,  ns:60, wd:0.000001,p=[0.2 0.5 0], 0.01,  mmbas:0.7  mb:20,  nn:25,  ns:120, wd:0.000001,p=[0.2 0.5 0]
thyroid:0.05,  mmbas:0.7  mb:100,  nn:35,  ns:60, wd:0.000001, p=[0.2 0.5 0], 0.05,  mmbas:0.7  mb:100,  nn:25,  ns:60, wd:0.000001, p=[0.2 0.5 0]
tictactoe:0.05,  mmbas:0.7  mb:50,  nn:25,  ns:120, wd:0.000001, p=[0 0.2 0]
titanic:0.05,  mmbas:0.7  mb:80,  nn:25,  ns:60, wd:0.000001, p=[0.2 0.5 0]
twonorm:0.05,  mmbas:0.7  mb:100,  nn:25,  ns:60, wd:0.000001, p=[0.2 0.5 0], 0.05,  mmbas:0.7  mb:100,  nn:25,  ns:60, wd:0.000001, p=[0.2 0.5 0]
vote:0.05,  mmbas:0.7  mb:50,  nn:25,  ns:60, wd:0.000001, p=[0.2 0.5 0], 0.05,  mmbas:0.7  mb:50,  nn:25,  ns:60, wd:0.000001, p=[0.2 0.5 0]
wine:0.05,  mmbas:0.7  mb:20,  nn:25,  ns:60, wd:0.000001, p=[0.2 0.5 0]
yeast:0.05,  mmbas:0.7  mb:80,  nn:25,  ns:120, wd:0.000001, p=[0.2 0.5 0], 0.05,  mmbas:0.7  mb:80,  nn:25,  ns:200, wd:0.000001, p=[0 0.3 0]
zoo:0.05,  mmbas:0.7  mb:10,  nn:25,  ns:60, wd:0.000001, p=[0.2 0.5 0], 0.05,  mmbas:0.7  mb:10,  nn:25,  ns:60, wd:0.000001, p=[0.2 0.5 0]
%}

clear;
dataset = 'pageblock';
name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEA\',dataset,'raw'];
matobj = matfile(name);
trainallCell = matobj.train;
testall = matobj.testall;
valCell = matobj.val;
trainTargetCell = matobj.trainTarget;
testTarget = matobj.testTarget;
valTargetCell = matobj.valTarget;
clear matobj;
testtemp = testall;


funcList{1} = 'logsig';  funcList{2} = 'softmax'; %funcList{3} = 'softmax';
dropProb = [0,0.2,0];
EC = 'cross-entropy';
isnorm = true;
isDropout = false;  epoch = 200000;
DogruTrain = 0;
DogruTest = 0;
for i = 1:10
    trainall = trainallCell{i};
    trainTarget = trainTargetCell{i};
    val = valCell{i};
    valTarget = valTargetCell{i};
    trainTarget = full(ind2vec(trainTarget));
    testTarget = full(ind2vec(testTarget));
    valTarget = full(ind2vec(valTarget));
    [trainall,st] = zNormalize(trainall);
    val = applyZNormalize(val,st);
    
    arc = [size(trainall,1),120, size(trainTarget,1)];
    lrbas = 0.05;
    mmbas = 0.7;
    wd = 0.000001;
    
    %[weights, bias,~,~,~, ~,valerror, trainerror, son] = onlineLearningVal(trainall,trainTarget,[],[],funcList,arc, EC,isnorm,isDropout,epoch, dropProb,80,25,lrbas,mmbas,wd, val, valTarget);
    [weights, bias] = onlineLearningVal(trainall,trainTarget,[],[],funcList,arc, EC,isnorm,isDropout,epoch, dropProb,150,25,lrbas,mmbas,wd, val, valTarget);
    %plot(1:son-1,valerror(2:son-1),'b',1:son-1,trainerror(2:son-1),'r');
    output = FNNDropout( val, weights,bias, funcList, dropProb);
    [~,output] = max(output);
    [~,valTarget] = max(valTarget);
    dogruval(i) = sum(valTarget == output);
    %plot(1:length(errorval),errorval);
    output = FNNDropout( trainall, weights,bias, funcList, dropProb);
    [~,output] = max(output);
    [~,trainTarget] = max(trainTarget);
    dogrutrain(i) = sum(trainTarget == output);
    DogruTrain = DogruTrain + dogrutrain(i);
    fprintf('Toplam = %d, Dogru = %d\n',size(output,2), dogrutrain(i));
    
    testall = applyZNormalize(testtemp,st);
    output = FNNDropout(testall, weights,bias, funcList, dropProb);  %output5 = output;  save output5;
    [~,output] = max(output);
    [~,testTarget] = max(testTarget);
    dogrutest(i) = sum(output == testTarget);
    DogruTest = DogruTest + dogrutest(i);
    fprintf('Toplam = %d,Dogru = %d\n',size(output,2), dogrutest(i));
    [TFP(:,:,i)] = hesapet(output, testTarget, false);
end
DogruTrain = round(DogruTrain/10)
DogruTest = round(DogruTest/10)
[~,ind] = max(dogruval);
dogrutraint = dogrutrain(ind)
dogrutestt = dogrutest(ind)
[micro, macro] = micmac(mean(TFP,3))
%load output1;
%load output2;
%load output3;
%load output4;
%out = output1 + output2 + output3 + output4 + output5;
%[~,output] = max(out);
%fprintf('Toplam = %d,Dogru = %d\n',size(output,2), sum(output == testTarget));
%}

%% classicial NN
clear;
dataset = 'pageblock';
name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfiles\',dataset];
matobj = matfile(name);
trainall = matobj.trainall;
testall = matobj.testall;
trainTarget = matobj.trainTarget;
testTarget = matobj.testTarget;
clear matobj;

[trainall,st] = zNormalize(trainall);

funcList{1} = 'logsig';  funcList{2} = 'softmax'; %funcList{3} = 'softmax';
arc = [size(trainall,1),120, size(trainTarget,1)];
EC = 'cross-entropy';
isnorm = true;
maxIter = 1400;  minError = 1e-5;

[ weights, bias] = trainNN(  trainall,trainTarget,[],[],funcList,arc,EC,maxIter,minError,isnorm);
output = FNN( trainall, weights,bias, funcList);
[~,output] = max(output);
[~,trainTarget] = max(trainTarget);
fprintf('Toplam = %d, Dogru = %d\n',size(output,2), sum(trainTarget == output));

testall = applyZNormalize(testall,st);
output = FNN( testall, weights,bias, funcList);  %output6 = output;
[~,output] = max(output);
[~,testTarget] = max(testTarget);
fprintf('Toplam = %d,Dogru = %d\n',size(output,2), sum(output == testTarget));

%load output1;
%load output2;
%load output3;
%load output4;
%load output5;
%out = output1 + output2 + output3 + output4 + output5 + output6;
%[~,output] = max(out);
%fprintf('Toplam = %d,Dogru = %d\n',size(output,2), sum(output == testTarget));





output = 1;
end

function data = applyZNormalize(data,st)
m = repmat(st.mean,1,size(data,2));
std = repmat(st.std,1,size(data,2));
data = data - m;
data = data./std;
data(isnan(data)) = 0;
data(isinf(data)) = 0;
end

function [data, st] = zNormalize(data)
%if size(data,1) > size(data,2)
%    error('hata in zNormalize');
%end
st.mean = mean(data,2);
st.std = std(data,[],2);
data = data - repmat(st.mean,1,size(data,2));
data = data./repmat(st.std,1,size(data,2));
data(isnan(data)) = 0;
data(isinf(data)) = 0;
end

function W = applyPCA(data, energy)
C = cov(data');  
SD = sqrt(diag(C));
C = C./(SD*SD');
C(isnan(C)) = 0;
[U,V] = eig(C);
[V,ind] = sort(diag(V),1,'descend');
U = U(:,ind);
yuzde = (V/sum(V))*100;
V = V(cumsum(yuzde)<=energy);  % %95 enerji korundu
%V = V(1:8);
U = U(:,1:length(V));
W = U';
end

function [dogru,tfp] = asist(train, target, valtrain, valtarget, tex,multi)
if multi
    outputsvm = multisvm(train,valtrain,target, valtarget, tex, '-b 1');
    [~,outputsvm] = max(outputsvm,[],2);
    dogru = sum(outputsvm == valtarget);
    tfp = hesapet(outputsvm, valtarget, false);
else
    svmstruct = svmtrain(target,train,tex);
    [~,~,outputsvm] = svmpredict(valtarget,valtrain,svmstruct, '-b 1');
    outputsvm = outputsvm(:,svmstruct.Label);
    [~,outputsvm] = max(outputsvm,[],2);
    dogru = sum(outputsvm == valtarget);
    tfp = hesapet(outputsvm, valtarget, false);
end
end

function [TFP] = hesapet(output,testtarget,ismax)
if ismax
    [~,testtarget] = max(testtarget);
end
confmat = confusionmat(testtarget,output);
tp = diag(confmat)';
fp = sum(confmat);  fp = fp - tp;
fn = sum(confmat,2)'; fn = fn - tp;
TFP = [tp;fp;fn];
end

function [micro ,macro] = micmac(tfp)
tpsum = sum(tfp(1,:));
fi = tpsum/(sum(tfp(2,:)) + tpsum);
p = tpsum/(sum(tfp(3,:)) + tpsum);
micro = (2*fi*p)/(fi + p);

fi = tfp(1,:)./(sum(tfp(1:2,:)));
p = tfp(1,:)./(tfp(1,:)+tfp(3,:));
macro = (2*(fi.*p))./(fi + p);
macro = mean(macro);

end
