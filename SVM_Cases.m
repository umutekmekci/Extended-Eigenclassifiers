function [ output ] = SVM_Cases
clc
%rng(150);
dataset = 'pageblock';
name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEA\',dataset];
matobj = matfile(name);
trainPCATemp = matobj.trainPCA;
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

name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfiles\',dataset];
matobj = matfile(name);
testCC = matobj.testCC;
testTarget = matobj.testTarget;
clear matobj;


DogruTest = 0;
DogruTrain = 0;
c=1;
g=1;
for cl = 1: 10 
    trainCC = trainCCTemp{cl};
    trainPCA = trainPCATemp{cl};
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
    
    W = applyPCA(trainPCA,95);
    dimsize = size(W,1);
    trainPCA = zeros(dimsize*size(trainCC,2),size(trainCC,3));
    avaragePreTrain = zeros(1,size(trainCC,3));
    SVMtrain = zeros(size(trainCC,1)*size(trainCC,2),size(trainCC,3));
    for i = 1:size(trainCC,3)
        temp = trainCC(:,:,i);
        SVMtrain(:,i) = temp(:);
        [~,avaragePreTrain(i)] = max(mean(temp));
        temp = W*temp;
        trainPCA(:,i) = temp(:);
    end
    testData = zeros(dimsize*size(testCC,2),size(testCC,3));
    avaragePreTest = zeros(1,size(testCC,3));
    SVMtest = zeros(size(testCC,1)*size(testCC,2),size(testCC,3));
    for i = 1:size(testCC,3)
        temp = testCC(:,:,i);
        SVMtest(:,i) = temp(:);
        [~,avaragePreTest(i)] = max(mean(temp));
        temp = W*temp;
        testData(:,i) = temp(:);
    end
    valData = zeros(dimsize*size(valset,2),size(valset,3));
    SVMval = zeros(size(valset,1)*size(valset,2),size(valset,3));
    for i = 1:size(valset,3)
        temp = valset(:,:,i);
        SVMval(:,i) = temp(:);
        temp = W*temp;
        valData(:,i) = temp(:);
    end
    

    if size(trainTarget,1) == 2
        multi = false;
    else
        multi = true;
    end
 
    %[c,dogruval] = SVMoptimalLin(SVMtrain', vec2ind(trainTarget)', SVMval', vec2ind(valsetTarget)',c, multi);
    %[c,g,dogruval] = SVMoptimal(SVMtrain', vec2ind(trainTarget)', SVMval', vec2ind(valsetTarget)',c,g, multi);
    [c,dogruval] = SVMoptimalLin(trainPCA', vec2ind(trainTarget)', valData', vec2ind(valsetTarget)',c, multi);
    %[c,g,dogruval] = SVMoptimal(trainPCA', vec2ind(trainTarget)', valData', vec2ind(valsetTarget)',c,g, multi);
    dogruvec(cl) = dogruval;
    cvec(cl) = c;
        
    tex = sprintf('-t 0 -c %f -b 1 -q',c);
    %tex = sprintf('-c %f -g %f -b 1 -q',c,g);
    %dogrutrain(cl) = asist(SVMtrain', vec2ind(trainTarget)', SVMtrain',vec2ind(trainTarget)' , tex,multi);
    dogrutrain(cl) = asist(trainPCA', vec2ind(trainTarget)', trainPCA',vec2ind(trainTarget)' , tex,multi);
    DogruTrain = DogruTrain + dogrutrain(cl);

    %[dogrutest(cl),TFP(:,:,cl)] = asist(SVMtrain', vec2ind(trainTarget)', SVMtest',vec2ind(testTarget)' , tex,multi);
    [dogrutest(cl), TFP(:,:,cl)] = asist(trainPCA', vec2ind(trainTarget)', testData',vec2ind(testTarget)' , tex,multi);
    DogruTest = DogruTest + dogrutest(cl);
end
DogruTest = round(DogruTest/10)
DogruTrain = round(DogruTrain/10)
[~,ind] = max(dogruvec);
dogrutest(ind)
dogrutrain(ind)
[micro, macro] = micmac(mean(TFP,3))
output=1;

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