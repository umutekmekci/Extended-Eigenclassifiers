function [ DogruTest ] = EigenClassifiersMulti3Modal(dataset)
%clc
%rng(150);
dataset = 'australian';
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

name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfiles\',dataset];
matobj = matfile(name);
testCC = matobj.testCC;
testTarget = matobj.testTarget;
clear matobj;

%name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEASVD\',dataset,'multipcamapping'];
%matobj = matfile(name,'Writable',true);
%mapping = matobj.mapping;
%clear matobj;


name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEASVD\',dataset,'multipcamappingmodal2'];
matobj = matfile(name,'Writable',true);
U_map = matobj.U;
mean_K = matobj.mean_K;
clear matobj;

%maxsize = ayarla2(mappingCell);

weights = cell(1,10);
bias = cell(1,10);
DogruTest = 0;
DogruTrain = 0;
DogruAvarage = 0;
DogruAvarageTest = 0;

for cl = 1: 10 
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
    
    %dimsize = 8*size(trainTarget,1);
    %dimsize = size(trainCC,1);
    %dimsize = maxsize;
    dimsize = 6;
    %dimsize = size(mapping.M,2);
    %dimsize = length(mapping.lambda);
    trainPCA = zeros(dimsize,size(trainCC,3));
    avaragePreTrain = zeros(1,size(trainCC,3));
    TT = vec2ind(trainTarget);
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
        [~,avaragePreTrain(i)] = max(mean(temp));
        [U,S,~] = svd(temp);
        S = diag(S);  [S,ind] = sort(S, 'descend');  U = U(:,ind);
        K = U(:,1)*S(1) - mean_K;
        trainPCA(:,i) = U_map(:,1:dimsize)'*K;
    end
    vartraindata(cl) = sum(sum(abs(cov(trainPCA'))));
    KK = max(TT);
    for i = 1:KK
        MM(:,i) = mean(trainPCA(:,TT == i),2);
    end
    
    
    testData = zeros(dimsize,size(testCC,3));
    avaragePreTest = zeros(1,size(testCC,3));
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
        [~,avaragePreTest(i)] = max(mean(temp));
        [U,S,~] = svd(temp);
        S = diag(S);  [S,ind] = sort(S, 'descend'); U = U(:,ind);
        K = U(:,1)*S(1) - mean_K;
        testData(:,i) = U_map(:,1:dimsize)'*K;
    end
    for i = 1:KK
        EE(i,:) = sum((repmat(MM(:,i),1,size(testData,2)) - testData).^2);
    end
    [~,ind] = min(EE);
    sum(ind == vec2ind(testTarget))
    vartestdata(cl) = sum(sum(abs(cov(testData'))));
    
    valData = zeros(dimsize,size(valset,3));
    for i = 1:size(valset,3)
        temp = valset(:,:,i);
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
        valData(:,i) = U_map(:,1:dimsize)'*K;
    end

    % prepare linear classifier
    funcList{1} = 'purelin';
    arc = [size(trainPCA,1),size(trainTarget,1)];
    EC = 'mse';
    maxIter = 500; minError = 1e-5;
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
    ddtestAv(cl) = sum(avaragePreTest == vec2ind(testTarget));
    DogruAvarageTest = DogruAvarageTest + ddtestAv(cl);
    fprintf('Toplam = %d, avarage = %d,  Dogru = %d\n',size(output,2), sum(avaragePreTest == vec2ind(testTarget)), sum(output == vec2ind(testTarget)));
    [TFP(:,:,cl)] = hesapet(output, testTarget, true);
end
DogruTest = round(DogruTest/10);
DogruTest = (DogruTest/size(testData,2))*100;
%DogruTrain = round(DogruTrain/10)
%DogruAvarage = round(DogruAvarage/10)
%DogruAvarageTest = round(DogruAvarageTest/10)
%[~,ind] = min(valerror);
%ddtest(ind)
%ddtrain(ind);
%[micro, macro] = micmac(mean(TFP,3))
sum(vartraindata)
sum(vartestdata)
output = 1;


end

function la = ayarla2(mapping)
len = length(mapping);
lalist=zeros(1,len);
for i = 1:len
    if ~isempty(mapping{i})
        lalist(i) = size(mapping{i}.lambda,1);
    end
end
la = max(lalist);
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
