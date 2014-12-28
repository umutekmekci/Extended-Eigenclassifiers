function [ output ] = prepareEA( )
clc;
dataset = 'zoo';
clnames{1} = 'c45'; clnames{2} = 'gsn'; clnames{3} ='knn';
clnames{4} ='knn';  clnames{5} ='knn';  clnames{6} ='knn';
clnames{7} = 'ldt'; clnames{8} = 'lgs'; clnames{9} ='mdt';
clnames{10} = 'mlp'; clnames{11} = 'mlp'; clnames{12} ='mlp';
clnames{13} = 'mlp'; clnames{14} = 'mlp'; clnames{15} ='mlp';
clnames{16} = 'smx'; clnames{17} = 'svm'; clnames{18} ='svm';
clnames{19} ='svm';

clnames2{1} = 'c45'; clnames2{2} = 'gsn'; clnames2{3} ='1nn';
clnames2{4} ='3nn';  clnames2{5} ='5nn';  clnames2{6} ='7nn';
clnames2{7} = 'ldt'; clnames2{8} = 'lgs'; clnames2{9} ='mdt';
clnames2{10} = 'lnp'; clnames2{11} = 'ml1'; clnames2{12} ='ml2';
clnames2{13} = 'ml3'; clnames2{14} = 'ml4'; clnames2{15} ='ml5';
clnames2{16} = 'smx'; clnames2{17} = 'sv2'; clnames2{18} ='svl';
clnames2{19} ='svr';

which = '-train.txt';
[trainCC,targetCC,trainPCA] = tempPrepare(dataset,which,clnames,clnames2);
which = '-val-1.txt';
[trainCCVal1,targetCCVal1,trainPCAVal1] = tempPrepare(dataset,which,clnames, clnames2);
which = '-val-2.txt';
[trainCCVal2,targetCCVal2] = tempPrepare(dataset,which,clnames,clnames2);


for i = 1:10
    targetCC{i} = [targetCC{i},targetCCVal1{i}];
    trainPCA{i} = [trainPCA{i}, trainPCAVal1{i}];
    tt = trainCC{i};  tt2 = trainCCVal1{i};
    trainCC{i} = reshape([tt(:);tt2(:)],size(tt,1),size(tt,2),size(tt,3)+size(tt2,3));
end

name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEA\',dataset];
matobj = matfile(name);
matobj.trainCCVal1 = trainCCVal1;
matobj.trainPCAVal1 = trainPCAVal1;
matobj.targetCCVal1 = targetCCVal1;
matobj.trainCCVal2 = trainCCVal2;
matobj.targetCCVal2 = targetCCVal2;
matobj.trainCC = trainCC;
matobj.trainPCA = trainPCA;
matobj.targetCC = targetCC;
clear matobj;
output = 1;
end

function [trainCCVal1, targetCCVal1, trainPCAVal1] = tempPrepare(dataset,which,clnames,clnames2)
% classifier combination for train
iter = 1;
trainCCVal1 = {};
targetCCVal1 = {};
tempTrain = [];
targetTemp = [];
dirpath = ['C:\\Users\\daredavil\\Documents\\MATLAB\\Fusion\\data\\',dataset,'\\5x2'];
for k1 = 1:5
    for k2 = 1:2
        for i = 1:19
            %disp(i)
            filepath = [dirpath, '\\', clnames{i}, '\\posterior\\',dataset,'-',num2str(k1),'-',num2str(k2),'-',clnames2{i},which];
            file = fopen(filepath);
            line = fgets(file);
            C = textscan(line(3:end),'%f');
            C = C{1};  instNum = C(1);
            for n = 1:instNum
                line = fgets(file);
                C = textscan(line,'%f');
                C = C{1};
                if i == 19
                    targetTemp(n) = C(end);
                end
                C = C(1:end-1)';
                tempTrain(i,:,n) = C;
            end
            fclose(file);
        end
        trainCCVal1{iter} = tempTrain;
        if min(targetTemp) == 0
            targetTemp = targetTemp + 1;
        end
        targetCCVal1{iter} = targetTemp;
        tempTrain = [];
        targetTemp = [];
        iter = iter + 1;
    end
end
disp('val1 CC bitti');


if nargout == 3
    trainPCAVal1 = {};
    for iter = 1:10
        tt = trainCCVal1{iter};
        trainTarget = targetCCVal1{iter};
        temp = zeros(19,length(trainTarget));
        for i = 1:length(trainTarget)
            temp2 = tt(:,:,i);
            temp(:,i) = temp2(:,trainTarget(i));
        end
        trainPCAVal1{iter} = temp;
    end
end







end

