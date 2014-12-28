function [ output ] = prepareDataFusionEA(  )
clc;
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
for dats = 1:38
    dataset = datasets{dats};
    disp(dataset);
    [DD,~] = isDiscrete('C:\Users\daredavil\Documents\MATLAB\Fusion\data\datasets.ini', dataset);
    [uzunluk,classNum,featureMap,targetMap] = readDatadef('C:\Users\daredavil\Documents\MATLAB\Fusion\data\',dataset);

    trainCell = cell(1,10);
    trainTargetCell = cell(1,10);
    valCell = cell(1,10);
    valTargetCell = cell(1,10);
    fold = 1;
    for i = 1:5
        for j = 1:2
            disp(fold);
            tex = sprintf('-%d-%d-train.txt',i,j);
            [train,trainTarget, instTrain, kTrain] = forTrainandTest(dataset,tex ,DD,targetMap,uzunluk,classNum,featureMap);
            tex = sprintf('-%d-%d-val-1.txt',i,j);
            [val1,val1Target, instval1, kval1] = forTrainandTest(dataset,tex ,DD,targetMap,uzunluk,classNum,featureMap);
            tex = sprintf('-%d-%d-val-2.txt',i,j);
            [val2,val2Target, instval2, kval2] = forTrainandTest(dataset,tex ,DD,targetMap,uzunluk,classNum,featureMap);
            train = [train val1];
            trainTarget = [trainTarget val1Target];
            trainCell{fold} = train;
            trainTargetCell{fold} = trainTarget;
            valCell{fold} = val2;
            valTargetCell{fold} = val2Target;
            fold = fold + 1;
        end
    end
    [testall,testTarget, instTest, kTest] = forTrainandTest(dataset,'-testall.txt' ,DD,targetMap,uzunluk,classNum,featureMap);
    disp('train ve test bitti');



%trainTarget = full(ind2vec(trainTarget));
%testTarget = full(ind2vec(testTarget));

    name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEA\',dataset,'raw'];
    matobj = matfile(name);
    matobj.train = trainCell;
    matobj.testall = testall;
    matobj.trainTarget = trainTargetCell;
    matobj.testTarget = testTarget;
    matobj.val = valCell;
    matobj.valTarget = valTargetCell;
    clear matobj;
    output = 1;
    disp('bitti');
end
end

function [uzunluk,classNum,featureMap,targetMap] = readDatadef(name,dataset)
name = [name, dataset,'\data.def'];
file = fopen(name,'r');
line = fgets(file);
uzunluk = 0;
classNum = [];
featureMap = {};
iter = 1;
while ischar(line)
    if any(line == ';')
        uzunluk = uzunluk + 1;
        if strcmp(line(end-1),';')
            C = textscan(line(1:end-2),'%s','Delimiter',';');
            C = C{1};
            featureMap{iter} = containers.Map(C, 1:length(C));
            iter = iter + 1;
            classNum = [classNum,length(C)];
        end
    end
    line = fgets(file);
end
uzunluk = uzunluk-1;
classNum = classNum(1:end-1);
targetMap = featureMap{end};
featureMap = featureMap(1:end-1);
fclose(file);
end


function [D,targetMap] = isDiscrete(path, dataset)
file = fopen(path,'r');
line = fgets(file);
while ischar(line)
    if strcmp(line(2:end-2),dataset)
        line = fgets(file);
        line = fgets(file); line = line(7:end-1);
        targetMap = textscan(line,'%s','Delimiter',';');
        targetMap = lower(targetMap{1});
        targetMap = containers.Map(targetMap,1:length(targetMap));
        line = fgets(file);
        line = fgets(file);
        D = [];
        while strcmp(line(1:7),'Feature')
            C = textscan(line,'%s','Delimiter','=');
            C = C{1};
            if strcmp(C{2}, 'String')
                D = [D,true];
            else
                D = [D, false];
            end
            line = fgets(file);
        end
        D = D(1:end-1);
        break;
    end
    line = fgets(file);
end
fclose(file);
end

function [data,target,instNum, classNumTarget] = forTrainandTest(dataset,trainortest ,DD,targetMap,uzunluk,classNum,featureMap)
% reads train and test text files and saves them in .mat format
%eg. dataset = 'pageblock'

name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\data\', dataset, '\5x2\', dataset, trainortest];
file = fopen(name,'r');
line = fgets(file);  % ilk satırı oku burda instance sayısı ve feature sayısı var
C = textscan(line(3:end),'%f');
C = C{1};  instNum = C(1);  featNum = C(2)-1;

train = zeros(featNum, instNum);
target = zeros(1,instNum);
missing = [];
for i = 1:instNum   %satır satır oku
    line = fgets(file);
    C = textscan(line,'%s');  %missing value olabilir ? ile göstermişler
    C = C{1};   % her bir değer text halinde
    if isKey(targetMap,C{end})
        target(i) = targetMap(C{end});
    else
        error('Hata in targetMap');
    end
    iter = 1;
    for k = 1:length(C)-1
        switch C{k}
            case '?',
                if DD(k)
                    iter = iter + 1;
                end
                missing = [missing, [k;i]];
            otherwise,
                if ~DD(k)
                    train(k,i) = str2num(C{k});
                else
                    ffmap = featureMap{iter};  iter = iter + 1;
                    if isKey(ffmap,C{k})
                        train(k,i) = ffmap(C{k});
                    else
                        aaa = 1;
                    end
                end
        end
    end
end
if min(target) == 0
    target = target + 1;
end
classNumTarget = length(targetMap);
if ~isempty(missing)
    mm = mean(train(missing(1,:),:),2);
    for i = 1:length(mm)
        train(missing(1,i),missing(2,i)) = mm(i);
    end
end
fclose(file);

DD = logical(DD);
CC = convertDiscrete(train(DD,:),uzunluk,classNum,featureMap);  %convert discrete numbers to 1-n representation
data = [];
k=1;
for i = 1:featNum
    if DD(i)
        data = [data;CC{k}];
        k = k + 1;
    else
        data = [data; train(i,:)];
    end
end

end

function DD = convertDiscrete(data,uzunluk,classNum,featureMap)
if isempty(data)
    DD = 0;
    return;
end
if size(data,1) ~= length(classNum)
    error('hata in convertDiscrete');
end
DD = cell(1,size(data,1));
for i = 1:size(data,1)
    target = data(i,:);
    FF = featureMap{i};
    temp = ind2target(target,FF);
    DD{i} = temp;
end
end

function target2 = ind2target(target,ff)
ff = cell2mat(values(ff));
ff = sort(ff);
target2 = zeros(size(ff,1),size(target,2));
for i = 1:length(ff)
    ind = target == ff(i);
    target2(i,ind) = 1;
end
end
