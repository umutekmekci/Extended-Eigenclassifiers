function [ output ] = prepareDataFusion(  )
clc;
dataset = 'australian';
[DD,~] = isDiscrete('C:\Users\daredavil\Documents\MATLAB\Fusion\data\datasets.ini', dataset);
[uzunluk,classNum,featureMap,targetMap] = readDatadef('C:\Users\daredavil\Documents\MATLAB\Fusion\data\',dataset);
[trainall,trainTarget, instTrain, kTrain] = forTrainandTest(dataset,'-trainall.txt' ,DD,targetMap,uzunluk,classNum,featureMap);
[testall,testTarget, instTest, kTest] = forTrainandTest(dataset,'-testall.txt' ,DD,targetMap,uzunluk,classNum,featureMap);
disp('train ve test bitti');
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

% classifier combination for train
trainCC = zeros(19,kTrain,instTrain);
dirpath = ['C:\\Users\\daredavil\\Documents\\MATLAB\\Fusion\\data\\',dataset,'\\5x2'];
for i = 1:19
    filepath = [dirpath, '\\', clnames{i}, '\\posterior\\',dataset,'-',clnames2{i},'-trainall.txt'];
    file = fopen(filepath);
    line = fgets(file);
    C = textscan(line(3:end),'%f');
    C = C{1};  instNum = C(1);
    for n = 1:instNum
        line = fgets(file);
        C = textscan(line,'%f');
        C = C{1};  C = C(1:end-1)';
        trainCC(i,:,n) = C;
    end
end
disp('train CC bitti');
fclose(file);
% classifier combination for test
testCC = zeros(19,kTest,instTest);
dirpath = ['C:\\Users\\daredavil\\Documents\\MATLAB\\Fusion\\data\\',dataset,'\\5x2'];
for i = 1:19
    filepath = [dirpath, '\\', clnames{i}, '\\posterior\\',dataset,'-',clnames2{i},'-testall.txt'];
    file = fopen(filepath);
    line = fgets(file);
    C = textscan(line(3:end),'%f');
    C = C{1};  instNum = C(1);
    for n = 1:instNum
        line = fgets(file);
        C = textscan(line,'%f');
        C = C{1};  C = C(1:end-1)';
        testCC(i,:,n) = C;
    end
end
disp('test CC bitti');
fclose(file);
% doðru çýkýþlar için sonuçlar
trainPCA = zeros(19,instTrain);
for i = 1:instTrain
    temp = trainCC(:,:,i);
    trainPCA(:,i) = temp(:,trainTarget(i));
end

trainTarget = full(ind2vec(trainTarget));
testTarget = full(ind2vec(testTarget));

name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfiles\',dataset];
matobj = matfile(name);
matobj.trainall = trainall;
matobj.testall = testall;
matobj.trainCC = trainCC;
matobj.testCC = testCC;
matobj.trainPCA = trainPCA;
matobj.trainTarget = trainTarget;
matobj.testTarget = testTarget;
clear matobj;
output = 1;
disp('bitti');
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
line = fgets(file);  % ilk satýrý oku burda instance sayýsý ve feature sayýsý var
C = textscan(line(3:end),'%f');
C = C{1};  instNum = C(1);  featNum = C(2)-1;

train = zeros(featNum, instNum);
target = zeros(1,instNum);
missing = [];
for i = 1:instNum   %satýr satýr oku
    line = fgets(file);
    C = textscan(line,'%s');  %missing value olabilir ? ile göstermiþler
    C = C{1};   % her bir deðer text halinde
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