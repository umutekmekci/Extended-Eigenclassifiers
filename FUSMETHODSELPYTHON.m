function [ output ] = FUSMETHODSELPYTHON( )

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

%% lambdas
lambda1_data = zeros(19,length(datasets));
lambda2_data = zeros(19,length(datasets));
for i = 1:length(datasets)
    dataset = datasets{i};
    name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfiles\',dataset];
    matobj = matfile(name);
    trainPCA = matobj.trainPCA;
    [~, ~, lambda] = pca(trainPCA', 19);
    lambda1 = (lambda./(sum(lambda))).*100;
    lambda2 = (lambda./(sum(lambda)));
    lambda1_data(:,i) = lambda1;
    lambda2_data(:,i) = lambda2;
    clear matobj;
end

%% divergence
div_data = zeros(5,length(datasets));
for i = 1:length(datasets)
    dataset = datasets{i};
    name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfiles\',dataset];
    matobj = matfile(name);
    trainPCA = matobj.trainPCA;
    clear matobj;
    [N11, N00, N10, N01,validIndex, ent] = numberStatistics(trainPCA);
    div_data(1,i) = ent;
    c1 = N11.*N00;  c2 = N10.*N01;
    Qstatistic = (c1 - c2)./(c1+c2);
    Qstatistic(isnan(Qstatistic)) = 0;
    Qstatistic = Qstatistic(validIndex);
    Qstatistic = sum(Qstatistic)/sum(sum(validIndex));
    div_data(2,i) = Qstatistic;
    pcoef = (c1-c2)./sqrt((N11 + N10).*(N01 + N00).*(N11 + N01).*(N10 + N00));
    pcoef(isnan(pcoef)) = 0;
    pcoef = pcoef(validIndex);
    pcoef = sum(pcoef)/sum(sum(validIndex));
    div_data(3,i) = pcoef;
    dis = (N01 + N10)./(size(trainPCA,2));
    dis = dis(validIndex);
    dis = sum(dis)/sum(sum(validIndex));
    div_data(4,i) = dis;
    df = N00/(size(trainPCA,2));
    df = df(validIndex);
    df = sum(df)/sum(sum(validIndex));
    div_data(5,i) = df;
end

X1 = [lambda1_data;div_data];
X2 = [lambda2_data;div_data];

dd = [X1(1,:);X1(2,:)];
scatter(dd(1,:),dd(2,:),'filled');

name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfiles\FUSSELECTPYTHON'];
matobj = matfile(name);
matobj.X1 = X1;
matobj.X2 = X2;
clear matobj;

output = 1;
end


function [N11, N00, N10, N01,validIndex, Ent] = numberStatistics(matrix)
N11 = zeros(size(matrix,1));
N00 = zeros(size(matrix,1));
N10 = zeros(size(matrix,1));
N01 = zeros(size(matrix,1));
validIndex = false(size(matrix,1));
res = false(size(matrix));
res(matrix > 0.5) = true;
for i = 1:size(res,1)
    pair1 = res(i,:);
    for j = i+1:size(res,1)
        pair2 = res(j,:);
        n11 = sum(pair1 & pair2);  %ikiside dogru
        n00 = sum(~(pair1 | pair2));  %ikiside yanliş
        n10 = sum(pair1 & ~pair2);
        n01 = sum(~pair1 & pair2);
        N11(i,j) = n11;
        N00(i,j) = n00;
        N10(i,j) = n10;
        N01(i,j) = n01;
        validIndex(i,j) = true;
    end
end
res = sum(res);
L = size(matrix,1);
temp = res;
temp2 = L-res;
ind = temp2 < res;
temp(ind) = temp2(ind);
Ent = sum(temp)/(size(matrix,2)*ceil(L-L/2));
end

