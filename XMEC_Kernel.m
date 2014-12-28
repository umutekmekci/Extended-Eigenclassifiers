function [ output ] = XMEC_Kernel()

%{
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
%}

datasets{1} = 'pageblock'; datasets{2} = 'pendigits'; datasets{3} = 'ringnorm';
datasets{4} = 'spambase'; datasets{5} = 'twonorm';
for i = 1:length(datasets)
    dataset = datasets{i};
    output = save_kernel(dataset);
    fprintf('%d..%s bitti\n',i,dataset);
end

end

function output = save_kernel(dataset)

output = 1;
%{
if strcmp(dataset,'mushroom')
    return;
elseif strcmp(dataset,'nursery')
    return;
elseif strcmp(dataset, 'pageblock')
    return;
elseif strcmp(dataset, 'pendigits')
    return;
elseif strcmp(dataset,'ringnorm')
    return;
elseif strcmp(dataset, 'spambase')
    return;
elseif strcmp(dataset, 'twonorm')
    return;        
end
%}

name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfiles\',dataset];
matobj = matfile(name);
trainCC = matobj.trainCC;
trainTarget = matobj.trainTarget;
clear matobj;

name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEA\',dataset,'kernelmapping'];
matobj = matfile(name);
mapping = matobj.mapping;
clear matobj;
dim_size_before = size(mapping.V,2);
param1 = mapping.param1;
param2 = mapping.param2;
kernel_type = mapping.kernel;
clear mapping;

TT = vec2ind(trainTarget);
K = zeros(size(trainCC,1)*size(trainCC,2),size(trainCC,3));
for ii = 1:size(trainCC,3)
    temp = trainCC(:,:,ii);
    Temp = zeros(size(temp,1)*size(temp,2),size(temp,2));
    satbas=1; aralik = size(temp,1); satson = aralik;
    for j = 1:size(temp,2)
        Temp(satbas:satson,j) = temp(:,j);
        satbas = satson + 1;
        satson = satson + aralik;
    end
    temp = Temp;
    [U,S,~] = svd(temp);
    S = diag(S);  [S,ind] = sort(S,'descend');  U = U(:,ind);
    K(:,ii) = U(:,1)*S(1); %+ U(:,2)*S(2);
end
mean_K = mean(K,2);
K = K - repmat(mean_K,1,size(K,2));
%KT = 0;
for ii = 1:max(TT)
    ind = ii == TT;
    w = sum(ind)/length(ind);
    %KT = KT + w*w*K(:,ind)*K(:,ind)';
    K(:,ind) = K(:,ind);
end
%K = KT./size(trainCC,3);
%dim_size_before = 20;
param1 = 10;
%param2 = 3;
[~, mapping] = kernel_pca(K',dim_size_before, kernel_type, param1, param2);
mapping.name = 'KernelPCA';
%[U,~] = eig(K);  % change to kernel_pca
name = ['C:\Users\daredavil\Documents\MATLAB\Fusion\matfilesEASVD\',dataset,'multikernelmappingmodal2'];
matobj = matfile(name,'Writable',true);
matobj.mapping = mapping;
matobj.mean_K = mean_K;
clear matobj;

output = 1;
end

