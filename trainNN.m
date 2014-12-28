function [ weights,bias,error,errorVec, valerror ] = trainNN( input,target,weights,bias,funcList,arc,EC,maxIter,minError,isnorm, valData, valTarget, varargin)

if isempty(varargin)
    valsize = 20;
else
    valsize = varargin{1};
end
weightBag = cell(1,valsize);
biasBag = cell(1,valsize);

if isempty(weights) && isempty(bias)
    [weights,bias] = initWeights(arc);
end

errorVec = zeros(1,maxIter);
lr = 1;
iter = 1;
error = minError + 1;
devam = true;
valerror = zeros(1,valsize);
[~,valerror(iter) ] = FNN( valData, weights,bias, funcList, valTarget, EC);
weightBag{1} = weights;
biasBag{1} = bias;
while iter <= maxIter && error > minError && devam
    %[~,valerror(iter) ] = FNN( valData, weights,bias, funcList, valTarget, EC);
    [ dw, db, ~ ] = BNN( input, weights,bias, funcList, target, EC, isnorm);
    [lr, error] = fminsearch(@(lr)func(input, weights,bias,dw,db, funcList, target, EC, lr),lr);
    for i = 1:length(weights)
        weights{i} = weights{i} - lr*dw{i};
        bias{i} = bias{i} - lr*db{i}; 
    end
    errorVec(iter) = error;
    fprintf('iter-->%d,  error = %f\n',iter, error);
    iter = iter + 1;
    if iter > valsize
        devam = testValError(valerror);
        valerror(1:end-1) = valerror(2:end);
        [~,valerror(end) ] = FNN( valData, weights,bias, funcList, valTarget, EC);
        weightBag(1:end-1) = weightBag(2:end);
        weightBag{end} = weights;
        biasBag(1:end-1) = biasBag(2:end);
        biasBag{end} = bias;
        if ~devam
            [valerror,ind] = min(valerror);
            weights = weightBag{ind};
            bias = biasBag{ind};
        end
    else
        [~,valerror(iter) ] = FNN( valData, weights,bias, funcList, valTarget, EC);
        weightBag{iter} = weights;
        biasBag{iter} = bias;
    end
    
end
valerror = valerror(end);    
end

function devam = testValError(valerror)
ce = round(length(valerror)/2);
devam = (mean(valerror(ce+1:end)) <= mean(valerror(1:ce)));
end

function error = func(input, weights,bias,dw,db, funcList, target, EC, lr)
for i = 1:length(weights)
    weights{i} = weights{i} - lr*dw{i};
    bias{i} = bias{i} - lr*db{i}; 
end
[~,error] = FNN(input, weights,bias, funcList, target, EC);
end

function [weights,bias] = initWeights(arc)
len = length(arc)-1;
weights = cell(1,len);
bias = cell(1,len);
for i = 1:len
    weights{i} = -0.1 + 0.2*rand(arc(i+1),arc(i));
    bias{i} = -0.1 + 0.2*rand(arc(i+1),1);
end
end