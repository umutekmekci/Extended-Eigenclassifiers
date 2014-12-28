function [ weights, bias,weightsh,biash, error, valerror ] = onlineLearningVal( input,target,weights,bias,funcList,arc, EC,isnorm,isDropout,epoch, dropProb,mbs,nn,lrbas,mmbas,wd, valData, valTarget )
%mbs: mini batch size
%nn: maximum norm
weightsh = [];
biash = [];
error = [];
if isempty(weights) && isempty(bias)
    [weights,bias] = initWeights(arc);
end
dwbefore = cell(1,length(weights));
dbbefore = cell(1,length(weights));
for i = 1:length(weights)
    dwbefore{i} = 0;
    dbbefore{i} = 0;
end
inputTemp = input;
targetTemp = target;


%epsilon = 10;  %initial learning rate
%ff = 0.998;  % decay rate
%pi = 0.5;  %initial momentum
%pf = 0.99; %last momentum
%TT = floor((epoch*9)/10);  %cutoff time for momentum
%mmbas = 0.7;
son = 0.99;
moment = (son-mmbas)/epoch;
cutpoint = round(size(input,2)/mbs);
[input, target] = shufleInput(inputTemp, targetTemp, cutpoint);
devam = true;
valsize = 80;
valerror = zeros(1,valsize);
%valerrorfull = zeros(1,epoch);
%trainerrorfull = zeros(1,epoch);
[~,valerror(1) ] = FNNDropout(valData, weights,bias, funcList,dropProb, valTarget, EC);
%valerrorfull(1) = valerror(1);
valiter = 2;
weightBag = cell(1,valsize);
biasBag = cell(1,valsize);

for e = 0:epoch-1
    %[~,valerrorfull(e+1) ] = FNNDropout(valData, weights,bias, funcList,dropProb, valTarget, EC);
    if ~devam
        [valerror,ind] = min(valerror);
        weights = weightBag{ind};
        bias = biasBag{ind};
        break;
    end
    indis = mod(e,cutpoint)+1;
    [ dw, db ] = BNNDropout( input{indis}, weights,bias, funcList, target{indis}, EC, isnorm,dropProb);
    lr = (e./epoch)*(0.001-lrbas) + lrbas;
    momentum = moment*e + mmbas;
    %if e > TT
    %    pi = pf;
    %else
    %    pi = (e/TT)*pf + (1-(e/TT))*pi;
    %end
    %wd = 0.0005;
    %wd = 0;
    for i = 1:length(dw)
        dw{i} = dwbefore{i}*momentum - lr*dw{i} - lr*wd*weights{i};
        db{i} = dbbefore{i}*momentum - lr*db{i} - lr*wd*bias{i};
    end
    dwbefore = dw;
    dbbefore = db;
    for i = 1:length(dw)
        weights{i} = weights{i} + dw{i};
        bias{i} = bias{i} + db{i};
    end
    
    
    for i = 1:length(weights)
        temp = [weights{i}, bias{i}];
        dd = diag(temp*temp');
        ind = dd > nn;
        if any(ind)
            tt = repmat(sqrt(dd(ind)./nn),1,size(temp,2));
            temp(ind,:) = temp(ind,:)./(tt);
            weights{i} = temp(:,1:end-1);
            bias{i} = temp(:,end);
        end
    end
    
    if indis == cutpoint;
        if valiter > valsize
            devam = testValError(valerror);
            valerror(1:end-1) = valerror(2:end);
            [~,valerror(end) ] = FNNDropout(valData, weights,bias, funcList,dropProb, valTarget, EC);
            weightBag(1:end-1) = weightBag(2:end);
            weightBag{end} = weights;
            biasBag(1:end-1) = biasBag(2:end);
            biasBag{end} = bias;
        else
            [~,valerror(valiter) ] = FNNDropout(valData, weights,bias, funcList,dropProb, valTarget, EC);
            weightBag{valiter} = weights;
            biasBag{valiter} = bias;
        end
       %[~,valerrorfull(valiter) ] = FNNDropout(valData, weights,bias, funcList,dropProb, valTarget, EC);
        
        %[www,bbb] = tohalf(weights, bias);
        %[~,error] = FNNDropout( inputTemp, weights,bias, funcList, [0 0],targetTemp, EC );
        [~,error] = FNNDropout( inputTemp, weights,bias, funcList,dropProb,targetTemp, EC );
        %trainerrorfull(valiter) = error; 
        fprintf('epoch-->%d, lr=%f, error = %f\n',e,lr, error);
        [input, target] = shufleInput(inputTemp, targetTemp, cutpoint);
        valiter = valiter + 1;
    end
    
end
valerror = valerror(end);

if isDropout
    [weightsh,biash] = tohalf(weights,bias);
end
end

function devam = testValError(valerror)
ce = round(length(valerror)/2);
devam = (mean(valerror(ce+1:end)) <= mean(valerror(1:ce)));
end

function [weights,bias] =  tohalf(weights, bias)
for i = 1:length(weights)-1
    weights{i} = weights{i}*1;
    bias{i} = bias{i}*1;
end
bas = i+1;
for i = bas:length(weights)
    weights{i} = weights{i}*0.8;
    bias{i} = bias{i}*1;
end
end


function [input, target] = shufleInput(in,ta,c)
ind = randperm(size(ta,2));
in = in(:,ind);  ta = ta(:,ind);
ind = round(linspace(1,size(ta,2),c+1));
input = cell(1,length(ind)-1);
target = cell(1,length(ind)-1);

for i = 1:length(ind)-1
    input{i} = in(:,ind(i):ind(i+1));
    target{i} = ta(:,ind(i):ind(i+1));
end
end


function [weights,bias] = initWeights(arc,WW,BB,k)
len = length(arc)-1;
weights = cell(1,len);
bias = cell(1,len);
for i = 1:len
    weights{i} = -0.1 + 0.2*rand(arc(i+1),arc(i));
    bias{i} = -0.1 + 0.2*rand(arc(i+1),1);
end
if nargin > 1 && k>0
    for i = 1:len
        weights{i} = (weights{i} + WW{i}/k)*0.5;
        bias{i} = (bias{i} + BB{i}/k)*0.5;
    end
end
end

