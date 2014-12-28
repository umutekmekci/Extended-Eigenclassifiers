function [ dw, db, error, output ] = BNNDropout( input, weights,bias, funcList, target, EC, isnorm,dropProb)


len = length(weights);
y = cell(1,len);
dy = cell(1,len);
multiplier = ones(1,size(input,2));

func = cell(1,len);
for i=1:len
    switch funcList{i}
        case 'logsig'
            func{i} = @logsig;
        case 'tansig'
            func{i} = @tansig;
        case 'purelin'
            func{i} = @purelin;
        case 'softmax'
            func{i} = @softmax;
    end
end

sat = size(input,1);
input((rand(1,sat) > (1-dropProb(1))),:) = 0;  %dropout on input
tinput = input;

for i=1:len-1
    netinput = weights{i}*input + bias{i}*multiplier;
    f = func{i};
    tt = f(netinput);
    sat = size(tt,1);
    tt((rand(1,sat) > (1- dropProb(i+1))),:) = 0;
    y{i} = tt;
    switch funcList{i}
        case 'logsig'
            dy{i} = tt.*(1-tt);
        case 'tansig'
            dy{i} = 1-(tt.^2);
        case 'purelin'
            dy{i} = ones(size(tt));
    end
    input = y{i};
end

netinput = weights{end}*input + bias{end}*multiplier;
f = func{end};
y{end} = f(netinput);
dy{end} = ones(size(netinput));

output = y{end};

if nargout > 2
    error = errorFunc(output,target,EC,funcList{end});
end

input = output-target;
switch EC
    case 'mse'
        switch funcList{end}
            case 'logsig'
                input = input.*(output.*(1-output));
            case 'tansig'
                input = input.*(1-(output.*output));
        end
end
p = cell(1,len);
dw = cell(1,len);
db = cell(1,len);

for i=len:-1:2
    p{i} = input.*dy{i};
    tempY = y{i-1};
    dw{i} = p{i}*tempY';
    db{i} = sum(p{i},2);
    tempLW = weights{i};
    input = tempLW'*p{i};
end
p{1} = input.*dy{1};
dw{1} = p{1}*tinput';
db{1} = sum(p{1},2);

if isnorm
    for i=1:len
        ndw = norm(dw{i});
        if ndw ~= 0
            dw{i} = dw{i}./ndw;
        end
        ndb = norm(db{i});
        if ndb ~= 0
            db{i} = db{i}./ndb;
        end
    end
end


%if isnorm
%    for i=1:len
%        temp = [dw{i}, db{i}];
%        dd = diag(temp*temp');
%        ind = dd == 0;
%        nd = repmat(dd,1,size(temp,2));
%        temp(~ind) = temp(~ind)./nd(~ind);
%        dw{i} = temp(:,1:end-1);
%        db{i} = temp(:,end);
%    end
%end

end

function error = errorFunc(output,target,EC,F)
error = 0;
switch EC
    case 'mse'
        error = 0.5*mse(output-target);
    case 'cross-entropy'
        switch F
            case 'logsig'
                output2 = 1-output;
                for i = 1:size(target,1)
                    ind = target(i,:) == 1;
                    error = error + sum(log2(output(i,ind))) + sum(log2(output2(i,~ind)));
                end
            error = -error;
            case 'softmax'
                for i = 1:size(target,1)
                    ind = target(i,:) == 1;
                    error = error + sum(log2(output(i,ind)));
                end
            error = -error;
        end
end
end

