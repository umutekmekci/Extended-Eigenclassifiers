function [ output,error ] = FNNDropout( input, weights,bias, funcList, dropProb, target, EC )

len = length(weights);
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
input = input.*(1-dropProb(1));
for i=1:len-1
    netinput = weights{i}*input + bias{i}*multiplier;
    f = func{i};
    input = f(netinput);
    input = input.*(1-dropProb(i+1));
end

netinput = weights{end}*input + bias{end}*multiplier;
f = func{end};
output = f(netinput);
output = output.*(1-dropProb(end));

if nargout == 2
    error = errorFunc(output,target,EC,funcList{end});
end

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

