function [c,dogru] = SVMoptimalLin(train, target, valtrain, valtarget,c,multi)

tex = sprintf('-t 0 -c %d -b 1 -q',c);
dogru = asist(train, target, valtrain, valtarget, tex,multi);
ctemp = c;
dogrutemp = dogru;

    sol = true;
    atanacak = c;
    iter = 1;
    while sol
        
        csol = c*0.5;
        if csol < 2^(-6)
            csol = c;
            break;
        end
        fprintf('csol:%f',csol);
        tex = sprintf('-t 0 -c %f -b 1 -q',csol);
        dogrusol = asist(train, target, valtrain, valtarget, tex,multi);
        if dogrusol < dogru
            sol = false;
            csol = atanacak;
        else if dogrusol > dogru
                c = csol;
                atanacak = c;
                dogru = dogrusol;
            else
                iter = iter + 1;
                if iter == 8
                    csol = atanacak;
                    break;
                end
                c = csol;
            end
        end
    end
    dogrusol = dogru;
    sag = true;
    c = ctemp;
    dogru = dogrutemp;
    atanacak = c;
    iter = 1;
    while sag
        
        csag = c*2;
        if csag > 2^5
            csag = c;
            break;
        end
        fprintf('csag:%d',csag);
        tex = sprintf('-t 0 -c %d -b 1 -q',csag);
        dogrusag = asist(train, target, valtrain, valtarget, tex,multi);
        if dogrusag < dogru
            sag = false;
            csag = atanacak;
        else if dogrusag > dogru
                c = csag;
                atanacak = c;
                dogru = dogrusag;
            else
                iter = iter + 1;
                if iter == 8
                    csag = atanacak;
                    break;
                end
                c = csag;
            end
        end
    end
    
dogrusag = dogru;
[dogru,ind] = sort([dogrutemp, dogrusol, dogrusag],'descend');
dogru = dogru(1);
cvec = [ctemp,csol,csag];
cvec = cvec(ind);
c = cvec(1);

end

function dogru = asist(train, target, valtrain, valtarget, tex,multi)
if multi
    outputsvm = multisvm(train,valtrain,target, valtarget, tex, '-b 1');
    [~,outputsvm] = max(outputsvm,[],2);
    dogru = sum(outputsvm == valtarget);
else
    svmstruct = svmtrain(target,train,tex);
    [~,~,outputsvm] = svmpredict(valtarget,valtrain,svmstruct, '-b 1');
    [~,outputsvm] = max(outputsvm,[],2);
    dogru = sum(outputsvm == valtarget);
end
end

