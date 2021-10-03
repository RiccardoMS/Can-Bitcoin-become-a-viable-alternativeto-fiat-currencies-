function [dataCleaned] = cleanData(data, destroy)
%Receive as input a multidimensional vector, find NaN occurencies and
%delete conresponding rows. Store new structure leaving unchanged the old
%one. If "destroy" is true, deallocate the old structure

n=size(data,1);
m=size(data,2);

k=1;
allocated=false;
while(~allocated)
    if (~any(isnan(data(k,:))))
        dataCleaned=[data(k,:)];
        allocated=true;
    else
        k=k+1;
    end
end

    
for i = (k+1):n
    if (~any(isnan(data(i,:))))
        dataCleaned=[dataCleaned; data(i,:)];
    end
    
end

if destroy
    clear data;
end



