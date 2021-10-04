function [log_data] = logreturns(data, col_index, destroy)
n= size(data,1);
m= size(data,1);
k= length(col_index);

log_data=data(2:n,:);

for i=1:k
    col=zeros(n-1,1);
    for j=1:n-1
       col(j)=log(data(j+1,col_index(i))/data((j), col_index(i)));
    end
    log_data(:,col_index(i))=col;
end

if(destroy)
    clear data;
end
end

