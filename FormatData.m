% Works on single data colums; check the dates starting from 2010-AUG-18 to
% 2017-MAR-17. If a date is not present (no recording), insert a NaN.

start=BPI(1,1); 
final=BPI(2404,1);
range=start:final;

data=zeros(2404,18);

%1st column: BPI
j=1;
for i=1:2404
    if (BPI(j,1)==range(i))
        data(i,1)=BPI(j,2);
        j=j+1;
    else
        data(i,1)=NaN;
    end
end

for k=1:size(data,1)
    if (data(k,1)==0)
        data(k,1)=NaN;
    end
end
%2nd column: CNY_USD
j=1;
for i=1:2404
    if (CNY_USD(j,1)==range(i) && j<size(CNY_USD,1))
        data(i,2)=CNY_USD(j,2);
        j=j+1;
    else
        data(i,2)=NaN;
    end
end
for k=1:size(data,1)
    if (data(k,2)==0)
        data(k,2)=NaN;
    end
end
clear CNY_USD;
%3rd column: EUR_USD
j=1;
for i=1:2404
    if (EUR_USD(j,1)==range(i) && j<size(EUR_USD,1))
        data(i,3)=EUR_USD(j,2);
        j=j+1;
    else
        data(i,3)=NaN;
    end
end
for k=1:size(data,1)
    if (data(k,3)==0)
        data(k,3)=NaN;
    end
end
 clear EUR_USD
%4th column: JPY_USD
j=1;
for i=1:2404
    if (JPY_USD(j,1)==range(i) && j<size(JPY_USD,1))
        data(i,4)=JPY_USD(j,2);
        j=j+1;
    else
        data(i,4)=NaN;
    end
end
for k=1:size(data,1)
    if (data(k,4)==0)
        data(k,4)=NaN;
    end
end
clear JPY_USD;
%5th column: GOLD_USD
j=1;
for i=1:2404
    if (GOLD_USD(j,1)==range(i) && j<size(GOLD_USD,1))
        data(i,5)=GOLD_USD(j,2);
        j=j+1;
    else
        data(i,5)=NaN;
    end
end
for k=1:size(data,1)
    if (data(k,5)==0)
        data(k,5)=NaN;
    end
end
clear GOLD_USD;
%6th column: Shangai
j=1;
for i=1:2404
    if (Shangai(j,1)==range(i) && j<size(Shangai,1))
        data(i,6)=Shangai(j,2);
        j=j+1;
    else
        data(i,6)=NaN;
    end
end
for k=1:size(data,1)
    if (data(k,6)==0)
        data(k,6)=NaN;
    end
end
clear Shangai;
%7th column: SP500
j=1;
for i=1:2404
    if (SP500(j,1)==range(i) && j<size(SP500,1))
        data(i,7)=SP500(j,2);
        j=j+1;
    else
        data(i,7)=NaN;
    end
end
for k=1:size(data,1)
    if (data(k,7)==0)
        data(k,7)=NaN;
    end
end
clear SP500;
%8th column: EUROSTOCK
j=1;
for i=1:2404
    if (EUROSTOCK(j,1)==range(i) && j<size(EUROSTOCK,1))
        data(i,8)=EUROSTOCK(j,2);
        j=j+1;
    else
        data(i,8)=NaN;
    end
end
for k=1:size(data,1)
    if (data(k,8)==0)
        data(k,8)=NaN;
    end
end
clear EUROSTOCK;
%9th column: NIKKEI
j=1;
for i=1:2404
    if (NIKKEI(j,1)==range(i) && j<size(NIKKEI,1))
        data(i,9)=NIKKEI(j,2);
        j=j+1;
    else
        data(i,9)=NaN;
    end
end
for k=1:size(data,1)
    if (data(k,9)==0)
        data(k,9)=NaN;
    end
end
clear NIKKEI;
%10th column: China10Y
j=1;
for i=1:2404
    if (China10Y(j,1)==range(i) && j<size(China10Y,1))
        data(i,10)=China10Y(j,2);
        j=j+1;
    else
        data(i,10)=NaN;
    end
end
for k=1:size(data,1)
    if (data(k,10)==0)
        data(k,10)=NaN;
    end
end
 clear China10Y;
%11th column: USA10Y
j=1;
for i=1:2404
    if (USA10Y(j,1)==range(i) && j<size(USA10Y,1))
        data(i,11)=USA10Y(j,2);
        j=j+1;
    else
        data(i,11)=NaN;
    end
end
for k=1:size(data,1)
    if (data(k,11)==0)
        data(k,11)=NaN;
    end
end
 clear USA10Y;
%12th column: Germany10Y
j=1;
for i=1:2404
    if (Germany10Y(j,1)==range(i) && j<size(Germany10Y,1))
        data(i,12)=Germany10Y(j,2);
        j=j+1;
    else
        data(i,12)=NaN;
    end
end
for k=1:size(data,1)
    if (data(k,12)==0)
        data(k,12)=NaN;
    end
end
clear Germany10Y;
%13th column: Japan10Y
j=1;
for i=1:2404
    if (Japan10Y(j,1)==range(i) && j<size(Japan10Y,1))
        data(i,13)=Japan10Y(j,2);
        j=j+1;
    else
        data(i,13)=NaN;
    end
end
for k=1:size(data,1)
    if (data(k,13)==0)
        data(k,13)=NaN;
    end
end
clear Japan10Y;
%14th column: Shibor
j=1;
for i=1:2404
    if (Shibor(j,1)==range(i) && j<size(Shibor,1))
        data(i,14)=Shibor(j,2);
        j=j+1;
    else
        data(i,14)=NaN;
    end
end

for k=1:size(data,1)
    if (data(k,14)==0)
        data(k,14)=NaN;
    end
end
clear Shibor;
%15th column: US3MLibor
j=1;
for i=1:2404
    if (US3MLibor(j,1)==range(i) && j<size(US3MLibor,1))
        data(i,15)=US3MLibor(j,2);
        j=j+1;
    else
        data(i,15)=NaN;
    end
end

for k=1:size(data,1)
    if (data(k,15)==0)
        data(k,15)=NaN;
    end
end
clear US3MLibor;
%16th column: Eur3MLibor
j=1;
for i=1:2404
    if (Eur3MLibor(j,1)==range(i) && j<size(Eur3MLibor,1))
        data(i,16)=Eur3MLibor(j,2);
        j=j+1;
    else
        data(i,16)=NaN;
    end
end

for k=1:size(data,1)
    if (data(k,16)==0)
        data(k,16)=NaN;
    end
end
clear Eur3MLibor;
%16th column: Jpy3MLibor
j=1;
for i=1:2404
    if (Jpy3MLibor(j,1)==range(i) && j<size(Jpy3MLibor,1))
        data(i,17)=Jpy3MLibor(j,2);
        j=j+1;
    else
        data(i,17)=NaN;
    end
end
for k=1:size(data,1)
    if (data(k,17)==0)
        data(k,17)=NaN;
    end
end
clear Jpy3MLibor;
%17th column: Timestamp
data(:,18)=734368:(734368+2403);

% Since we need CNY/USD and JPY/USD we invert the USD/CNY and the USD/JPY
data(:,2)=1./data(:,2);
data(:,4)=1./data(:,4);

%remove auxiliary variables
clear i; clear j; clear range; clear start; clear final; clear k;