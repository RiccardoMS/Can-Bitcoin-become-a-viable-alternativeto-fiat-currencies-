% Import Data from 'Yahoo Finance', from 'Federal Reserve Economic Data' or
% from 'Coindesk'

% Import Data from Yahoo Finance
% The importer.m function downloads the data from Yahoo Finance and saves
% them into asset.mat

%importer;
load('asset.mat');

Shangai = [ asset(5).dates asset(5).prices];
SP500  = [ asset(6).dates asset(6).prices];
EUROSTOCK = [ asset(7).dates asset(7).prices];

% Import BPI Data from Coindesk 
ImportBPI;
BPI = [datenum(table2array(BPI(:,1))) table2array((BPI(:,2)))];

% Import Data from Federal Reserve Economic Data
EUR_USD = readtable('Dataset\DEXUSEU.xls');
EUR_USD=[datenum(table2array(EUR_USD(:,1))) table2array((EUR_USD(:,2)))];

CNY_USD = readtable('Dataset\DEXCHUS.xls');
CNY_USD=[datenum(table2array(CNY_USD(:,1))) table2array((CNY_USD(:,2)))];

JPY_USD = readtable('Dataset\DEXJPUS.xls');
JPY_USD= [datenum(table2array(JPY_USD(:,1))) table2array((JPY_USD(:,2)))];

GOLD_USD = readtable('Dataset\GOLD.xls');
GOLD_USD= [datenum(table2array(GOLD_USD(:,1))) table2array((GOLD_USD(:,2)))];

NIKKEI = readtable('Dataset\NIKKEI225.xls');
NIKKEI= [datenum(table2array(NIKKEI(:,1))) table2array((NIKKEI(:,2)))];

USA10Y = readtable('Dataset\DGS10.xls');
USA10Y= [datenum(table2array(USA10Y(:,1))) table2array((USA10Y(:,2)))];

Japan10Y = readtable('Dataset\Giappone 10 anni Dati Storici Rendimento Bond.csv');
Japan10Y=Japan10Y(:,1:2);
Japan10Y= [datenum(table2array(Japan10Y(:,1))) str2double(table2array(Japan10Y(:,2))).*10^-3];
Japan10Y=flip(Japan10Y,1);


Germany10Y=readtable('Dataset\Germany 10-Year Bond Yield Historical Data (1).csv');
Germany10Y=Germany10Y(:,1:2);
Germany10Y=[datenum(table2array(Germany10Y(:,1))) table2array((Germany10Y(:,2)))*10^-3];
Germany10Y=flip(Germany10Y,1);

China10Y=readtable('Dataset\Cina 10 anni Dati Storici Rendimento Bond.csv');
China10Y=China10Y(:,1:2);
China10Y=[datenum(table2array(China10Y(:,1))) str2double(table2array((China10Y(:,2))))*10^-3];
China10Y=flip(China10Y,1);

Eur3MLibor = readtable('Dataset\EUR3MLIBOR.xls');
Eur3MLibor=[datenum(table2array(Eur3MLibor(:,1))) table2array((Eur3MLibor(:,2)))];

Jpy3MLibor = readtable('Dataset\JPY3MLIBOR.xls');
Jpy3MLibor=[datenum(table2array(Jpy3MLibor(:,1))) table2array((Jpy3MLibor(:,2)))];

US3MLibor = readtable('Dataset\USD3MLIBOR.xls');
US3MLibor=[datenum(table2array(US3MLibor(:,1))) table2array((US3MLibor(:,2)))];

Shibor=readtable('Dataset\shibor - Copia.xlsx');
Shibor=[datenum(table2array(Shibor(:,1))) table2array((Shibor(:,2)))];

