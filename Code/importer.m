function [asset] = importer(toxlsx,graphics)
% Importer: imports data from Yahoo Finance, parses and saves them in
% asset.mat
    
    if nargin < 1
        toxlsx = false;
    end
    if nargin < 2
        graphics = false;
    end
    
    startdate = datenum("18/08/2010",'dd/mm/yyyy');
    enddate = datenum("31-Mar-2021"); % day after the 11th
    assets = readtable("AssetsList.xlsx");
    
    for k=1:9
%         
            asset(k).name = cell2mat(table2array(assets(k,1)));
            asset(k).code = cell2mat(table2array(assets(k,2)));
            mktdata = getMarketDataViaYahoo(asset(k).code, datestr(startdate), datestr(enddate));
            asset(k).dates = datenum(table2array(mktdata(:,1)));
            asset(k).prices = table2array(mktdata(:,5));
            disp(asset(k).code+" imported");

           
    end
    
    newdates = asset(1).dates(asset(1).dates>=startdate);
    newdates = newdates(newdates<enddate);
    
    % Filters Dates 
    for k=1:length(asset)
        newprices = zeros(length(newdates),1);
        for d = 1:length(newdates)
            index = find(asset(k).dates == newdates(d));
            if ~isempty(index)
                newprices(d) = asset(k).prices(index);
            else
                if(d == 1)
                    pricesafterstart = asset(k).prices(asset(k).dates >= startdate);
                    nonzeroprices = pricesafterstart(pricesafterstart ~= 0);
                    newprices(d) = nonzeroprices(1);
                else
                    newprices(d) = newprices(d-1);
                end
            end
        end
        asset(k).dates = newdates;
        asset(k).prices = newprices;
    end
    
    % Fixes NaN
    for k=1:length(asset)
        nanindex = sort([find(isnan(asset(k).prices));find(asset(k).prices == 0)]);
        if ~isempty(nanindex) %found nan
            for nix = 1:length(nanindex)
                ni = nanindex(nix);
                if(ni == 1)
                    nonnanprices = asset(k).prices(setdiff((1:length(asset(k).prices))',nanindex));
                    asset(k).prices(ni) = nonnanprices(1);
                else
                    asset(k).prices(ni) = asset(k).prices(ni-1);
                end
            end
        end
    end
        
    % Currency converter
    asset(1).prices = 1./asset(1).prices; % First row is EUR / USD 

    save("asset.mat","asset");
    

