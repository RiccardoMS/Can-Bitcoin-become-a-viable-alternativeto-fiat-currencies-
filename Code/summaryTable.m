function sumTable=summaryTable(data, dataTable)
% Compute the summary statistics of the dataset
% data: matrix containing all the dataset
% dataTable: table containing all the dataset labeled

sumTable=table('rownames',dataTable.Properties.VariableNames);
sumTable.N=sum(~isnan(data(:,1:18)))';
sumTable.Mean=mean(data(:,1:18),'omitnan')';
sumTable.SD=std(data(:,1:18),'omitnan')';
sumTable.Min=min(data(:,1:18),[],'omitnan')';
sumTable.Max=max(data(:,1:18),[],'omitnan')';
disp(sumTable)
end

