%Import BPI data
BPI1 = readtable('Dataset\BPI.csv');
BPIvalues = rows2vars(BPI1(2,:));
BPIvalues = BPIvalues(:,2);
BPIvalues.Properties.VariableNames = ["BPI"];

BPI2 = readtable('Dataset\BPI_col.xlsx');
BPIdates = BPI2(:,1);
BPIdates.Properties.VariableNames = ["Date"];

BPI = [BPIdates BPIvalues];

clear BPI1;
clear BPI2;
clear BPIdates;
clear BPIvalues;
save("BPI.mat","BPI");