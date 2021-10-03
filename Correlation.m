%% Original variables
 Data = [Y X];
 varnames = {'BPI' 'CNYUSD' 'EURUSD' 'JPYUSD' 'GOLDUSD' 'SHANGAICompInd' 'S&P500' 'EUROSTOCK' 'NIKKEI' 'CHINA10Y' 'USA10Y' 'GERMANY10Y' 'Japan10Y' '3MShibor' 'US3MLibor' 'EUR3MLibor' 'JPY3MLibor' 'Timestamp'};
 
% Pearson correlation
figure()
 corrplot(Data, 'varNames',varnames)
 set(findall(gcf, 'type', 'axes'), 'XTickLabel', [])
 set(findall(gcf, 'type', 'axes'), 'YTickLabel', [])
 set(findall(gcf, 'type', 'axes'), 'YLabel', [])
% Spearman correlation
% corrplot(Data,'type', 'Spearman', 'varNames',varnames)
% set(findall(gcf, 'type', 'axes'), 'XTickLabel', [])
% set(findall(gcf, 'type', 'axes'), 'YTickLabel', [])
% set(findall(gcf, 'type', 'axes'), 'YLabel', [])
% Heatmap
figure()
c = corr(Data);
imagesc(c); 
set(gca, 'XTick', 1:19); 
set(gca, 'YTick', 1:19); 
set(gca, 'XTickLabel', varnames);
set(gca, 'YTickLabel', varnames); 
title('Heatmap - Pearson correlation matrix', 'FontSize', 10); 
colormap('cool'); 
colorbar;  
 
%% Corrplot : Lagged
 X_tm1=lagmatrix(X,1);
 Y_2=Y(2:end);
 Data_2=[Y_2 X_tm1(2:end,:)];

% Pearson correlation
 figure()
 corrplot(Data_2, 'varNames',varnames)
 set(findall(gcf, 'type', 'axes'), 'XTickLabel', [])
 set(findall(gcf, 'type', 'axes'), 'YTickLabel', [])
 set(findall(gcf, 'type', 'axes'), 'YLabel', [])
% Spearman correlation
% corrplot(Data_2,'type', 'Spearman', 'varNames',varnames)
% set(findall(gcf, 'type', 'axes'), 'XTickLabel', [])
% set(findall(gcf, 'type', 'axes'), 'YTickLabel', [])
% set(findall(gcf, 'type', 'axes'), 'YLabel', [])
% Heatmap
figure()
c = corr(Data_2);
imagesc(c); 
set(gca, 'XTick', 1:19); 
set(gca, 'YTick', 1:19); 
set(gca, 'XTickLabel', varnames);
set(gca, 'YTickLabel', varnames); 
title('Heatmap Lagged Variables- Spearman correlation matrix', 'FontSize', 10); 
colormap('cool'); 
colorbar;
 