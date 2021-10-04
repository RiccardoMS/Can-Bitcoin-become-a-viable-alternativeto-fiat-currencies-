function [W, pvalueW]= Wtest(X,Y,Eps)
% check for heteroskedasticity using White test

n = length(Y); % sample size

Xw = [X X.^2]; % construct regressors
[~, ~, ~, R2_w, ~, ~] = myOLS(Eps.^2, Xw);
W = n*R2_w;
df = size(Xw,2); 
pvalueW = 1 - chi2cdf(W,df);

end