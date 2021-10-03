function [coeff, SE, tstat, R2, adjR2, Eps] = myOLS(Y, X)

% this function implements the standard OLS estimation

% INPUT: 
% - Dependent variable: Y (N x 1)
% - Regressors: X (N x k)
%
% OUTPUT: 
% - Estimated coefficients: coeff (k+1 x 1)
% - Standard Errors: SE (k+1 x 1)
% - t statistics: tstat (k+1 x 1)
% - Residuals: Eps (N x 1)
% - R squared: R2 (1 x 1)
% - Adjusted R squared: adjR2 (1 x 1)

n = size(X,1); % sample size
k = size(X,2); % number of regressors

% rewrite the model s.t. Y = A*b + Eps 
% where b = [alpha beta]' 
A = [ones(n,1) X];

% estimate coefffcients
% coeff = (A'*A)\(A'*y) can be simplified as
coeff = A\Y;
% compute residuals
Eps = Y - A*coeff;
% compute standard errors
SE = sqrt(diag(inv(A'*A)*var(Eps)));
% compute t statistics
tstat = (coeff./SE);
% compute R squared
R2 = 1 - var(Eps)/var(Y);
% adjusted: R2-(1-R2)*k/(n-k-1)
adjR2 = R2-(1-R2)*k/(n-k-1);

end