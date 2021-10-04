function autocorr_test(Eps, X, lags)

T=length(Eps);

% Durbin-Watson - one-step correlation test
disp('Durbin-Watson')
[DW_pVal, DW_stat] = dwtest(Eps,[ones(length(Eps),1) X],'Tail','right')
% DW_stat near to 2 gives no evidence to reject H0: phi=0
% DW_stat near to 0 gives evidence to accept H1: phi!=0

% Ljung-Box
disp('Ljung-Box')
[~, LB_pVal, LB_stat, ~] = lbqtest(Eps,'lags',lags)
% Statistics distributed as a chi-squared with "lags" degrees of freedom


% Breusch-Godfrey
disp('Breusch-Godfrey')
laggedEps = lagmatrix(Eps,1:lags); % lagged residuals
Xbg = [X laggedEps]; % new regressors
Eps_bg = Eps; % create a copy of Eps (avoid overwriting Eps if needed later)

NaN_values = (isnan(Eps_bg) | any(isnan(Xbg),2));
NaNs = any(NaN_values);
if NaNs
   Eps_bg(NaN_values) = [];
   Xbg(NaN_values,:) = [];
   n = length(Eps_bg);
end

[~, ~, ~, R2_bg1, ~, ~] = myOLS(Eps_bg, Xbg);

BG_stat = (T-lags)*R2_bg1;
df = lags;
BG_pVal = 1-chi2cdf(BG_stat,df)
BG_stat

end
% Statistics distributed as a chi-squared with "lags" degrees of freedom

