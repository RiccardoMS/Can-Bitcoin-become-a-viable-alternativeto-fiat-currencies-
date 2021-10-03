%main
clear all; clc;
%importer;
load('asset.mat');

ImportData;

% Clean the data, insert NaN where the Data are not present and create the
% matrix data containing all the Dataset  from 2010-08-18 to 2017-03-17
FormatData;

%Create a labeled table of the Dataset
dataTable=array2table(data);
dataTable.Properties.VariableNames = {'BPI' 'CNY_USD' 'EUR_USD' 'JPY_USD' 'GOLD_USD' 'SHANGAICompInd' 'S&P500' 'EUROSTOCK' 'NIKKEI' 'CHINA10Y' 'USA10Y' 'GERMANY10Y' 'Japan10Y' '3MShibor' 'US3MLibor' 'EUR3MLibor' 'JPY3MLibor'  'Timestamp'};

% Summary statistics table: N, Mean, Sd, min,  max

sumTable=summaryTable(data, dataTable);

%% Plots
%bitcoin price index
figure()
plot(data(:,1))
title('Bitcoin Price Index')

%comparison of log returns of BTC, CNY, EUR, JPY
figure()
subplot(2,2,1)
plot(price2ret(data(:,1)))
title('Log returns of BTC')
subplot(2,2,2)
plot(price2ret(data(:,2)))
title('Log returns of CNY') 
subplot(2,2,3)
plot(price2ret(data(:,3)))
title('Log returns of EUR')
subplot(2,2,4)
plot(price2ret(data(:,4)))
title('Log returns of JPY')
%% Data cleaning and Logreturn
% Clean the data removing the rows containing a NaN
data_cleaned = cleanData(data,false);

%% BPI Analysis 

y = log(BPI(1:2404,2)); % log prices until 17/3/2017
T = length(y); % sample size
figure()
qqplot(y)

% Plot the log prices, the ACF and the PACF of y
plot_dependent_var(y, ' log price of $y_t$');
% The ACF decrease to zero slowly -> possible presence of a unit root
% From the PACF plot we see that probably there is a unit root

%% Dickey Fuller and Augmented Dickey Fuller to establish if the series has a unit root

% DF (lags = 0)
[hDF0,pvalDF0,tstatDF0,~,reg_yDF] = adftest(y,'model','TS','lags',0);
pvalDF0
reg_yDF.names
[reg_yDF.coeff reg_yDF.se reg_yDF.tStats.t reg_yDF.tStats.pVal]
% Accept unit root with 1% and 5% level (pValue=0.4877)
% Reject drift =0 with 1% level
% Accept trend = 0 with 1% and 5% level

% ADF - start with p=8
[hADF,pvalADF,~,~,reg_yADF] = adftest(y,'model','TS','lags',8);
pvalADF
reg_yADF.names 
[reg_yADF.coeff reg_yADF.se reg_yADF.tStats.t reg_yADF.tStats.pVal]
% Accept unit root with 1% and 5% level (pValue=0.4104)
% Reject drift =0 with 1% level
% Accept trend = 0 with 1% and 5% level
% Relevant lags with 1% level: 1, 2, 5, 6, 7  -> ADF with p=7 and model ARD

[h,pvalADF,~,~,reg_yADF] = adftest(y,'model','ARD','lags',7);
pvalADF
reg_yADF.names
[reg_yADF.coeff reg_yADF.se  reg_yADF.tStats.t  reg_yADF.tStats.pVal]
% Accept unitary root with evidence at 3.9% level (pValue=0.0393) -> consider dy 

%% We pass to the log returns dy

dy = diff(y); % log returns
T = length(dy);
figure
qqplot(dy)

% Plot the log returns, the ACF and the PACF of dy
plot_dependent_var(dy, ' log returns of $y_t$');
% From the plot of the ACf and the PACF we notice a seasonality effect(lags 5-6)
% and that the lag 1 is relevant

% Perform Durbin-Watson, Ljung-Box and Breusch-Godfrey tests for autocorrelation
autocorr_test(dy,[],6)
%The three pValues are very small -> reject the null hypothesis of no autocorrelation
%% Dickey Fuller and Augmented Dickey Fuller to establish if dy has a unit root

% DF (lags = 0)
[h,pvalADF,tstatADF,~,reg_yDF] = adftest(dy,'model','TS','lags',0);
pvalADF
reg_yDF.names
[reg_yDF.coeff reg_yDF.se reg_yDF.tStats.t reg_yDF.tStats.pVal]
% Reject trend = 0 (2.7% level)
% Reject drift = 0
% Reject unitary root with over 0.1% level

% ADF - p=7
[h,pvalADF,tstatADF,~,reg_yADF] = adftest(dy,'model','TS','lags',7);
pvalADF
reg_yADF.names 
[reg_yADF.coeff reg_yADF.se reg_yADF(1).tStats.t reg_yADF(1).tStats.pVal]
% Relevant lags at 5% level: 2,3,6
% Accept trend = 0
% Reject drift = 0
% Reject unitary root with over 0.1% level  -> ADF with p=6, model ARD

[h,pvalADF,tstatADF,~,reg_yADF] = adftest(dy,'model','ARD','lags',6);
pvalADF
reg_yADF.names 
[reg_yADF.coeff reg_yADF.se reg_yADF(1).tStats.t reg_yADF(1).tStats.pVal]
% Reject drift = 0
% Reject unitary root with over 0.1% level  
% Reject unitary root!

%% ARMA model selection

disp("ARMA")

loglikelihood = zeros(7,7); % Initialize
pq = zeros(7,7); % from ACF/PACF 6 lags seem enough
T1=zeros(7,7);

for p = 0:6
    for q = 0:6
        Mdl = arima(p,0,q);
        [EstMdl, ~, logl] = estimate(Mdl,dy(Mdl.P+1:end),'Y0', dy(1:Mdl.P),'Display','off');
        loglikelihood(p+1,q+1) = logl;
        pq(p+1,q+1) = p + q;
        T1(p+1,q+1)=size(dy(Mdl.P+1:end),1);

     end
end

loglikelihood = reshape(loglikelihood,49,1);
pq = reshape(pq, 49, 1);
T1=reshape(T1, 49, 1);
[aic, bic] = aicbic(loglikelihood, pq+1, T1);
% The rows correspond to the AR(p) and the columns correspond to the MA(q)
bic = reshape(bic,7,7)
min(min(bic))
% The model with lower BIC is ARMA(0,1)
aic = reshape(aic,7,7)
min(min(aic))
% The model with lower AIC is ARMA(0,6)
% Following "Parsimony Principle" we would go for ARMA(0,1)

%% Test for ARCH effects
Mdl_ARMA = arima(0,0,1);
[EstMdl_ARMA, ~, logl] = estimate(Mdl_ARMA,dy,'Display','off');
[e_ARMA, ~, ~] = infer(EstMdl_ARMA,dy); % residuals
[~, pValue_ARMA, ~, ~] = archtest(e_ARMA);
pValue_ARMA
% pValue of 0 -> We reject the null hypothesis of no ARCH effects 

% Plot ACF and PACF of squared residuals
plot_ACF_PACF(e_ARMA.^2,' of squared residuals ARMA')

%% ARMA-GARCH
Mdl_ARMA_GARCH = arima(0,0,1);
CVarMdl = garch(1,1);  % specify GARCH(1,1) for conditional variance
Mdl.Variance = CVarMdl;
[EstMdl_ARMA_GARCH, ~, logl] = estimate(Mdl_ARMA_GARCH,dy,'Display','off');
[e_ARMA_GARCH, ~, ~] = infer(EstMdl_ARMA_GARCH,dy); % residuals
[~, pValue_ARMA_GARCH, ~, ~] = archtest(e_ARMA_GARCH);
pValue_ARMA_GARCH

% Plot ACF and PACF of squared residuals
plot_ACF_PACF(e_ARMA_GARCH.^2,' of squared residuals ARMA GARCH')


%% AR(1)-GARCH(1,1) as in the paper
Mdl_AR_GARCH = arima(1,0,0);
CVarMdl = garch(1,1);  % specify GARCH(1,1) for conditionall variance
Mdl_AR_GARCH.Variance = CVarMdl;
[EstMdl_AR_GARCH, ~, logl] = estimate(Mdl_AR_GARCH,dy,'Display','off');
[e_AR_GARCH, ~, ~] = infer(EstMdl_AR_GARCH,dy); % residuals
[~, pValue_AR_GARCH, ~, ~] = archtest(e_AR_GARCH);
pValue_AR_GARCH

% Plot ACF and PACF of squared residuals
plot_ACF_PACF(e_AR_GARCH.^2,' of squared residuals AR GARCH')


%% 2 Linear regression
% Why using differences? Regressors involve integrated series?
% Analyze acf/pcf plots of one variable per group
% CNY_USD
plot_ACF_PACF(data_cleaned(:,2),' of CNY/USD')
% ShangaiCompInd
plot_ACF_PACF(data_cleaned(:,6),' of ShangaiCompInd')
% China10Y
plot_ACF_PACF(data_cleaned(:,10),' of China10Y')
% 3MShibor
plot_ACF_PACF(data_cleaned(:,14),' of 3MShibor')

% Report conrespondent dickey_fuller tests: 
% CNY_USD
[~,ADF_CNY,~,~,~] = adftest(data_cleaned(:,2),'model','TS','lags',2);
% ShangaiCompInd
[~,ADF_ShCI,~,~,~] = adftest(data_cleaned(:,6),'model','TS','lags',6);
% China10Y
[~,ADF_Ch10Y,~,~,~] = adftest(data_cleaned(:,10),'model','TS','lags',5);
% 3MShibor
[~,ADF_3MSh,~,~,~] = adftest(data_cleaned(:,14),'model','TS','lags',9);

% data
X =[price2ret(data_cleaned(:,2:9)) diff(data_cleaned(:,10:17)) normalize(data_cleaned(2:end,18))];
Y = price2ret(data_cleaned(:,1));

% Run to see the scatterplots of the variables in the dataset
 Scatterplots;

% Run to see correlation matrices for variables and lagged variables
 Correlation;


%% Regression 1: without lagged data

disp("Linear Regression")

LM_1=linear_regression(X,Y,1)
% From the fat tail of the qqplot we suppose that residuals are not normal
% distributed

% Shapiro-Wilk test: pValue=0 -> Normality of residuals is rejected; 
% insights from fat tailed qqplot are backed up by the pval of a SW normality test

% Breusch-Pagan Test: pValue=1.6202e-09 -> Reject the null hypothesis of no heteroskedasticity

% White Test: pValue=0 -> Reject the null hypothesis of no heteroskedasticity

% Autocorrelation test: The three pValues are very small -> reject the null hypothesis of no
% autocorrelation

%% Regression 2: adding lagged variables
% Simple idea: maybe exogenous variables affect dependent one with delay
% (1day considered)
X_tm1=lagmatrix(X,1);
X_2=[X(2:end,:) X_tm1(2:end,:)];
Y_2=Y(2:end);

LM_2=linear_regression(X_2,Y_2,2);

% Breusch-Pagan Test: pValue=0 -> Reject the null hypothesis of no heteroskedasticity

% White Test: pValue=0 -> Reject the null hypothesis of no heteroskedasticity

% Autocorrelation check: all test show evidence for autocorr of residuals:
% time series modeling of Y is required

% Autocorrelation test: The three pValues are very small -> reject the null hypothesis of no
% autocorrelation

% F-TEST
% Non-lagged variables: non significant, pval=0.5831
% Lagged variables: significant at 0.08

%% Regression 3: only lagged variables
X_3=X_2(:,18:34);

LM_3=linear_regression(X_3,Y_2,1);

% Breusch-Pagan Test: pValue=1.4475e-07 -> Reject the null hypothesis of no heteroskedasticity

% White Test: pValue=4.5992e-05 -> Reject the null hypothesis of no heteroskedasticity

% Autocorrelation check: all test show evidence for autocorr of residuals:
% time series modeling of Y is required

% Autocorrelation test: The three pValues are very small -> reject the null hypothesis of no
% autocorrelation

%% 3 Armax model selection

disp("ARMAX")

X =[price2ret(data_cleaned(:,2:9)) diff(data_cleaned(:,10:17))];
% We lag the X matrix since Y(t) is a function of X(t-1)
% X = lagmatrix(Xraw,1);
X = X(1:end-1,:);
Y = price2ret(data_cleaned(:,1));
Y = Y(2:end);
T = size(Y,1);


loglikelihood = zeros(7,7); % Initialize
pq = zeros(7,7);
T1=zeros(7,7);
for p = 0:6
    for q = 0:6
        Mdl = arima(p,0,q);
        fprintf('%i %%\n', ceil(100*(7*p+q)/48))
        [EstMdl, ~, logl] = estimate(Mdl,Y(Mdl.P+1:end),'X',X,'Display','off');
        loglikelihood(p+1,q+1) = logl;
        pq(p+1,q+1) = p + q;
        T1(p+1,q+1)=size(Y(Mdl.P+1:end),1);
     end
end

loglikelihood = reshape(loglikelihood,49,1);
T1=reshape(T1, 49, 1);
pq = reshape(pq, 49, 1);
[aic, bic] = aicbic(loglikelihood, pq+1, T1);
% The rows correspond to the AR(p) and the columns correspond to the MA(q)
bic = reshape(bic,7,7)
min(min(bic))
% The model with lower BIC is ARMAX(3,6)

aic = reshape(aic,7,7)
min(min(aic))
% The model with lower AIC is ARMAX(3,6)
%% Test for ARCH effects

Mdl_ARMAX = arima(3,0,6);
[EstMdl_ARMAX, ~, logl] = estimate(Mdl_ARMAX,Y(Mdl_ARMAX.P+1:end),'X',X,'Display','off');
e_ARMAX = infer(EstMdl_ARMAX,Y(Mdl_ARMAX.P+1:end),'X',X); % residuals
[~, pValue_ARMAX, ~, ~] = archtest(e_ARMAX);
pValue_ARMAX
% pValue of 0 -> We reject the null hypothesis of no ARCH effects

% Plot ACF and PACF of squared residuals
plot_ACF_PACF(e_ARMAX.^2,' of squared residuals ARMAX')
% We observe ARCH effects

%% 3 ARMAX-GARCH(1,1) model selection
% We model the variance in a GARCH or EGARCH model to fit the ARCH effects
% We rely on GARCH(1,1) and EGARCH(1,1) as they are widely used in
% literature

disp("ARMAX-GARCH(1,1)")

X =[price2ret(data_cleaned(:,2:9)) diff(data_cleaned(:,10:17))];
% We lag the X matrix since Y(t) is a function of X(t-1)
X = X(1:end-1,:);
Y = price2ret(data_cleaned(:,1));
Y = Y(2:end);
T = size(Y,1);

loglikelihood = zeros(7,7); % Initialize
pq = zeros(7,7);
T1=zeros(7,7);
for p = 0:6
    for q = 0:6
        Mdl = arima(p,0,q);
        fprintf('%i %%\n', ceil(100*(7*p+q)/48))
        Mdl.Variance=garch(1,1);
        [EstMdl, ~, logl] = estimate(Mdl,Y(Mdl.P+1:end),'X',X,'Display','off');
        loglikelihood(p+1,q+1) = logl;
        pq(p+1,q+1) = p + q;
        T1(p+1,q+1)=size(Y(Mdl.P+1:end),1);
     end
end
% Remark: on one of the three machines we used, the model estimation 
% encountered some issues for MA = 6, then it can happen that a machine
% has convergence problem for MA = 6.


loglikelihood = reshape(loglikelihood,49,1);
T1=reshape(T1, 49, 1);
pq = reshape(pq, 49, 1);
[aic, bic] = aicbic(loglikelihood, pq+1, T1);
% The rows correspond to the AR(p) and the columns correspond to the MA(q)
bic = reshape(bic,7,7)
min(min(bic))
% The model with lower BIC is ARMAX(4,2)

aic = reshape(aic,7,7)
min(min(aic))
% The model with lower AIC is ARMAX(6,6)

%% ARMAX(4,2)-GARCH(1,1)

Mdl_ARMAX_GARCH = arima(4,0,2);
Mdl_ARMAX_GARCH.Variance = garch(1,1);
[Mdl_ARMAX_GARCH, ~, logl] = estimate(Mdl_ARMAX_GARCH,Y(Mdl_ARMAX_GARCH.P+1:end),'X',X);
[e_ARMAX_GARCH, Var_ARMAX_GARCH] = infer(Mdl_ARMAX_GARCH,Y(Mdl_ARMAX_GARCH.P+1:end),'X',X); % residuals
[~, pValue_ARMAX_GARCH, ~, ~] = archtest(e_ARMAX_GARCH);
pValue_ARMAX_GARCH
% pValue of 0 -> We reject the null hypothesis of no ARCH effects

% Plot ACF and PACF of squared residuals
plot_ACF_PACF(e_ARMAX_GARCH.^2,' of squared residuals ARMAX')
[~, bic_ARMAX_GARCH] = aicbic(logl, 3+2+1+1+1, T);

figure()
hold on
Y_est = Y(Mdl_ARMAX_GARCH.P+1:end) - e_ARMAX_GARCH;
x = 1:length(Y_est);
x_area = [x, fliplr(x)];
y_area = [(Y_est-1.96*Var_ARMAX_GARCH)', fliplr((Y_est+1.96*Var_ARMAX_GARCH)')];
fill(x_area, y_area, [1 0.7 0.7]);
plot(Y(Mdl_ARMAX_GARCH.P+1:end),'b-')
plot(Y_est+1.96*Var_ARMAX_GARCH, 'r-')
plot(Y_est, 'r')
plot(Y_est-1.96*Var_ARMAX_GARCH, 'r-')
xlim([500 1000])
% Save coefficients

lambda42 = log(Mdl_ARMAX_GARCH.Variance.Constant);
gamma42 = cell2mat(Mdl_ARMAX_GARCH.Variance.GARCH);
alpha42 = cell2mat(Mdl_ARMAX_GARCH.Variance.ARCH);
delta42_1 = cell2mat(Mdl_ARMAX_GARCH.AR);
delta42_2 = cell2mat(Mdl_ARMAX_GARCH.MA);
beta042 = Mdl_ARMAX_GARCH.Constant;
beta42 = Mdl_ARMAX_GARCH.Beta;

%% 3 ARMAX-EGARCH(1,1) model selection

disp("ARMAX-EGARCH(1,1)")

X =[price2ret(data_cleaned(:,2:9)) diff(data_cleaned(:,10:17))];
% We lag the X matrix since Y(t) is a function of X(t-1)
X = X(1:end-1,:);
Y = price2ret(data_cleaned(:,1));
Y = Y(2:end);
T = size(Y,1);

loglikelihood = zeros(7,7); % Initialize
pq = zeros(7,7);
T1=zeros(7,7);
for p = 0:6
    for q = 0:6
        Mdl = arima(p,0,q);
        fprintf('%i %%\n', ceil(100*(7*p+q)/48))
        Mdl.Variance= egarch(1,1);
        [EstMdl, ~, logl] = estimate(Mdl,Y(Mdl.P+1:end),'X',X,'Display','off');
        loglikelihood(p+1,q+1) = logl;
        pq(p+1,q+1) = p + q;
        T1(p+1,q+1)=size(Y(Mdl.P+1:end),1);
     end
end

loglikelihood = reshape(loglikelihood,49,1);
T1=reshape(T1, 49, 1);
pq = reshape(pq, 49, 1);
[aic, bic] = aicbic(loglikelihood, pq+1, T1);
% The rows correspond to the AR(p) and the columns correspond to the MA(q)
bic = reshape(bic,7,7)
min(min(bic))
% The model with lower BIC is ARMAX(1,1)   

aic = reshape(aic,7,7)
min(min(aic))
% The model with lower AIC is ARMAX(1,1)   
%% ARMAX(1,1)-EGARCH(1,1)
Mdl_ARMAX_EGARCH = arima(1,0,1);
Mdl_ARMAX_EGARCH.Variance = egarch(1,1);
[Mdl_ARMAX_EGARCH, ~, logl] = estimate(Mdl_ARMAX_EGARCH,Y(Mdl_ARMAX_EGARCH.P+1:end),'X',X,'Display','off');
[e_ARMAX_EGARCH, Var_ARMAX_EGARCH] = infer(Mdl_ARMAX_EGARCH,Y(Mdl_ARMAX_EGARCH.P+1:end),'X',X); % residuals
[~, pValue_ARMAX_EGARCH, ~, ~] = archtest(e_ARMAX_EGARCH);
pValue_ARMAX_EGARCH
% pValue of 0 -> We reject the null hypothesis of no ARCH effects

% Plot ACF and PACF of squared residuals
plot_ACF_PACF(e_ARMAX_EGARCH.^2,' of squared residuals ARMAX')
[~, bic_ARMAX_EGARCH] = aicbic(logl, 3+2+1+1+1, T);

figure()
plot(Y(Mdl_ARMAX_EGARCH.P+1:end))
hold on
Y_est = Y(Mdl_ARMAX_EGARCH.P+1:end) - e_ARMAX_EGARCH;
plot(Y_est+1.96*Var_ARMAX_EGARCH, 'r-')
plot(Y_est, 'r')
plot(Y_est-1.96*Var_ARMAX_EGARCH, 'r-')

% Garch(1,1) and Egarch(1,1) appear unable to fit properly the ARCH effects

%% 4 ARX_GARCHX
% We then build a ARX(1)-GARCHX(1,1) model, where the variance can be
% estimated on the base of the exogeneous regresseros X, in addition to the
% ARCH and GARCH terms. The regressors are preprocessed in two different
% ways for the ARX and for the GARCHX part (X_m and X_s respectively)

X =[price2ret(data_cleaned(:,2:9)) data_cleaned(2:end,10:17)];             % 19/8 - 17/3
X = X(1:end-1,:);                                                          % 19/8 - 16/3

X_modified = [price2ret(data_cleaned(:,2:9)) diff(data_cleaned(:,10:17))]; % 19/8 - 17/3
X_modified = X_modified(1:end-1,:);                                        % 19/8 - 16/3
T = size(X,1);

% Add ones column for the intercept of both ARMAX and GARCHX part, in 
% accordance with GARCHX_Neg_Logl funcition
X_s = [ones(T,1), X];
X_m = [ones(T,1), X_modified];

Y = price2ret(data_cleaned(:,1));                                          % 19/8 - 17/3                       
Y0 = Y(1);                                                                 % presample 19/8
Y = Y(2:end);                                                              % 20/8 - 17/3

%% ARX(1)-GARCH(1,1)
% First we fit the ARX(1)-GARCH(1,1) model to retrieve a first estimate of
% the AR, ARCH, GARCH coefficients, and also of the Regressors X of the ARX
% model
Mdl = arima(1,0,0);
Mdl.Variance = garch(1,1);
Est_Mdl = estimate(Mdl, Y, 'Y0', Y0,  'X', X_modified)

% We save the found coefficients
lambda0 = log(Est_Mdl.Variance.Constant);
gamma0 = cell2mat(Est_Mdl.Variance.GARCH);
alpha0 = cell2mat(Est_Mdl.Variance.ARCH);
delta0 = cell2mat(Est_Mdl.AR);
beta0 = Est_Mdl.Constant;
beta = Est_Mdl.Beta;
%% ARX(1)-GARCHX(1,1) - optimization problem

% We set the boundaries. A large bound for the regressors, which should
% never be reached. For the AR, ARCH, GARCH elements we can ask for a
% tighter bound to ensure the stability of the model
LB = [-100, -500*ones(1,16),-5,-5, -500*ones(1,17), -1];     
UB = [1, 500*ones(1,16),5,5, 500*ones(1,17), 1];

% We set highly demanding options to ensure a precise estimation of the
% convergence point
options = optimoptions('fmincon','UseParallel',false,...
    'MaxFunEvals', 500000,'algorithm','interior-point', ...
    'TolFun' ,1e-12,'TolX',1e-12,'MaxIter', 2000, 'StepTolerance', 1e-80, 'ObjectiveLimit', -1e50);

% Sensitivity Analysis find an effective initial guess for the Maximum
% Likelihood problem
% Note that the problem is subjet to instability

% Best_guess=Sensitivity_Analysis(X_s, X_m, Y0, Y, T, LB, UB, options, Est_Mdl);
Best_guess=5.5;

% Using the best guess
param0 = [lambda0, Best_guess*ones(1,16), gamma0, alpha0, beta0, beta, delta0];

[paramMLE, Opt_NLogl] = fmincon(@(theta) GARCHX_Neg_Logl(theta, T, Y, X_s, X_m, Y0), param0,...
           [zeros(1,17) 1 1 zeros(1,18)],1,...  % alpha + gamma <1
           [],[],LB,UB,[],options);

paramMLE
[aic_Opt, bic_Opt] = aicbic(-Opt_NLogl, 37, T);

% Optimal Parameter according to the paper
param_paper = [-6.789, -199.9, 61.10 ,3.604, -21.24, 39.91, 36.08, -38.95, ...
   -7.737, 1.159, -2.101, 4.550, -9.660, -0.2688, -1.726, 2.042, 3.606,...
    0.1924, 0.366, 0.03093, -0.6110, -0.7133, 0.2073, 0.1738, 0.0820, ...
    0.0802, 0.1361, -0.1218, -0.0044, -0.00300, 0.0106, -0.0090, -0.0015,...
    -0.0094, -0.0011, -0.03025, 0.01958]; 

% Compare found Loglikelihood with paper parameters loglikelihood
% No AIC/BIC is needed, as they have the same number of parameters
disp("Compare Optimal Parameters with Paper Parameters")
Opt_NLogl
Pap_NLogl = GARCHX_Neg_Logl(param_paper, T, Y, X_s, X_m, Y0)
% This is surely not a likelihood maximum, w.r.t. our data
[paramMLE2, Opt_NLogl2] = fmincon(@(theta) GARCHX_Neg_Logl(theta, T, Y, X_s, X_m, Y0), ...
           param_paper, [zeros(1,17) 1 1 zeros(1,18)],1,...  % alpha + gamma <1
           [],[],LB,UB,[],options);
Opt_NLogl2
%% Likelihood ratio test (LRT)
% Infer parameters significativity
% It takes almost 20 minutes - results are previously saved

[chi_test,pValue_LRT] = likelihood_ratio_test(X_s, X_m, Y0, Y, T, Opt_NLogl ,Est_Mdl, paramMLE, param0, param_paper);

% Estimated Parameter and relative pValue based on LRT
[paramMLE', pValue_LRT']
%% Variable Selection
% On the base of a group LRT we select a subset of variables to keep in the
% model. The function outputs the restricted model parameters and relative
% pValues
[param_Rest, pValue_Rest] = Variable_Selection(X_s, X_m, Y0, Y, T, Opt_NLogl ,Est_Mdl, paramMLE, param0, param_paper);

% Plot remaining parameters and relative pValues
j=1;
for i=1:length(pValue_Rest)
    if(pValue_Rest(i)==1)
        a=0;
    else
        left_param_pos(j)=i;
        pVal_Rest(j) = pValue_Rest(i);
        coeff_Rest(j) = param_Rest(i);
        j=j+1;
    end
end
disp("Parameter Position    Estimated coeff   pValue")
[left_param_pos', coeff_Rest', pVal_Rest']

[aic_Rest, bic_Rest] = aicbic(-GARCHX_Neg_Logl(param_Rest, T, Y, X_s, X_m, Y0), length(coeff_Rest), T);

%% Plot of the real and predicted series and its CI at 95%

% Plot of the full model
Opt_residual = inference(X_s, X_m, Y, Y0, T, paramMLE);

% Test for ARCH effects in the full model
[~, pValue_res, ~, ~] = archtest(Opt_residual);
pValue_res

% Plot ACF and PACF of squared residuals
plot_ACF_PACF(Opt_residual.^2,' of squared residuals ARMAX-GARCHX')

% Plot of the restricted model
Rest_residual = inference(X_s, X_m, Y, Y0, T, param_Rest);

% Test for ARCH effects in the restricted model
[~, pValue_res, ~, ~] = archtest(Rest_residual);
pValue_res

% Plot ACF and PACF of squared residuals
plot_ACF_PACF(Rest_residual.^2,' of squared residuals ARMAX-GARCHX Rest')


%% ARMAX(4,2) - GARCHX(1,1)
% Note: delta_1: AR(4) - delta_2: MA(2)
% boundaries 
LB32 = [-100, -500*ones(1,16),-5,-5, -500*ones(1,17), -100*ones(1,6)];     
UB32 = [1, 500*ones(1,16),5,5, 500*ones(1,17), 100*ones(1,6)];

% We set highly demanding options to ensure a precise estimation of the
% convergence point
options = optimoptions('fmincon','UseParallel',false,...
    'MaxFunEvals', 500000,'algorithm','interior-point', ...
    'TolFun' ,1e-12,'TolX',1e-12,'MaxIter', 2000, 'StepTolerance', 1e-80, 'ObjectiveLimit', -1e50);

param042 = [lambda42, paramMLE(2:17), gamma42, alpha42, beta042, beta42, delta42_1, delta42_2];

% Data
X42 =[price2ret(data_cleaned(:,2:9)) data_cleaned(2:end,10:17)];             
X42 = X42(1:end-4,:);                                                          

X_modified42 = [price2ret(data_cleaned(:,2:9)) diff(data_cleaned(:,10:17))]; 
X_modified42 = X_modified42(1:end-4,:);                                        
T42 = size(X42,1);

X_s42 = [ones(T42,1), X42];
X_m42 = [ones(T42,1), X_modified42];

Y42 = price2ret(data_cleaned(:,1));                                                                 
Y042 = Y42(1:4);                                                                 
Y42 = Y42(5:end);  

Eps042 = [Y42(3)-mean(Y42); Y42(4)-mean(Y42)];

% Optimization

[paramMLE_42, Opt_NLogl_42] = fmincon(@(theta) GARCHX42_Neg_Logl(theta, T42, Y42, X_s42, X_m42, Y042, Eps042), param042,...
           [zeros(1,17) 1 1 zeros(1,23);zeros(1,36) 1 1 1 1 1 1],[1;1],...  % alpha + gamma <1 ; sum(delta) <1 
           [],[],LB32,UB32,[],options);

paramMLE_42
[aic_42, bic_42] = aicbic(-Opt_NLogl_42, 42, T42);


% Plot 95pc confidence interval
res42 = inference42(X_s42, X_m42, Y42, Y042,Eps042, T42, paramMLE_42);

% Test for ARCH effects in the full model
[~, pValue_res_42, ~, ~] = archtest(res42);
pValue_res_42

% Plot ACF and PACF of squared residuals
plot_ACF_PACF(res42.^2,' of squared residuals ARMAX(4,2)-GARCHX')

