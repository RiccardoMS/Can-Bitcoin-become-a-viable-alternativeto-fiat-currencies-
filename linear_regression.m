function LM= linear_regression(X,Y,flag)

% X: matrix of regressors -> Exchange rates, Stock Indeces, 10Y Bond, 3MLibor, Time
% Y: dependendt variable -> Bitcoin price
% Goal: grasp some insight on dependencies betweene Y and regressors

LM=fitlm(X,Y)

% Residuals plot
figure()
subplot(1,3,1)
qqplot(LM.Residuals.Raw)
title('QQplot','interpreter','latex')
xlabel('Standard Normal Quantiles','interpreter','latex')
ylabel('Input Sample Quantiles','interpreter','latex')
set(gca,'FontSize',10)

%subplot(1,2,2)
%histfit(LM.Residuals.Standardized)

if(flag==1)
% Shapiro-Wilk test
[~,pvalSW,~]=swtest(LM.Residuals.Raw)
end

% Heteroskedatisticity

subplot(1,3,2)
plot(LM.Fitted, LM.Residuals.Raw, 'bo')
xlabel('Fitted','interpreter','latex')
ylabel('Residuals','interpreter','latex')
title('Residuals VS Fitted','interpreter','latex')
set(gca,'FontSize',10)

subplot(1,3,3)
plot(LM.Fitted, sqrt(abs(LM.Residuals.Standardized)), 'bo')
xlabel('Fitted','interpreter','latex')
ylabel('$\surd{residuals}$','interpreter','latex')
title('Scale-Location','interpreter','latex')
set(gca,'FontSize',10)

% Breusch-Pagan Test 
[~,pvalBP_1,~]=BPtest(LM,true)

% White Test
[~, pvalueW_1]= Wtest(X,Y,LM.Residuals.Raw)

% Plot ACF and PACF of residuals
plot_ACF_PACF(LM.Residuals.Raw,' of residuals')

% Autocorrelation Test
autocorr_test(LM.Residuals.Raw,X,9)

switch flag 
    case 1  
        % Corrected variance estimator
        [~, SE_HAC, coeff_HAC] = hac(X,Y,'display','off');

        tstat_HAC = coeff_HAC./SE_HAC;

        NW_selection=[tstat_HAC 2*(1-tcdf(abs(tstat_HAC), size(X,1)-1))]

        %variable selection
        LM_reduced=stepwiselm(X,Y)


        % Model comparison
        Crit=table('rowNames', {'AIC' 'AICc' 'BIC'});
        Crit.Model_1 = [LM.ModelCriterion.AIC LM.ModelCriterion.AICc LM.ModelCriterion.BIC]';
        Crit.Model_Reduced = [LM_reduced.ModelCriterion.AIC LM_reduced.ModelCriterion.AICc LM_reduced.ModelCriterion.BIC]';
        disp(Crit);
        
    case 2 
        % F-TEST
        % Non lagged variable
        H=zeros(35,35)+diag(ones(34,1),1);
        H1=H(1:16,:);
        [p_NL,F_NL,~] = coefTest(LM,H1)

        % Lagged variables: significant at 0.08
        H2=H(17:32,:);
        [p_L,F_L,~] = coefTest(LM,H2)
end

end

