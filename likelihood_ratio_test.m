function [chi_test,pValue_LRT] = likelihood_ratio_test(X_s, X_m, Y0, Y, T, Opt_NLogl ,Est_Mdl, paramMLE, param0, param_paper)
%% Likelihood Ratio Test

% We save the found coefficients
lambda0 = log(Est_Mdl.Variance.Constant);
gamma0 = cell2mat(Est_Mdl.Variance.GARCH);
alpha0 = cell2mat(Est_Mdl.Variance.ARCH);
delta0 = cell2mat(Est_Mdl.AR);
beta0 = Est_Mdl.Constant;
beta = Est_Mdl.Beta;

% Allocate Negative Loglikelihood
Res_NLogl = zeros(length(paramMLE),1);

options = optimoptions('fmincon','UseParallel',false,'Display','off',...
    'MaxFunEvals', 500000,'algorithm','interior-point', ...
    'TolFun' ,1e-12,'TolX',1e-12,'MaxIter', 2000, 'StepTolerance', 1e-80, 'ObjectiveLimit', -1e50);

disp('Likelihood Ratio Test Computing...')

for i = 1:length(paramMLE)
    
    LB = [-100, -500*ones(1,16),-5,-5, -500*ones(1,17), -1];     
    UB = [1, 500*ones(1,16),5,5, 500*ones(1,17), 1];
    % Restrict variable i=0
    LB(i)=0;
    UB(i)=0;
    
    % 5 Starting points to estimate the restricted Loglikelihood 
    param_ST = [param0; paramMLE; 0.1*ones(1,37); [lambda0, 9*ones(1,16), gamma0, alpha0, beta0, beta, delta0]; param_paper];
    param_res = param_ST;
    temp = -Inf*ones(size(param_ST,1),1);
    
    for j = 1:size(param_ST,1)
        %fprintf('%d\n',(i-1)*size(param_ST,1)+j)
        param_st = param_ST(j,:);
        param_st(i)=0;

        [param_res(j,:), temp(j)] = fmincon(@(theta) GARCHX_Neg_Logl(theta, T, Y, X_s, X_m, Y0), param_st,...
               [zeros(1,17) 1 1 zeros(1,18)],1,...  % alpha + gamma <1
               [],[],LB,UB,[],options);
       
    end
    
    fprintf('%i %%\n',ceil(100*i/length(paramMLE)))
    [Res_NLogl(i), k] = min(temp);
    %paramMLE_restricted = param_res(k,:);

end

chi_test = 2*(Res_NLogl - Opt_NLogl);
pValue_LRT=1-cdf('chi2', chi_test', 1);

end

