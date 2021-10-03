function [paramMLE_restricted,pValue_LRT] = Variable_Selection(X_s, X_m, Y0, Y, T, Opt_NLogl ,Est_Mdl, paramMLE, param0, param_paper)


% We save the found coefficients
lambda0 = log(Est_Mdl.Variance.Constant);
gamma0 = cell2mat(Est_Mdl.Variance.GARCH);
alpha0 = cell2mat(Est_Mdl.Variance.ARCH);
delta0 = cell2mat(Est_Mdl.AR);
beta0 = Est_Mdl.Constant;
beta = Est_Mdl.Beta;

LB = [-100, -500*ones(1,16),-5,-5, -500*ones(1,17), -1];     
UB = [1, 500*ones(1,16),5,5, 500*ones(1,17), 1];

options = optimoptions('fmincon','UseParallel',false,'Display','off',...
    'MaxFunEvals', 500000,'algorithm','interior-point', ...
    'TolFun' ,1e-12,'TolX',1e-12,'MaxIter', 2000, 'StepTolerance', 1e-80, 'ObjectiveLimit', -1e50);

% We perform a LRT on a set of indeces in order to exclude them from the
% model. The choice is performed on the basis of one by one LRT
ind_1 = [4, 6, 7, 8, 9, 17, 21, 22, 23, 26, 28, 30, 31, 32, 33, 34, 35, 36];
LB(ind_1)=0;
UB(ind_1)=0;

param_ST = [param0; paramMLE; 0.1*ones(1,37); [lambda0, 9*ones(1,16), gamma0, alpha0, beta0, beta, delta0]; param_paper];
param_ST(:, ind_1) = 0;
param_res = param_ST;
temp = zeros(size(param_ST,1),1);

for j = 1:size(param_ST,1)
    %fprintf('%d\n',(i-1)*size(param_ST,1)+j)
    param_st = param_ST(j,:);

    [param_res(j,:), temp(j)] = fmincon(@(theta) GARCHX_Neg_Logl(theta, T, Y, X_s, X_m, Y0), param_st,...
           [zeros(1,17) 1 1 zeros(1,18)],1,...  % alpha + gamma <1
           [],[],LB,UB,[],options);

end

[Res_NLogl_1, k] = min(temp);
paramMLE_restricted = param_res(k,:);

chi_test = 2*(Res_NLogl_1 - Opt_NLogl);
disp("pValue of group significtivity");
pValue_1=1-cdf('chi2', chi_test', length(ind_1))
% pValue = 0.561
%%
% We perform a LRT of the remaining variables
Res_NLogl = paramMLE;
% LRT on remaining variable
for i = 1:length(paramMLE)
    % if/else to perform the LRT only on survived variables
    if sum(ind_1==i)
        Res_NLogl(i)=Res_NLogl_1;
    else

        LB_temp = LB(i);
        UB_temp = UB(i);
        ST_temp = param_ST(:,i);
        % Restrict Variable to 0
        LB(i)=0;
        UB(i)=0;
        param_ST(:,i) = 0;
        temp = -Inf*ones(size(param_ST,1),1);
        for j = 1:size(param_ST,1)
            %fprintf('%d\n',(i-1)*size(param_ST,1)+j)
            param_st = param_ST(j,:);
            param_st(i)=0;

            [param_res(j,:), temp(j)] = fmincon(@(theta) GARCHX_Neg_Logl(theta, T, Y, X_s, X_m, Y0), param_st,...
                   [zeros(1,17) 1 1 zeros(1,18)],1,...  % alpha + gamma <1
                   [],[],LB,UB,[],options);

        end
        
        LB(i)=LB_temp;
        UB(i)=UB_temp;
        param_ST(:,i) = ST_temp;

        fprintf('%f %%\n',100*i/length(paramMLE))
        Res_NLogl(i) = min(temp);
        %paramMLE_restricted = param_res(k,:);
    end

end

chi_test = 2*(Res_NLogl - Res_NLogl_1);
pValue_LRT=1-cdf('chi2', chi_test', 1);
