function best_guess = Sensitivity_Analysis(X_s, X_m, Y0, Y, T,LB, UB, options, Est_Mdl)

% In this file we aim at finding an effective initial guess for the Maximum
% Likelihood problem, to find the correct regressors of the ARMAX-GARCHX 
% model.  

% We save the found coefficients
lambda0 = log(Est_Mdl.Variance.Constant);
gamma0 = cell2mat(Est_Mdl.Variance.GARCH);
alpha0 = cell2mat(Est_Mdl.Variance.ARCH);
delta0 = cell2mat(Est_Mdl.AR);
beta0 = Est_Mdl.Constant;
beta = Est_Mdl.Beta;

%% Sensitivity Analysis
% We perform a sensitivity analysis considering as initial guess 13
% different possible initial guesses for the regressors of the volatility
% The other regressors are guessed starting from the ARMAX-GARCH(1,1) model

% We allocate a matrix where to save the estimated parameters 
paramMLE_mat = zeros(13,37);

% In position from 2 to 17 we have the several initial guesses for the
% volatility regressors of the exogenous part
guesses=[-1, 0.1 3 5 7 9 15 5 8 4 4.5 5.5 5];

%1
param0 = [lambda0, guesses(1)*ones(1,16), gamma0, alpha0, beta0, beta, delta0];
paramMLE = fmincon(@(theta) GARCHX_Neg_Logl(theta, T, Y, X_s, X_m, Y0), param0,...
           [zeros(1,17) 1 1 zeros(1,18)],1,...  % alpha + gamma <1 & delta<1
           [],[],LB,UB,[],options);
paramMLE_mat(1,:)= paramMLE;
disp('1/13')

%2
param0 = [lambda0, guesses(2)*ones(1,16), gamma0, alpha0, beta0, beta, delta0];
paramMLE = fmincon(@(theta) GARCHX_Neg_Logl(theta, T, Y, X_s, X_m, Y0), param0,...
           [zeros(1,17) 1 1 zeros(1,18)],1,...  % alpha + gamma <1 & delta<1
           [],[],LB,UB,[],options);
paramMLE_mat(2,:)= paramMLE;
disp('2/13')

%3
param0 = [lambda0, guesses(3)*ones(1,16), gamma0, alpha0, beta0, beta, delta0];
paramMLE = fmincon(@(theta) GARCHX_Neg_Logl(theta, T, Y, X_s, X_m, Y0), param0,...
           [zeros(1,17) 1 1 zeros(1,18)],1,...  % alpha + gamma <1 & delta<1
           [],[],LB,UB,[],options);
paramMLE_mat(3,:)= paramMLE;
disp('3/13')

%4
param0 = [lambda0, guesses(4)*ones(1,16), gamma0, alpha0, beta0, beta, delta0];
paramMLE = fmincon(@(theta) GARCHX_Neg_Logl(theta, T, Y, X_s, X_m, Y0), param0,...
           [zeros(1,17) 1 1 zeros(1,18)],1,...  % alpha + gamma <1 & delta<1
           [],[],LB,UB,[],options);
paramMLE_mat(4,:)= paramMLE;
disp('4/13')

%5
param0 = [lambda0, guesses(5)*ones(1,16), gamma0, alpha0, beta0, beta, delta0];
paramMLE = fmincon(@(theta) GARCHX_Neg_Logl(theta, T, Y, X_s, X_m, Y0), param0,...
           [zeros(1,17) 1 1 zeros(1,18)],1,...  % alpha + gamma <1 & delta<1
           [],[],LB,UB,[],options);
paramMLE_mat(5,:)= paramMLE;
disp('5/13')

%6
param0 = [lambda0, guesses(6)*ones(1,16), gamma0, alpha0, beta0, beta, delta0];
paramMLE = fmincon(@(theta) GARCHX_Neg_Logl(theta, T, Y, X_s, X_m, Y0), param0,...
           [zeros(1,17) 1 1 zeros(1,18)],1,...  % alpha + gamma <1 & delta<1
           [],[],LB,UB,[],options);
paramMLE_mat(6,:)= paramMLE;
disp('6/13')

%7
param0 = [lambda0, guesses(7)*ones(1,16), gamma0, alpha0, beta0, beta, delta0];
paramMLE = fmincon(@(theta) GARCHX_Neg_Logl(theta, T, Y, X_s, X_m, Y0), param0,...
           [zeros(1,17) 1 1 zeros(1,18)],1,...  % alpha + gamma <1 & delta<1
           [],[],LB,UB,[],options);
paramMLE_mat(7,:)= paramMLE;
disp('7/13')

%8
param0 = [lambda0, guesses(8)*ones(1,16).*sin(cumsum(ones(1,16))), gamma0, alpha0, beta0, beta, delta0];
paramMLE = fmincon(@(theta) GARCHX_Neg_Logl(theta, T, Y, X_s, X_m, Y0), param0,...
           [zeros(1,17) 1 1 zeros(1,18)],1,...  % alpha + gamma <1 & delta<1
           [],[],LB,UB,[],options);
paramMLE_mat(8,:)= paramMLE;
disp('8/13')

%9
param0 = [lambda0, guesses(9)*ones(1,16).*cos(cumsum(ones(1,16))), gamma0, alpha0, beta0, beta, delta0];
paramMLE = fmincon(@(theta) GARCHX_Neg_Logl(theta, T, Y, X_s, X_m, Y0), param0,...
           [zeros(1,17) 1 1 zeros(1,18)],1,...  % alpha + gamma <1 & delta<1
           [],[],LB,UB,[],options);
paramMLE_mat(9,:)= paramMLE;
disp('9/13')

%10
param0 = [lambda0, guesses(10)*ones(1,16), gamma0, alpha0, beta0, beta, delta0];
paramMLE = fmincon(@(theta) GARCHX_Neg_Logl(theta, T, Y, X_s, X_m, Y0), param0,...
           [zeros(1,17) 1 1 zeros(1,18)],1,...  % alpha + gamma <1 & delta<1
           [],[],LB,UB,[],options);
paramMLE_mat(10,:)= paramMLE;
disp('10/13')

%11
param0 = [lambda0, guesses(11)*ones(1,16), gamma0, alpha0, beta0, beta, delta0];
paramMLE = fmincon(@(theta) GARCHX_Neg_Logl(theta, T, Y, X_s, X_m, Y0), param0,...
           [zeros(1,17) 1 1 zeros(1,18)],1,...  % alpha + gamma <1 & delta<1
           [],[],LB,UB,[],options);
paramMLE_mat(11,:)= paramMLE;
disp('11/13')

%12
param0 = [lambda0, guesses(12)*ones(1,16), gamma0, alpha0, beta0, beta, delta0];
paramMLE = fmincon(@(theta) GARCHX_Neg_Logl(theta, T, Y, X_s, X_m, Y0), param0,...
           [zeros(1,17) 1 1 zeros(1,18)],1,...  % alpha + gamma <1 & delta<1
           [],[],LB,UB,[],options);
paramMLE_mat(12,:)= paramMLE;
disp('12/13')

%13
param0 = [lambda0, guesses(13)*ones(1,16) + sin(cumsum(ones(1,16))), gamma0, alpha0, beta0, beta, delta0];
paramMLE = fmincon(@(theta) GARCHX_Neg_Logl(theta, T, Y, X_s, X_m, Y0), param0,...
           [zeros(1,17) 1 1 zeros(1,18)],1,...  % alpha + gamma <1 & delta<1
           [],[],LB,UB,[],options);
paramMLE_mat(13,:)= paramMLE;
disp('13/13')

% Plot of the convergence points
paramMLE_mat
% We recognize three different convergene points which are found by several
% optimization instances, plus some other points to which the algorithm
% converged just for one initial guess

% To establish which of the found convergence point maximizes the
% Likelihood, we find the minimum negative loglikelihood achieved 
f_val = zeros(13,1);
for i = 1:13
    f_val(i) = GARCHX_Neg_Logl(paramMLE_mat(i,:), T, Y, X_s, X_m, Y0);
end

[minimum, index] = min(f_val);
minimum
% Initial guesses 1-12 converge more or less to the same point which 
% maximizes the Likelihood among the ones found. We cannot be sure that 
% this is the Maximum of the loglikelihood, as the problem optimizes 37 
% different parameters, but our extensive sensitivity analysis let us 
% suppose that this can be a good choice

% Find the best initial guess
best_guess=guesses(index);

end

