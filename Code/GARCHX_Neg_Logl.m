function[sum_loglike] = GARCHX_Neg_Logl(params, T, y, X_s, X_m, y0)

% This function computes the Gaussian log-likelihood for the GARCH(1,1)
% model

% INPUT: 
% - vector of unknown parameters: params (Nparam x 1)
% - Sample Size: T (1 x 1)
% - y: BPI price (Tx1)
% - Regressors for X_s (con colonna di 1) (Tx17)
% - Regressors for X_m (con colonna di 1) (Tx17)
% - Presample 
%
% OUTPUT: 
% - Total negative log-likelihood: sum_loglike (1 x 1)

% assign parameters
lambda = params(1:17);
gamma = params(18);
alpha = params(19);

beta = params(20:36);
delta = params(37);

% pre allocate variables
loglike = zeros(T,1); 
sigma = zeros(T,1);
r = zeros(T,1);
eps2 = zeros(T,1);   %Squared observations: eps2 (T x 1)

% loglike at time t=1
omega = exp(lambda*X_s(1,:)');
sigma(1)   = sqrt(omega/(1-gamma-alpha)); % sigma is volatility
r(1) = beta*X_m(1,:)' + delta*y0;
eps2(1)    = (y(1) - r(1))^2;
loglike(1) = log(sigma(1)) + eps2(1)/(2*sigma(1)^2);

for t=2:T

% 1. compute today's volatility     
omega = exp(lambda*X_s(t,:)');
sigma(t) = sqrt(omega + gamma*sigma(t-1)^2 + alpha*eps2(t-1) );
r(t) = beta*X_m(t,:)' + delta*y(t-1);
eps2(t) = (y(t) - r(t))^2;
% 2. construct logl-likelihood of today's observation
loglike(t) = log(sigma(t)) + eps2(t)/(2*sigma(t)^2);
    
end

% compute total log-likelihood
sum_loglike = sum(loglike)+T/2*log(2*pi);

end