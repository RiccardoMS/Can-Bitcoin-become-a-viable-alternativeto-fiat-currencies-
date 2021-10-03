function[sum_loglike] = GARCHX42_Neg_Logl(params, T, y, X_s, X_m, y0, eps0)

% This function computes the Gaussian log-likelihood for the ARMAX(4,2)-GARCH(1,1)
% model

% INPUT: 
% - vector of unknown parameters: params (Nparam x 1)
% - Sample Size: T (1 x 1)
% - y: BPI price (Tx1)
% - Regressors for X_s (con colonna di 1) (Tx17)
% - Regressors for X_m (con colonna di 1) (Tx17)
% - Presample :
%    - y0: 4x1 (BPI(1,2,3,4))
%    - eps0: 2x1 ( 
%
% OUTPUT: 
% - Total negative log-likelihood: sum_loglike (1 x 1)

% assign parameters
lambda = params(1:17);
gamma = params(18);
alpha = params(19);
beta = params(20:36);
delta_1 = params(37:40);
delta_2 = params(41:42);

% pre allocate variables
loglike = zeros(T,1); 
sigma = zeros(T,1);
r = zeros(T,1);
eps = zeros(T,1);   %Squared observations: eps2 (T x 1)

% loglike at time t=1
omega = exp(lambda*X_s(1,:)');
sigma(1)   = sqrt(omega/(1-gamma-alpha)); % sigma is volatility
r(1) = beta*X_m(1,:)' + delta_1*y0 + delta_2*eps0;
eps(1)    = (y(1) - r(1));
loglike(1) = log(sigma(1)) + eps(1)^2/(2*sigma(1)^2);

for t=2:T

 if t == 2
    y0new = [y(t-1) y0(3) y0(2) y0(1)];
    eps0new = [eps(t-1) eps0(2)];
 elseif t == 3
    y0new = [y(t-1) y(t-2) y0(3) y0(2)];
    eps0new = [eps(t-1) eps(t-2)];
 elseif t == 4
    y0new = [y(t-1) y(t-2) y(t-3) y0(3)];
    eps0new = [eps(t-1) eps(t-2)];
 else
    y0new = [y(t-1) y(t-2) y(t-3) y(t-4)];
    eps0new = [eps(t-1) eps(t-2)];
 end
    
% 1. compute today's volatility
omega = exp(lambda*X_s(t,:)');
sigma(t) = sqrt(omega + gamma*sigma(t-1)^2 + alpha*eps(t-1)^2 );
r(t) = beta*X_m(t,:)' + delta_1*y0new' +delta_2*eps0new';
eps(t) = (y(t) - r(t));
% 2. construct log-likelihood of today's observation
loglike(t) = log(sigma(t)) + eps(t)^2/(2*sigma(t)^2);
    
end

% compute total log-likelihood
sum_loglike = sum(loglike)+T/2*log(2*pi);

end

