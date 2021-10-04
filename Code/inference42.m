function eps = inference42(X_s, X_m, y, y0, eps0, T, param)
% Assign optimal parameters
lambda = param(1:17);
gamma = param(18);
alpha = param(19);
beta = param(20:36);
delta_1 = param(37:40);
delta_2 = param(41:42);

% Allocate volatility, logreturns, and squared errors
sigma = zeros(T,1);
r = zeros(T,1);
eps = zeros(T,1);

% Compute volatility, logreturns, and squared errors
% t=1
omega = exp(lambda*X_s(1,:)');
sigma(1)   = sqrt(omega/(1-alpha-gamma)); %sigma is volatility - unconditional expectation
r(1) = beta*X_m(1,:)' + delta_1*y0 + delta_2*eps0;
eps(1)    = y(1) - r(1);

% t=2:T
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
    
 omega = exp(lambda*X_s(t,:)');
 sigma(t) = sqrt(omega + gamma*sigma(t-1)^2 + alpha*eps(t-1)^2 );
 r(t) = beta*X_m(t,:)' + delta_1*y0new' +delta_2*eps0new' ;
 eps(t) = y(t) - r(t);
end

figure()
plot(y)
hold on
plot(r+1.96*sigma, 'r-')
plot(r, 'r')
plot(r-1.96*sigma, 'r-')

figure()
hold on
x = 1:length(r);
x_area = [x, fliplr(x)];
y_area = [(r-1.96*sigma)', fliplr((r+1.96*sigma)')];
fill(x_area, y_area, [1 0.7 0.7]);
plot(r+1.96*sigma, 'r-')
plot(r, 'r')
plot(r-1.96*sigma, 'r-')
plot(y,'b')
xlim([500 1000])
ylim([-0.7 0.7])
end

