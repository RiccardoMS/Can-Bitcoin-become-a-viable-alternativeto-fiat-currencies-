function eps2 = inference(X_s, X_m, Y, Y0, T, param)
% Assign optimal parameters
lambda = param(1:17);
gamma = param(18);
alpha = param(19);

beta = param(20:36);
delta = param(37);

% Allocate volatility, logreturns, and squared errors
sigma = zeros(T,1);
r = zeros(T,1);
eps2 = zeros(T,1);

% Compute volatility, logreturns, and squared errors
% t=1
omega = exp(lambda*X_s(1,:)');
sigma(1)   = sqrt(omega/(1-alpha-gamma)); %sigma is volatility - unconditional expectation
r(1) = beta*X_m(1,:)' + delta*Y0;
eps2(1)    = (Y(1) - r(1))^2;

% t=2:T
for t=2:T
    omega = exp(lambda*X_s(t,:)');
    sigma(t) = sqrt(omega + alpha*eps2(t-1) + gamma*sigma(t-1)^2);
    r(t) = beta*X_m(t,:)' + delta*Y(t-1);
    eps2(t) = (Y(t) - r(t))^2;
    
end

figure()
hold on
x = 1:length(r);
x_area = [x, fliplr(x)];
y_area = [(r-1.96*sigma)', fliplr((r+1.96*sigma)')];
fill(x_area, y_area, [1 0.7 0.7]);
plot(r+1.96*sigma, 'r-')
plot(r, 'r')
plot(r-1.96*sigma, 'r-')
plot(Y,'b')
legend(["95% CI" "Upper bound" "Estimated mean" "Lower bound" "BPI"])

figure()
hold on
x = 1:length(r);
x_area = [x, fliplr(x)];
y_area = [(r-1.96*sigma)', fliplr((r+1.96*sigma)')];
fill(x_area, y_area, [1 0.7 0.7]);
plot(r+1.96*sigma, 'r-')
plot(r, 'r')
plot(r-1.96*sigma, 'r-')
plot(Y,'b')
xlim([500 1000])
ylim([-0.7 0.7])

eps= Y-r;
figure()
plot(r,eps./sigma, 'b+', 'MarkerSize', 3)
xlabel("Fitted", 'interpreter', 'latex')
ylabel("Residuals",'interpreter', 'latex')


hold off


end

