function plot_ACF_PACF(e, plot_name)
% Plot the ACF and PACF of the residuals/squared residuals
% e: residuals/squared residuals
% plot_name: title of the plot

% Plot ACF of squared residuals
figure()
subplot(1,2,1)
autocorr(e)
ylabel('')
xlabel('')
ylim([-0.1 1.1])
var_name=strcat('ACF', plot_name);
title(var_name,'interpreter','latex')
set(gca,'FontSize',20)

% Plot PACF of squared residuals
subplot(1,2,2)
parcorr(e)
ylabel('')
xlabel('')
ylim([-0.1 1.1])
var_name=strcat('PACF' , plot_name);
title(var_name,'interpreter','latex')
set(gca,'FontSize',20)

end