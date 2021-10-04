function plot_dependent_var(y, plot_name)
% Plot the dependent variable, its ACF and its PACF
% y: dependent variable
% plot_name: title of the plot

figure()
subplot(2,2,1:2)
plot(y,'linewidth',1.2)
grid on
var_name=strcat(plot_name);
title(var_name,'interpreter','latex')
xlim([0 length(y)])
set(gca,'FontSize',20)

subplot(2,2,3)
autocorr(y)
var_name=strcat('ACF of', plot_name);
title(var_name,'interpreter','latex')
ylabel('')
xlabel('')
ylim([-0.1 1.1])
set(gca,'FontSize',20)

subplot(2,2,4)
parcorr(y)
ylabel('')
xlabel('')
ylim([-0.1 1.1])
set(gca,'FontSize',20)
var_name=strcat('PACF of', plot_name);
title(var_name,'interpreter','latex')


end