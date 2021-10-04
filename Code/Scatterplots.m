%% Scatterplots of the dataset
% Run to plot all the scatterplots

%% Currencies
figure()

subplot(2,2,1)
plot(X(:,1),Y,'bo')
xlabel('CNY/USD','interpreter','latex')
ylabel('BPI','interpreter','latex')
set(gca,'FontSize',10)

subplot(2,2,2)
plot(X(:,2),Y,'bo')
xlabel('EUR/USD','interpreter','latex')
ylabel('BPI','interpreter','latex')
set(gca,'FontSize',10)

subplot(2,2,3)
plot(X(:,3),Y,'bo')
xlabel('JPY/USD','interpreter','latex')
ylabel('BPI','interpreter','latex')
set(gca,'FontSize',10)

subplot(2,2,4)
plot(X(:,4),Y,'bo')
xlabel('GOLD/USD','interpreter','latex')
ylabel('BPI','interpreter','latex')
set(gca,'FontSize',10)


%% Stock indexes
figure()

subplot(2,2,1)
plot(X(:,5),Y,'bo')
xlabel('SHANGAICompInd','interpreter','latex')
ylabel('BPI','interpreter','latex')
set(gca,'FontSize',10) 

subplot(2,2,2)
plot(X(:,6),Y,'bo')
xlabel('SP500','interpreter','latex')
ylabel('BPI','interpreter','latex')
set(gca,'FontSize',10)

subplot(2,2,3)
plot(X(:,7),Y,'bo')
xlabel('EUROSTOCK','interpreter','latex')
ylabel('BPI','interpreter','latex')
set(gca,'FontSize',10)

subplot(2,2,4)
plot(X(:,8),Y,'bo')
xlabel('NIKKEI','interpreter','latex')
ylabel('BPI','interpreter','latex')
set(gca,'FontSize',10)


%% 10ybond
figure()

subplot(2,2,1)
plot(X(:,9),Y,'bo')
xlabel('CHINA10Y','interpreter','latex')
ylabel('BPI','interpreter','latex')
set(gca,'FontSize',20)

subplot(2,2,2)
plot(X(:,10),Y,'bo')
xlabel('USA10Y','interpreter','latex')
ylabel('BPI','interpreter','latex')
set(gca,'FontSize',20)

subplot(2,2,3)
plot(X(:,11),Y,'bo')
xlabel('GERMANY10Y','interpreter','latex')
ylabel('BPI','interpreter','latex')
set(gca,'FontSize',20)

subplot(2,2,4)
plot(X(:,12),Y,'bo')
xlabel('JAPAN10Y','interpreter','latex')
ylabel('BPI','interpreter','latex')
set(gca,'FontSize',20)


%% Interbank rates
figure()

subplot(2,2,4)
plot(X(:,13),Y,'bo')
xlabel('3MShibor','interpreter','latex')
ylabel('BPI','interpreter','latex')
set(gca,'FontSize',10)

subplot(2,2,1)
plot(X(:,14),Y,'bo')
xlabel('US3MLibor','interpreter','latex')
ylabel('BPI','interpreter','latex')
set(gca,'FontSize',10)

subplot(2,2,2)
plot(X(:,15),Y,'bo')
xlabel('EUR3MLibor','interpreter','latex')
ylabel('BPI','interpreter','latex')
set(gca,'FontSize',10)

subplot(2,2,3)
plot(X(:,16),Y,'bo')
xlabel('JPY3MLibor','interpreter','latex')
ylabel('BPI','interpreter','latex')
set(gca,'FontSize',10)


%% time
figure()

plot(X(:,17),Y,'bo')
xlabel('Time','interpreter','latex')
ylabel('BPI','interpreter','latex')
set(gca,'FontSize',10)