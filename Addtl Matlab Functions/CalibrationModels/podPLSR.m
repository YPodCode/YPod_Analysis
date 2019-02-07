function func = podPLSR(a)
%Use partial least squares to select components that maximize the explained
%variance in both X and Y

switch a
    case 1; func = @PartLeastSqrsRegressGen;
    case 2; func = @PartLeastSqrsRegressApply;
end

end

function [mdl, y_hat_fin] = PartLeastSqrsRegressGen(Y,X,settingsSet)

xnames = X.Properties.VariableNames;
ynames = Y.Properties.VariableNames;
try
    t = X.telapsed;
catch 
    t = 1:size(X,1);
end
X = table2array(X);
Y = table2array(Y);
ymed = zeros(size(Y,2),1);
ystd = zeros(size(Y,2),1);
for i = 1:size(Y,2)
    ymed(i) = median(Y(:,i));
    ystd(i) = std(Y(:,i));
    Y(:,i) = (Y(:,i)-ymed(i))/ystd(i);
end
n_try = min(12,size(X,2));
[XL,yl,XS,YS,beta,PCTVAR,MSE,stats] = plsregress(X,Y,n_try);

%Plot relevant statistics
figure('Position',[0 0 1000 1000]);
subplot(2,2,1) %PCT variance plot
yyaxis left
plot(1:n_try,cumsum(100*PCTVAR(2,:)),'-bo'); %Cumulative pct variance explained in the response
xlabel('Number of PLS components');
ylabel('Percent Variance Explained in y');
yyaxis right
[axes,h1,h2] = plotyy(0:n_try,MSE(1,:),0:n_try,MSE(2,:));
set(h1,'Marker','o') %Predictor fit error
set(h2,'Marker','.') %Response fit error
ylabel('MSE for X (o) and Y (.)');
grid on

subplot(2,2,2)  %Plot vs reference
y_hat = [ones(size(X,1),1) X]*beta;
plot(t,Y,t,y_hat,'o');
ylabel('Reference'); 
xlabel('Fitted'); 
title('Normalized Estimates and Concentrations');

subplot(2,2,3:4) %Predictor weights
a = stem(1:size(X,2),stats.W);
colors = jet(n_try);
for z = 1:n_try
    a(z).Color = colors(z,:);
end
grid on;
xlabel('Predictor'); 
xticks(1:size(X,2));xtickangle(45);
xticklabels(xnames);
ylabel('Weight');
legend(strtrim(cellstr(num2str([1:n_try]'))),'Location','best');


%%  Re-run with reduced number of components
keepComps = input('How many components would you like to use?   ');
close(gcf)

%Re-fit using those values
[~,~,~,~,beta_red,PCTVAR,MSE,stats] = plsregress(X,Y,keepComps);
y_hat_fin = [ones(size(X,1),1) X]*beta_red;
for i = 1:size(Y,2)
    Y(:,i) = Y(:,i)*ystd(i)+ymed(i);
    y_hat_fin(:,i) = y_hat_fin(:,i)*ystd(i)+ymed(i);
end

%% Plot relevant statistics
figure('Position',get( groot, 'Screensize' ));
subplot(2,2,1) %PCT variance plot
yyaxis left
plot(1:keepComps,cumsum(100*PCTVAR(2,:)),'-bo'); %Cumulative pct variance explained in the response
xlabel('Number of PLS components');
ylabel('Percent Variance Explained in y');
yyaxis right
[axes,h1,h2] = plotyy(0:keepComps,MSE(1,:),0:keepComps,MSE(2,:));
set(h1,'Marker','o') %Predictor fit error
set(h2,'Marker','.') %Response fit error
ylabel('MSE for X (o) and Y (.)');
grid on

subplot(2,2,2)  %Plot vs reference
plot(t,Y,t,y_hat_fin,'o');
ylabel('Reference'); 
xlabel('Fitted'); 

subplot(2,2,3:4) %Predictor weights
a = stem(1:size(X,2),stats.W);
colors = jet(keepComps);
for z = 1:keepComps
    a(z).Color = colors(z,:);
end
grid on;
xlabel('Predictor'); 
xticks(1:size(X,2));xtickangle(45);
xticklabels(xnames);
ylabel('Weight');
legend(strtrim(cellstr(num2str([1:keepComps]'))),'Location','best');

%Allow saving out this plot
if settingsSet.savePlots
    currentPod = settingsSet.podList.podName{settingsSet.loops.j};
    nref   = length(settingsSet.fileList.colocation.reference.files.bytes);
    if nref==1; reffileName = settingsSet.fileList.colocation.reference.files.name;
    else; reffileName = settingsSet.fileList.colocation.reference.files.name{settingsSet.loops.i}; end
    currentRef = split(reffileName,'.');
    currentRef = currentRef{1};
    nfold = settingsSet.loops.kk;
    temppath = [currentPod '_' currentRef 'fold' num2str(nfold) '_PLS_stats'];
    temppath = fullfile(settingsSet.outpath,temppath);
    saveas(gcf,temppath,'jpeg');
    clear temppath
    close(gcf)
end

%% Plot y vs yhat
figure('Position',get( groot, 'Screensize' ));
for i=0:size(Y,2)-1
    c = mod(i,4)+1;
    if i>3;r=2;else;r=1;end
    disp([num2str(r) ' ' num2str(c) ';']);
    g(r,c)=gramm('x',Y(:,i+1),'y',y_hat_fin(:,i+1));
    g(r,c).geom_point();
    errstd = round(std(Y(:,i+1)-y_hat_fin(:,i+1)),1);
    g(r,c).stat_cornerhist('aspect',0.8,'edges',(-5*errstd):errstd/2:(5*errstd));
    g(r,c).geom_abline();
    g(r,c).set_title(ynames{i+1});
    g(r,c).axe_property('TickDir','out','XGrid','on','Ygrid','on','GridColor',[0.5 0.5 0.5]);
    g(r,c).set_names('x','Reference','y','Estimate');
end
g.draw();

%Allow saving out this plot
if settingsSet.savePlots
    temppath = [currentPod '_' currentRef 'fold' num2str(nfold) '_PLS_fit'];
    temppath = fullfile(settingsSet.outpath,temppath);
    saveas(gcf,temppath,'jpeg');
    clear temppath
    close(gcf)
end

%% Save out necessary data to apply to new data
mdl = {beta_red, ymed, ystd};
end

function y_hat = PartLeastSqrsRegressApply(X,mdl,~)

%Extract fitted model
beta = mdl{1};
ymed = mdl{2};
ystd = mdl{3};

%Use the weights to apply the fitted model to new X
X = table2array(X);
y_hat = [ones(size(X,1),1) X]*beta;

for i = 1:size(y_hat,2)
    y_hat(:,i) = y_hat(:,i)*ystd(i)+ymed(i);
end

end