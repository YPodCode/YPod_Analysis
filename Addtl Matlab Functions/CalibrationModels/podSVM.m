function func = podSVM(a)

switch a
    case 1; func = @podSVMGen;
    case 2; func = @podSVMApply;
    case 3; func = @podSVMReport;
end

end

function [mdl, y_hat] = podSVMGen(Y,X,~)

X = table2array(X);
Y = table2array(Y);

%Make logarithmically spaced set of lambdas to try
Lambda = logspace(-5,-1,15);

%Can make the process slower but more optimized with this:
CVMdl = fitrlinear(X,Y,'Lambda',Lambda,'KFold',5,...
    'Learner','leastsquares','Solver','sparsa','Regularization','lasso');
mse = kfoldLoss(CVMdl);
Mdl = fitrlinear(X,Y,'Lambda',Lambda,...
    'Learner','leastsquares','Solver','sparsa','Regularization','lasso');
numNZCoeff = sum(Mdl.Beta~=0);

figure;
[h,hL1,hL2] = plotyy(log10(Lambda),log10(mse),...
    log10(Lambda),log10(numNZCoeff)); 
hL1.Marker = 'o';
hL2.Marker = 'o';
ylabel(h(1),'log_{10} MSE')
ylabel(h(2),'log_{10} nonzero-coefficient frequency')
xlabel('log_{10} Lambda')
hold off

%mdl = fitrlinear(X,Y);
y_hat = predict(mdl,X);
end

function y_hat = podSVMApply(X,mdl,~)

X = table2array(X);
y_hat = predict(mdl,X);


end

function podSVMReport(mdl,~)
linmdl = mdl{1};
try
    linmdl.Coefficients.Estimate'
catch err
    disp('Error reporting this model');
end

end