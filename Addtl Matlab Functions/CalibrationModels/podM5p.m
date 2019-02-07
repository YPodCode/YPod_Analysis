function func = podM5p(a)

switch a
    case 1; func = @podM5pGen;
    case 2; func = @podM5pApply;
    case 3; func = @podM5pReport;
end

end

function [mdl, y_hat] = podM5pGen(Y,X,~)

%Need as an array
X = table2array(X);
Y = table2array(Y);

%Fit model tree
params = m5pparams2('modelTree',true,'aggressivePruning',true);
isBinCat = false(1,size(X,2)); %None of the variables are catagorical
model = m5pbuild(X,Y,params,isBinCat);

% %Report on model
% m5pprint(model);
% m5pplot(model);

%Report cross-validation
%m5pcv(X,Y,params)

%Make predictions
y_hat = m5ppredict(model,X);

%Export the model
mdl = model;

end

function y_hat = podM5pApply(X,mdl,~)

%Need as an array
X = table2array(X);

%Make predictions
y_hat = m5ppredict(mdl,X);

end

function podM5pReport(fittedMdl,~)
try
    m5pprint(fittedMdl);
    m5pplot(fittedMdl);
catch err
    disp('Error reporting this model');
end

end