function func = podStepLM(a)
%% Fit a GLM using stepwise regression to select terms
switch a
    case 1; func = @podstepLMGen;
    case 2; func = @podstepLMApply;
    case 3; func = @podstepLMReport;
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function mdl = podstepLMGen(Y,X,~)

%xnames = X.Properties.VariableNames;
%ynames = Y.Properties.VariableNames;
%X = table2array(X);
%y = table2array(y);
%cats = categories(y);

mdl = cell(size(Y,2),1);
warning('off')%,'stats:glmfit:BadScaling'
for i = 1:size(Y,2)
    %Join tables for regression (default for stepwiseglm is that the last term is the response variable)
    XY = [X Y(:,i)];
    
    %Only use 25% of the data for initial model construction
    fitlist = randi(4,size(Y,1),1);
    fitlist = fitlist>1;
    XY = XY(fitlist,:);
    
    %Perform stepwise GLM regression
    rng(1)
    tempmdl = stepwiselm(XY, 'constant',... %Start with a constant model and add from there
        'ResponseVar',Y.Properties.VariableNames{i},... %Make sure we use the right response variable
        'upper','interactions',... %Maximum complexity is a full interactions model
        'Criterion','rsquared','PEnter',0.075); %Use the AIC to select additional terms
    mdl{i} = compact(tempmdl);
    clear tempmdl
end
warning('on')%,'stats:glmfit:BadScaling'

end

%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = podstepLMApply(X,mdl,~)

y_hat = zeros(size(X,1),length(mdl));
for i=1:length(mdl)
    %Make new predictions
    y_hat(:,i) = predict(mdl{i},X);
end
end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function podstepLMReport(mdl,~)
try
    plotDiagnostics(mdl)
catch
    warning('Error reporting the stepwise GLM model');
end

end