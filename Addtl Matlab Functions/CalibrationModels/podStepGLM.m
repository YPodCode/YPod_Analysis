function func = podStepGLM(a)
%% Fit a GLM using stepwise regression to select terms
%Note: this will not work with Y data that are less than or equal to 0, so
%those rows are removed (if present)

switch a
    case 1; func = @podstepGLMGen;
    case 2; func = @podstepGLMApply;
    case 3; func = @podstepGLMReport;
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function mdl = podstepGLMGen(Y,X,~)

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
    %Remove rows where Y is less than or equal to 0
    XY(any(table2array(Y)<=0,2),:)=[];
    
%     %Only use 25% of the data for initial model construction to speed up
%     fitlist = randi(4,size(Y,1),1);
%     fitlist = fitlist>1;
%     XY = XY(fitlist,:);
    
    %Perform stepwise GLM regression
    tempmdl = stepwiseglm(XY, 'constant',... %Start with a constant model and add from there
        'ResponseVar',Y.Properties.VariableNames{i},... %Make sure we use the right response variable
        'upper','interactions',... %Maximum complexity is a full interactions model
        'Distribution','normal',.... %Assume Y follows lognormal distribution
        'Link','log',... %Use a log link function
        'Criterion','rsquared'); %Use the R^2 to select additional terms
    mdl{i} = compact(tempmdl);
    clear tempmdl
end
warning('on')%,'stats:glmfit:BadScaling'

end

%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = podstepGLMApply(X,mdl,~)

y_hat = zeros(size(X,1),length(mdl));
for i=1:length(mdl)
    %Make new predictions
    y_hat(:,i) = predict(mdl{i},X);
end
end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function podstepGLMReport(mdl,~)
try
    plotDiagnostics(mdl)
catch
    warning('Error reporting the stepwise GLM model');
end

end