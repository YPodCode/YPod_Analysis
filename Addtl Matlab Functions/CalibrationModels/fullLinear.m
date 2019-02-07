function func = fullLinear(a)
%This fits a purely linear model using all columns of X as predictors
switch a
    case 1; func = @fullLinearGen;
    case 2; func = @fullLinearApply;
    case 3; func = @fullLinearReport;
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function mdl = fullLinearGen(Y,X,~)

X = table2array(X);
mdl = cell(size(Y,2),1);
for i = 1:size(Y,2)
    y = table2array(Y(:,i));
    
    tempmdl =fitlm(X,y,'linear');
    mdl{i} = compact(tempmdl);
    clear tempmdl
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = fullLinearApply(X,mdl,~)

X = table2array(X);
y_hat = zeros(size(X,1),length(mdl));
for i = 1:length(mdl)
    y_hat(:,i) = predict(mdl{i},X);
end

end
%--------------------------------------------------------------------------

%-------------Report relevant stats (coefficients, etc) about the model-------------
function fullLinearReport(fittedMdl,mdlStats,settingsSet)
fittedMdl
end