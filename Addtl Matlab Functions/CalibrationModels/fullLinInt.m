function func = fullLinInt(a)
%This fits a purely linear model using all columns of X as predictors and
%also includes interaction terms between each variable
switch a
    case 1; func = @fullLinInteractGen;
    case 2; func = @fullLinInteractApply;
    case 3; func = @fullLinInteractReport;
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function mdl = fullLinInteractGen(Y,X,~)

X = table2array(X);
mdl = cell(size(Y,2),1);
for i = 1:size(Y,2)
    y = table2array(Y(:,i));
    
    tempmdl =fitlm(X,y,'interactions');
    mdl{i} = compact(tempmdl);
    clear tempmdl
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = fullLinInteractApply(X,mdl,~)

X = table2array(X);
y_hat = zeros(size(X,1),length(mdl));
for i = 1:length(mdl)
    y_hat(:,i) = predict(mdl{i},X);
end
end
%--------------------------------------------------------------------------

%-------------Report relevant stats (coefficients, etc) about the model-------------
function fullLinInteractReport(fittedMdl,mdlStats,settingsSet)
fittedMdl
end