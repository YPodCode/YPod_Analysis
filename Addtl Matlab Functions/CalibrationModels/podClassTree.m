function func = podClassTree(a)

switch a
    case 1; func = @podClassTreeGen;
    case 2; func = @podClassTreeApply;
    case 3; func = @podClassTreeReport;
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function mdl  = podClassTreeGen(Y,X,~)

%Need Y as a vector, assume first column of Y is the response variable
yvar = Y.Properties.VariableNames{1};
Y = table2array(Y(:,1));
assert(iscategorical(Y)||isnumeric(Y)||islogical(Y)||ischar(Y),['Y column ',yvar,' is not an appropriate variable type!']);

%Fit classification tree using all columns of X
tree = fitctree(X,Y,'PredictorSelection','interaction-curvature','Prune','on','ResponseName',yvar);

%Make the tree model more compact to reduce memory (still works but doesn't
%keep the training data with it any more)
tree = compact(tree);

%Export the model
mdl = tree;

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = podClassTreeApply(X,tree,~)

%Make predictions on new data
y_hat = predict(tree,X);

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function podClassTreeReport(fittedMdl,~)
try
    
catch err
    disp('Error reporting the ClassTree regression model');
end

end