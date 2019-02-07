function func = SVM_multiClass(a)
%Fit a binary classification SVM for each column of Y. Each column of Y
%must contain exactly 2 classes/unique values

%% Uses the settings and code from the Classification learner app on a Linear Discriminant model
switch a
    case 1; func = @podSVMGen;
    case 2; func = @podSVMApply;
    case 3; func = @podSVMReport;
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function classSVM = podSVMGen(Y,X,~)

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
predictorNames = X.Properties.VariableNames;
X_fit = table2array(X);
classSVM = cell(size(Y,2),1);

%fit a model for each column of Y (predict 1/0, true/false)
for i = 1:size(Y,2)
    Y_fit = table2array(Y(:,i));
    Y_fit(Y_fit==0.5)=1;
    rng(1) %For repeatability
    classSVM{i} = fitcsvm(X_fit,Y_fit,...
        'PredictorNames',predictorNames,'ResponseName',Y.Properties.VariableNames{i},...
        'KernelFunction','linear',...
        'Standardize',true,...
        'OutlierFraction',0.01);
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = podSVMApply(X,classSVM,~)

%Make a matrix to hold estimates
y_hat = zeros(size(X,1),length(classSVM));
X = table2array(X);

for i = 1:length(classSVM)
    %Make predictions on new data
    y_hat(:,i) = predict(classSVM{i},X);
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function podSVMReport(classSVM,~)
try
    classSVM
catch err
    disp('Error reporting the kNN Classification model');
end

end