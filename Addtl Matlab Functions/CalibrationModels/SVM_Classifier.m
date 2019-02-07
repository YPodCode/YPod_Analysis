function func = SVM_Classifier(a)
%Create a custom SVM for each class of Y
switch a
    case 1; func = @podSVMFit;
    case 2; func = @podSVMApply;
    case 3; func = @podSVMReport;
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function mdlobj = podSVMFit(Y,X,settingsSet)
mdlobj = cell(size(Y,2),1);

%In case we want to transform X, we can do it here (make sure to do the
%same things in the "apply" function
predictors = X;

%----

%Check for existing optimization for this pod
if length(settingsSet.fileList.colocation.reference.files.bytes)==1; reffileName = settingsSet.fileList.colocation.reference.files.name;
else; reffileName = settingsSet.fileList.colocation.reference.files.name{settingsSet.loops.i}; end
currentRef = split(reffileName,'.');
currentRef = currentRef{1};
% if settingsSet.isRef
%     clasName = 'ref';
% else
%     clasName = 'est';
% end
clasName = 'raw';
filename = [settingsSet.podList.podName{settingsSet.loops.j} clasName currentRef 'CSVMsave.mat'];
regPath = fullfile(settingsSet.outpath,filename);

%If SVM has already been optimized for this pod, can skip the optimization, which is really slow
if exist(regPath,'file')==2
    %Load the previous analysis
    load(regPath);
else
    %Make a temporary reference array that is a categorical vector with a
    %category for each combination of classes that were seen in training
    %data
    tempRef = table2array(Y);
    tempRef = num2str(tempRef);
    tempRef = cellstr(tempRef);
    tempRef = categorical(tempRef);
    
    %Optimize the hyperparameters for a multiclass SVM model for use later
    rng(123)
    Mdl = fitcecoc(predictors, tempRef, 'Learners', 'svm',...
        'OptimizeHyperparameters',{'BoxConstraint','KernelScale','KernelFunction'},...
        'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus',...
        'UseParallel',true,'Verbose',1, 'MaxObjectiveEvaluations',50));
    
    CSVMstruct=Mdl.HyperparameterOptimizationResults.XAtMinObjective;
    
    close all
    
    %save the fitted parameters for later runs
    save(char(regPath),'CSVMstruct');
end

%Fit a one vs all binary SVM for each class (column of Y) using the determined parameters
for i = 1:size(Y,2)
    %Extract just the column for this class
    response = table2array(Y(:,i));
    
    disp(['Fitting model with optimized hyperparameters for ' Y.Properties.VariableNames{i}]);
    
    %Catch differences between gaussian and linear/polynomial
    if isnan(CSVMstruct.KernelScale)
        rng(123)
        mdlobj{i} = fitcsvm(predictors,response,...
            'KernelFunction', char(CSVMstruct.KernelFunction),...
            'BoxConstraint',CSVMstruct.BoxConstraint);
    else
        rng(123)
        mdlobj{i} = fitcsvm(predictors,response,...
            'KernelFunction', char(CSVMstruct.KernelFunction),...
            'KernelScale',CSVMstruct.KernelScale,...
            'BoxConstraint',CSVMstruct.BoxConstraint);
    end
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = podSVMApply(X,mdlobj,~)

y_hat = zeros(size(X,1),length(mdlobj));

predictors = X;

%Make predictions on new data
for i = 1:length(mdlobj)
    [estimates, scores] = predict(mdlobj{i},predictors);
    %Transform the scores to range 0-1 using the sigmoid function
    y_hat(:,i) = sigmf(scores(:,2),[1 0]);
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function podSVMReport(clasfier,~)
try
    
catch err
    disp('Error reporting the kNN Classification model');
end

end