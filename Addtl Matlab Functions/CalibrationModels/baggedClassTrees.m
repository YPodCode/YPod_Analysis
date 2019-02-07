function func = baggedClassTrees(a)
%% Uses bagging to create an ensemble of classification trees
switch a
    case 1; func = @bagClTreeFit;
    case 2; func = @bagClTreeApply;
    case 3; func = @bagClTreeReport;
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function mdlobj = bagClTreeFit(Y,X,settingsSet)

%%
%Check for existing optimization for this pod and reference
if length(settingsSet.fileList.colocation.reference.files.bytes)==1; reffileName = settingsSet.fileList.colocation.reference.files.name;
else; reffileName = settingsSet.fileList.colocation.reference.files.name{settingsSet.loops.i}; end
currentRef = split(reffileName,'.');
currentRef = currentRef{1};

%This is a workaround for my VOC stuff because models were fit in a few
%ways that was weird
% if settingsSet.isRef
%     clasName = 'ref';
% else
%     clasName = 'est';
% end
clasName = 'raw';
filename = [settingsSet.podList.podName{settingsSet.loops.j} clasName currentRef '_RFCsave.mat'];
regPath = fullfile(settingsSet.outpath,filename);

%% If RF has already been optimized for this pod, can skip the optimization, which is really slow
if exist(regPath,'file')==2
    %Load the previous analysis
    load(regPath);
else
    maxMinLS = max(2,floor(size(X,1)/2));
    minLS = optimizableVariable('minLS',[5,maxMinLS],'Type','integer');
    nTree = optimizableVariable('nTree',[10,1000],'Type','integer');
    maxNSplit = optimizableVariable('nSplit',[1,max(2,size(X,1)-1)],'Type','integer');
    hyperparametersRF = [minLS; maxNSplit];
    
    %Optimize hyperparameters on a classification tree that sees
    %all classes at once
    
    %Make a temporary reference array that is a categorical vector with a
    %category for each combination of classes that were seen in training
    %data
    if ~iscategorical(table2array(Y))
    tempRef = table2array(Y);
    tempRef = num2str(tempRef);
    tempRef = cellstr(tempRef);
    tempRef = categorical(tempRef);
    tempRef = array2table(tempRef,'VariableNames',{'RefCat'});
    else
        tempRef = Y;
    end
    C = [X tempRef];
    
    rng(42)
    results = bayesopt(@(params)oobErrRF(params,C),hyperparametersRF,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'Verbose',0,'UseParallel', true,'NumSeedPoints',10,'MaxObjectiveEvaluations',50);
    close all %Close the optimization plots
    
    %Get the optimized hyperparameters
    rfcstruct = results.XAtMinEstimatedObjective;
    
    %Make and save variable importance plots
    if settingsSet.savePlots
        for i = 1:size(Y,2)
            C = [X Y(:,i)];
            rng(42)
            Mdl = TreeBagger(500,C,Y.Properties.VariableNames{i},...
                'OOBPrediction','on','OOBPredictorImportance','on',...
                'MinLeafSize',rfcstruct.minLS,'MaxNumSplits',rfcstruct.nSplit,...
                'PredictorSelection','interaction-curvature');
            
            imp = Mdl.OOBPermutedPredictorDeltaError;
            figure;
            bar(imp);
            title([Y.Properties.VariableNames{i} ' Curvature Test']);
            ylabel('Predictor importance estimates');
            xlabel('Predictors');
            h = gca;
            xticks(h,1:length(Mdl.PredictorNames));
            h.XTickLabel = Mdl.PredictorNames;
            h.XTickLabelRotation = 45;
            h.TickLabelInterpreter = 'none';
            
            
            temppath = [settingsSet.podList.podName{settingsSet.loops.j} Y.Properties.VariableNames{i} clasName currentRef '_varImp'];
            temppath = fullfile(settingsSet.outpath,temppath);
            saveas(gcf,temppath,'jpeg');
            clear temppath
            close(gcf)
        end
    end
    
    %save the fitted parameters for later runs
    save(char(regPath),'rfcstruct');
end

%% Now go through and apply optimized hyperparameters to fit trees to predict the presence of each column of Y (to allow for more than 1 to be true at a time)
mdlobj = cell(size(Y,2),1);
for i = 1:size(Y,2)
    C = [X Y(:,i)];
    rng(42)
    Mdl = TreeBagger(500,C,Y.Properties.VariableNames{i},...
        'MinLeafSize',rfcstruct.minLS,'MaxNumSplits',rfcstruct.nSplit,...
        'PredictorSelection','interaction-curvature');
    mdlobj{i} = Mdl;

end

end

function testErr = oobErrRF(params,C)
%oobErrRF Trains random forest and estimates out-of-bag quantile error
%   oobErr trains a random forest of 300 regression trees using the
%   predictor data in X and the parameter specification in params, and then
%   returns the out-of-bag quantile error based on the median. X is a table
%   and params is an array of OptimizableVariable objects corresponding to
%   the minimum leaf size and number of predictors to sample at each node.

%ntrain = size(C,1) - floor(size(C,1)/3);
valList = repmat([zeros(15,1);ones(30,1)],ceil(size(C,1)/45),1);
valList = valList(1:size(C,1),1);
yname = C.Properties.VariableNames{end};

rng(42)
randomForest = TreeBagger(500,C(valList==1,:),yname,...
    'OOBPrediction','on','MinLeafSize',params.minLS,'MaxNumSplits',params.nSplit,...
    'PredictorSelection','interaction-curvature');

%Out of bag error
oobErr = median(oobError(randomForest));

%Test model
y_hat = predict(randomForest,C(valList==0,:));
y_hat = categorical(y_hat);
clear randomForest
Y = table2array(C(valList==0,end));

%Calculate the F1 measure
%Rows are true classes, columns are predicted classes, values are counts
cm = confusionmat(Y,y_hat);
F1err = zeros(size(cm,2),1);
for i =1:size(cm,2)
    tp = cm(i,i);
    P_hat = sum(cm(:,i));
    P = sum(cm(i,:));
    precision = tp/P_hat;
    recall = tp/P;
    if isnan(precision)
        precision = 0;
    end
    if isnan(recall)
        recall = 0;
    end 
    F1err(i) = 2 * precision * recall / (precision + recall);
    if isnan(F1err(i))
        F1err(i) = 0;
    end
end
F1err = 1-mean(F1err);

%Return the error metric
testErr = oobErr;% + etim/100;

end
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
function y_hat = bagClTreeApply(X,mdlobj,~)
y_hat = zeros(size(X,1),length(mdlobj));
for i=1:length(mdlobj)
    %Make predictions on new data
    treebag = mdlobj{i};
    [~,scores,~] = predict(treebag,X);
    y_hat(:,i) = scores(:,2);
end
end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function bagClTreeReport(clasfier,~)
try
    
catch err
    disp('Error reporting the kNN Classification model');
end

end