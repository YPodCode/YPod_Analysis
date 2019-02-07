function func = podRFR(a)
%Gaussian process regression implementation with optimization

switch a
    case 1; func = @rfrGen;
    case 2; func = @rfrApply;
    case 3; func = @rfrReport;
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function mdlobj = rfrGen(Y,X,settingsSet)

mdlobj = cell(size(Y,2),1);

%Check for existing RFR optimization for this pod
if length(settingsSet.fileList.colocation.reference.files.bytes)==1; reffileName = settingsSet.fileList.colocation.reference.files.name;
else; reffileName = settingsSet.fileList.colocation.reference.files.name{settingsSet.loops.i}; end
currentRef = split(reffileName,'.');
currentRef = currentRef{1};

filename = [settingsSet.podList.podName{settingsSet.loops.j} currentRef 'RFRsave.mat'];
regPath = fullfile(settingsSet.outpath,filename);

%% If RFR has already been optimized for this pod, can skip the optimization, which is really slow
if exist(regPath,'file')==2
    %Load the previous analysis
    load(regPath);
else
    maxMinLS = max(2,floor(size(X,1)/2));
    maxNsplit = max(2,floor(size(X,1)/2));
    minLS = optimizableVariable('minLS',[1,maxMinLS],'Type','integer');
    numPTS = optimizableVariable('numPTS',[1,size(X,2)-1],'Type','integer');
    nSplits = optimizableVariable('nSplits',[1 maxNsplit],'Type','integer');
    hyperparametersRF = [minLS; numPTS; nSplits];
    
    %Initialize a structure to hold the parameters for the fitted kernel
    rfrstruct = zeros(3,size(Y,2));
    
    %Go through and create fitted functions using the parameters fitted just
    %now or on an earlier run (like on a previous fold)
    for i = 1:size(Y,2)
        C = [X Y(:,i)];
        rng(42)
        results = bayesopt(@(params)oobErrRF(params,C),hyperparametersRF,...
            'AcquisitionFunctionName','expected-improvement-plus',...
            'Verbose',1,'UseParallel', true,'NumSeedPoints',10,'MaxObjectiveEvaluations',50);
        
        %Get the optimized hyperparameters
        bestHyperparameters = results.XAtMinEstimatedObjective;
        rfrstruct(1,i) = bestHyperparameters.minLS;
        rfrstruct(2,i) = bestHyperparameters.numPTS;
        rfrstruct(3,i) = bestHyperparameters.nSplits;
        
        close all
    end
    
    %save the fitted parameters for later runs
    save(char(regPath),'rfrstruct');
end

%% Now go through and apply optimized hyperparameters
for i = 1:size(Y,2)
    C = [X Y(:,i)];
    rng(42)
    disp('-----Fitting RFR with optimized hyperparameters');
    Mdl = TreeBagger(500,C,Y.Properties.VariableNames{i},'Method','regression',...
        'OOBPrediction','on','OOBPredictorImportance','on',...
        'PredictorSelection','interaction-curvature',...
        'MinLeafSize',rfrstruct(1,i),'NumPredictorstoSample',rfrstruct(2,i),'MaxNumSplits',rfrstruct(3,i));
    
    %Save variable importance plot
    if settingsSet.savePlots && settingsSet.loops.kk==1
        imp = Mdl.OOBPermutedPredictorDeltaError;
        figure('Position',get( groot, 'Screensize' ));
        [~, I] = sort(imp);
        bar(imp(I));
        title([Y.Properties.VariableNames{i} ' Curvature Test']);
        ylabel('Predictor importance estimates');
        xlabel('Predictors');
        h = gca;
        h.XTick = 1:size(X,2);
        h.XTickLabel = Mdl.PredictorNames(I);
        h.XTickLabelRotation = 45;
        h.TickLabelInterpreter = 'none';
        clear Mdl
        
        temppath = [settingsSet.podList.podName{settingsSet.loops.j} Y.Properties.VariableNames{i} currentRef '_RFRvarImp'];
        temppath = fullfile(settingsSet.outpath,temppath);
        saveas(gcf,temppath,'jpeg');
        clear temppath
        close(gcf)
    end
    
    %Keep a compact version of the model
    mdlobj{i} = compact(Mdl);
end
end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function testErr = oobErrRF(params,C)
%oobErrRF Trains random forest and estimates out-of-bag quantile error
%   oobErr trains a random forest of 300 regression trees using the
%   predictor data in X and the parameter specification in params, and then
%   returns the out-of-bag quantile error based on the median. X is a table
%   and params is an array of OptimizableVariable objects corresponding to
%   the minimum leaf size and number of predictors to sample at each node.

valList = repmat([zeros(15,1);ones(30,1)],ceil(size(C,1)/45),1);
valList = valList(1:size(C,1),1);
yname = C.Properties.VariableNames{end};
rng(42)
randomForest = TreeBagger(500,C(valList==1,:),yname,'Method','regression',...
    'OOBPrediction','on','MinLeafSize',params.minLS,'MaxNumSplits',params.nSplits,...
    'PredictorSelection','interaction-curvature',...
    'NumPredictorstoSample',params.numPTS);
%oobErr = oobQuantileError(randomForest);

%Test model using blocks of validation data
y_hat = predict(randomForest,C(valList==0,:));
Y = table2array(C(valList==0,end));
testErr = sqrt(mean((Y-y_hat).^2));

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = rfrApply(X,mdlobj,~)
%Initialize y_hat
y_hat = zeros(size(X,1),length(mdlobj));
ynames = cell(length(mdlobj),1);

%Make new predictions
for i = 1:length(mdlobj)
    y_hat(:,i) = predict(mdlobj{i},X);
    %ynames{i} = mdlobj{i}.ResponseName;
end
%Put back into a table
%y_hat = array2table(y_hat,'VariableNames',ynames);
end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function rfrReport(mdlobj,~)
try
    figure;
    subplot(2,2,1);plotResiduals(mdl);
    subplot(2,2,2);plotDiagnostics(mdl,'cookd');
    subplot(2,2,3);plotResiduals(mdl,'probability');
    subplot(2,2,4);plotResiduals(mdl,'lagged');
    plotSlice(mdl);
catch err
    disp('Error reporting the random forest model');
end

end
%--------------------------------------------------------------------------
