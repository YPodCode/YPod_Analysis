function func = podGPR(a)
%Gaussian process regression implementation with optimization

switch a
    case 1; func = @gprGen;
    case 2; func = @gprApply;
    case 3; func = @gprReport;
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function mdlobj = gprGen(Y,X,settingsSet)

mdlobj = cell(size(Y,2)+1,1);

%Don't want it to just fit to points that are close in time
%if any(strcmp('telapsed', X.Properties.VariableNames));X.telapsed = [];end

%Check for existing GPR for this pod
if length(settingsSet.fileList.colocation.reference.files.bytes)==1; reffileName = settingsSet.fileList.colocation.reference.files.name;
else; reffileName = settingsSet.fileList.colocation.reference.files.name{settingsSet.loops.i}; end
currentRef = split(reffileName,'.');
currentRef = currentRef{1};

filename = [settingsSet.podList.podName{settingsSet.loops.j} currentRef 'GPRsave.mat'];
regPath = fullfile(settingsSet.outpath,filename);

%Standardize X (subtract mean and divide by std deviation)
normmat = zeros(size(X,2),2);
for i = 1:size(X,2)
    tempx = table2array(X(:,i));
    medx = mean(tempx);
    stdx = std(tempx);
    if stdx == 0
        stdx = 1;
    end
    tempx = (tempx-medx)/stdx;
    normmat(i,:) = [medx stdx];
    X(:,i) = array2table(tempx,'VariableNames',X.Properties.VariableNames(i));
end

%% If GPR has already been optimized for this pod, can skip the optimization, which is really slow
if exist(regPath,'file')==2
    %Load the previous analysis
    load(regPath);
else
    %Initialize a structure to hold the parameters for the fitted kernel
    gprstruct = zeros(2,size(Y,2));
    
%     scalePar1 = optimizableVariable('scalePar1',[1e-4, 1e2],'Transform','log');
%     scalePar2 = optimizableVariable('scalePar2',[1e-4, 1e2],'Transform','log');
%     %kernFunc = optimizableVariable('kernFunc',{'matern32','squaredexponential','ardsquaredexponential'},'Type','categorical');
%     hyperparametersGPR = [scalePar1; scalePar2];
    
    %For each column of Y, fit a GPR model using all of X
    for i = 1:size(Y,2)
        C = [X Y(:,i)];
        yname = Y.Properties.VariableNames{i};
        
        %Automatically optimize the hyperparameters for a guassian process regression
        %         try %My own optimization
        %         results = bayesopt(@(params)oobErrGPR(params,C),hyperparametersGPR,...
        %             'AcquisitionFunctionName','expected-improvement-plus','Verbose',0);
        %         bestHyperparameters = results.XAtMinObjective;
        %         gprstruct(:,i) = [bestHyperparameters.scalePar1; bestHyperparameters.scalePar2];
        %
        %         catch %If it breaks, use the default optimization algorithm
        rng(1)
        testmdl = fitrgp(C,yname,'KernelFunction','squaredexponential',...
            'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',...
            struct('AcquisitionFunctionName','expected-improvement-plus',...
            'ShowPlots',true,'UseParallel',true));
        gprstruct(:,i) = testmdl.KernelInformation.KernelParameters;
%         end
        
        close all
    end
    
    %save the fitted parameters for later runs
    save(char(regPath),'gprstruct');
end

%Go through and create fitted functions using the parameters fitted just
%now or on an earlier run (like on a previous fold)
for i = 1:size(Y,2)
    C = [X Y(:,i)];
    rng(1)
    tempmdl = fitrgp(C,Y.Properties.VariableNames{i},'KernelFunction','squaredexponential','KernelParameters',gprstruct(:,i));
    mdlobj{i} = compact(tempmdl);
    clear tempmdl
end
%Also keep the matrix used for normalization
mdlobj{i+1} = normmat;

end
%--------------------------------------------------------------------------

% %--------------------------------------------------------------------------
% function testErr = oobErrGPR(params,C)
% %oobErrRF Fits a GPR model with specified parameters and then tests it
% %using simple 5 fold validation
% 
% 
% kparams0 = [params.scalePar1, params.scalePar2];
% 
% %scalePar = params.scalePar;
% %kernFunc = char(params.kernFunc);
% ntrain = size(C,1) - floor(size(C,1)/5);
% yname = C.Properties.VariableNames{end};
% 
% %Fit the model on a subset of the data
% testmdl = fitrgp(C(1:ntrain,:),yname,'KernelFunction','squaredexponential','KernelParameters',kparams0);%'Sigma',scalePar);
%      
% %oobErr = oobQuantileError(randomForest);
% 
% %Test model
% y_hat = predict(testmdl,C(ntrain:end,:));
% Y = table2array(C(ntrain:end,end));
% testErr = sqrt(mean((Y-y_hat).^2));
% 
% end
% %--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = gprApply(X,mdlobj,~)
%Don't want it to just fit to points that are close in time
%if any(strcmp('telapsed', X.Properties.VariableNames));X.telapsed = [];end
%Normalize the new X
normmat = mdlobj{end};
mdlobj = mdlobj(1:end-1);
for i = 1:size(X,2)
    medx = normmat(i,1);
    stdx = normmat(i,2);
    tempx = table2array(X(:,i));
    tempx = (tempx-medx)/stdx;
    normmat(i,:) = [medx stdx];
    X(:,i) = array2table(tempx,'VariableNames',X.Properties.VariableNames(i));
end

%Initialize y_hat
y_hat = zeros(size(X,1),length(mdlobj));
%ynames = cell(length(mdlobj),1);

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
function gprReport(mdlobj,~)
try
    figure;
    subplot(2,2,1);plotResiduals(mdl);
    subplot(2,2,2);plotDiagnostics(mdl,'cookd');
    subplot(2,2,3);plotResiduals(mdl,'probability');
    subplot(2,2,4);plotResiduals(mdl,'lagged');
    plotSlice(mdl);
catch err
    disp('Error reporting the gaussian process regression model');
end

end
%--------------------------------------------------------------------------
