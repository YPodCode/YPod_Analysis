function [classObj, C_hat, X_hat, X_all] = VOCClustering_multisource(settingsSet)
%This function loads a list of categorical class labels from a csv and then
%fits classification algorithms to predict that class using the estimated
%concentrations that were fitted during the main regression
%This was writted for Jacob Thorson's thesis work and hasn't been tested
%with datasets other than his own

%% Fix for running once on PC and once on Mac (or vise-versa)
if xor(settingsSet.ispc, ispc)
    disp('Original run was on different OS. Select the folder again to get the right file directory');
    disp(['Original path was: ' settingsSet.outpath])
    originalPath = uigetdir(pwd,'Select folder with previous analysis outputs');
    assert(~isequal(originalPath,0),'Error: no file selected, run ended'); %Check that file was selected
    settingsSet.outpath = originalPath;
else
    originalPath = settingsSet.outpath;
end
%settingsSet.classList = {'GLM_MultiClass'};
settingsSet.classList = {'SVM_Classifier','baggedClassTrees','GLM_MultiClass','SVM_multiClass','patternNet_multiClass'};
%,'boostTree_Classifier','podPatternRecog','baggedClassTrees','podkNN_Class','LDA_Classifier',

%% Load the "reference" classification data where each source is a column with 0/1 to indicate its presence
%Read in the data
disp('Select CSV containing reference categories')
[file, path] = uigetfile('*.csv');
assert(~isequal(file,0),'No file selected');
filePath = [path file];
opts = detectImportOptions(filePath,'Delimiter',',');
opts = setvartype(opts,opts.VariableNames,[{'char'} repmat({'double'},1,size(opts.VariableNames,2)-1)]);
opts.VariableNames{1} = 'datetime';
refData = readtable(filePath, opts);
C_ref = refData(:,2:end);
ct = refData.datetime;
ct = datetime(ct,'InputFormat','M/d/yy H:mm:ss');

%% Preprocess the classification reference data
settingsSet.refSmooth = 4;%0=median, 1=mean, 2=linear interpolation, 3=smoothing spline, 4=mode
settingsSet.refPreProcess = {'sortbyTime','removeDST','podSmooth'};
settingsSet.filtering = 'ref';
for ii = 1:length(settingsSet.refPreProcess)
    settingsSet.loops.ii=ii;
    %Get string representation of function - this must match the name of a filter function
    filtFunc = settingsSet.refPreProcess{ii};
    %Convert this string to a function handle to feed the pod data to
    filtFunc = str2func(filtFunc);
    %Save filtered reference data into Y
    [C_ref,ct] = filtFunc(C_ref, ct, settingsSet);
    %Clear for next loop
    clear filtFunc
end%loop for preprocessing reference data

%% These are the number of reference files, pods, regressions, validations, and folds to evaluate
nRef   = length(settingsSet.fileList.colocation.reference.files.bytes); %Number of reference files
nPods  = size(settingsSet.podList.timezone,1); %Number of unique pods
nRegs  = length(settingsSet.modelList); %Number of regression functions
nVal   = length(settingsSet.valList); %Number of validation functions
nReps = settingsSet.nFoldRep;  %Number of folds to evaluate
nClassReps = nReps; %number of repeats for classification folds

%nStats = length(settingsSet.statsList); %Number of statistical functions to apply
%nPlots = length(settingsSet.plotsList); %Number of plotting functions

%% Initialize large cell array to hold all estimates
X_hat = cell(nPods,nRegs,nVal,nReps);
X_all = cell(nPods,nRegs,nVal,nReps);
%X_old = cell(nPods,nRegs,nVal,nReps);

%Load the originally fitted models for later looks
mdlsstruct = cell(nPods,nRef,nRegs,nVal,nReps);

%Plotting arrays
t_plot = cell(nPods,nRef,nRegs,nVal,nReps,2);
ind_plot = cell(nPods,nRef,nRegs,nVal,nReps,2);
Y_plot = cell(nPods,nRef,nRegs,nVal,nReps,2);
Y_hat_plot = cell(nPods,nRef,nRegs,nVal,nReps,2);
g.pods = strings(nPods,nRef,nRegs,nVal,nReps,2);
g.regs = strings(nPods,nRef,nRegs,nVal,nReps,2);
g.poll = strings(nPods,nRef,nRegs,nVal,nReps,2);
g.vals = strings(nPods,nRef,nRegs,nVal,nReps,2);
g.reps = zeros(nPods,nRef,nRegs,nVal,nReps,2);
g.calval = strings(nPods,nRef,nRegs,nVal,nReps,2);g.calval(:,:,:,:,:,1)='cal';g.calval(:,:,:,:,:,2)='val';
%% Load all of the old fitted data
for j = 1:nPods
    currentPod = settingsSet.podList.podName{j};
    g.pods(j,:,:,:,:,:) = currentPod;
    for i = 1:nRef
        if nRef==1
            reffileName = settingsSet.fileList.colocation.reference.files.name;
            pollutant = settingsSet.fileList.colocation.reference.files.pollutants;
        else
            reffileName = settingsSet.fileList.colocation.reference.files.name{i};
            pollutant = settingsSet.fileList.colocation.reference.files.pollutants{i};
        end
        currentRef = split(reffileName,'.');
        currentRef = currentRef{1};
        
        %Load the estimates
        temppath = ['Estimates_' currentPod '_' currentRef '.mat'];
        temppath = fullfile(originalPath,temppath); %Create file path for estimates
        if exist(temppath,'file')~=2
            warning(['No estimates found for reference: ' currentRef ', file skipped']);
            %             t_plot{j,i,m,k,kk,1} = zeros(100,1);
            %             Y_plot{j,i,m,k,kk,1} = zeros(100,1);
            %             Y_hat_plot{j,i,m,k,kk,1} = zeros(100,1);
            %             t_plot{j,i,m,k,kk,2} = zeros(100,1);
            %             Y_plot{j,i,m,k,kk,2} = zeros(100,1);
            %             Y_hat_plot{j,i,m,k,kk,2} = zeros(100,1);
            continue
        end
        g.poll(:,i,:,:,:,:) = pollutant;
        load(temppath); clear temppath
        
        %Load the data used for fitting models
        temppath = ['FitData_' currentPod '_' currentRef '.mat'];
        temppath = fullfile(originalPath,temppath); %Create file path for estimates
        load(temppath); clear temppath
        Y = table2array(fittingStruct.Y);
        t = fittingStruct.t;
        X_OG = fittingStruct.X; ind = X_OG.telapsed;
        valList = fittingStruct.valLists;
        
        
        %Load the data used for fitting models
        temppath = ['fittedModels_' currentPod '_' currentRef '.mat'];
        temppath = fullfile(originalPath,temppath); %Create file path for estimates
        load(temppath); clear temppath
        mdlsstruct(j,i,:,:,:)=fittedMdls;
        clear fittedMdls
        
        for m=1:nRegs
            g.regs(:,:,m,:,:,:) = settingsSet.modelList{m};
            for k=1:nVal
                g.vals(:,:,:,k,:,:) =  settingsSet.valList{k};
                for kk=1:nReps
                    g.reps(:,:,:,:,kk,:) = kk;
                    %Make temporary table with estimates
                    t_temp = t;
                    t_temp(valList{k}~=kk) = t(valList{k}~=kk);
                    t_temp(valList{k}==kk) = t(valList{k}==kk);
                    Y_temp = Y;
                    Y_temp(valList{k}~=kk,:) = Y(valList{k}~=kk,:);
                    Y_temp(valList{k}==kk,:) = Y(valList{k}==kk,:);
                    Y_hat_temp = Y;
                    Y_hat_temp(valList{k}~=kk,:) = Y_hat.cal{m,k,kk};
                    Y_hat_temp(valList{k}==kk,:) = Y_hat.val{m,k,kk};
                    
                    
                    %Data for plotting:
                    t_plot{j,i,m,k,kk,1} = datenum(t(valList{k}~=kk));
                    Y_plot{j,i,m,k,kk,1} = Y(valList{k}~=kk,:);
                    Y_hat_plot{j,i,m,k,kk,1} = Y_hat.cal{m,k,kk};
                    t_plot{j,i,m,k,kk,2} = datenum(t(valList{k}==kk));
                    Y_plot{j,i,m,k,kk,2} = Y(valList{k}==kk,:);
                    Y_hat_plot{j,i,m,k,kk,2} = Y_hat.val{m,k,kk};
                    
                    ind_plot{j,i,m,k,kk,1} = ind(valList{k}~=kk);
                    ind_plot{j,i,m,k,kk,2} = ind(valList{k}==kk);
                    
                    %Join time and values to allow combining with other references
                    tempname = [pollutant '_hat'];
                    tempX = table(t_temp, Y_hat_temp, 'VariableNames',{'datetime',tempname});
                    tempY = table(t_temp, Y_temp, 'VariableNames',{'datetime',pollutant});
                    clear t_temp Y_temp Y_hat_temp tempname
                    
                    if i==1
                        X_hat{j,m,k,kk} = tempX;
                        X_all{j,m,k,kk} = tempY;
                    else
                        X_hat{j,m,k,kk} = outerjoin(X_hat{j,m,k,kk},tempX,'Keys','datetime','MergeKeys',true);
                        X_all{j,m,k,kk} = outerjoin(X_all{j,m,k,kk},tempY,'Keys','datetime','MergeKeys',true);
                    end
                    clear tempX tempY
                end%fold loop
            end%validation loop
        end%model loop
    end%reference loop
    
    clear currentPod
end%pods loop
g.pods = categorical(g.pods);
g.regs = categorical(g.regs);
g.poll = categorical(g.poll);
g.vals = categorical(g.vals);
g.calval = categorical(g.calval);

%------------------------------------------------------------------------------------------------------------------------
%------------------------------------------------------------------------------------------------------------------------
%------------------------------------------------------------------------------------------------------------------------
%% Try classifying the data
%------------------------------------------------------------------------------------------------------------------------
%------------------------------------------------------------------------------------------------------------------------
%------------------------------------------------------------------------------------------------------------------------

nClass = length(settingsSet.classList);

%Grouping Variables
C_hat = struct;
C_hat.Y.cal = cell(nPods,nRegs,nVal,nReps,nClass);
C_hat.Y.val = cell(nPods,nRegs,nVal,nReps,nClass);
C_hat.Y_hat.cal = cell(nPods,nRegs,nVal,nReps,nClass);
C_hat.Y_hat.val = cell(nPods,nRegs,nVal,nReps,nClass);
cmodels = cell(nPods,nRegs,nVal,nReps,nClass,2);
valList = cell(nPods,nRegs,nVal,nReps,nClass);
for j = 1:nPods
    settingsSet.loops.j=j;
    for m=1:nRegs
        settingsSet.loops.m=m;
        for k=1:nVal
            settingsSet.loops.k=k;
            for kk=1:nReps
                settingsSet.loops.kk=kk;
                %Get reference Y values
                Y = X_all{j,m,k,kk};
                yt = Y.datetime;
                keepCols = ~strcmpi(Y.Properties.VariableNames,'datetime');
                Y = Y(:,keepCols);
                settingsSet.isRef=true;
                %Align source and concentration data
                [Y, C, t] = alignRefandPod(Y,yt,C_ref,ct,settingsSet);
                
                fprintf('j=%u, m=%u, k=%u, kk=%u, Y: [%u %u] C [%u %u]\n',[j m k kk size(Y) size(C)])
                
                valList{j,m,k,kk} = timeFold(C, Y, t, settingsSet.nFolds);
                %valList{j,m,k,kk} = t>datetime(2018,7,1);
                tempvallist = valList{j,m,k,kk};
                
                
                %%Try different classification functions
                for zz = 1:nClass
                    settingsSet.loops.zz=zz;
                    try
                    disp(['-------Fitting and applying model ' settingsSet.classList{zz} ' with reference Y_hat from: ' settingsSet.modelList{m}]);
                    %Get string representation of functions - this must match the name of a function saved in the directory
                    modelFunc = settingsSet.classList{zz};
                    %Convert this string to a function handle for the regression
                    modelFunc = str2func(modelFunc);
                    %Get the generation function for that regression
                    %Note that the function must be set up correctly - see existing regression functions for an example
                    fitFunc = modelFunc(1);
                    %Get the prediction function for that regression
                    predictFunc = modelFunc(2);
                    %Clear the main regression function for tidyness
                    clear modelFunc
                    
                    %Fit the selected regression
                    %Also returns the estimates and fitted model details
                    %Indices for the regression model array are: (i=nPods,m=nRegs,k=nVal,kk=nReps)
                    disp('-----Fitting...');
                    cmodels{j,m,k,kk,zz,1} = fitFunc(C(tempvallist~=kk,:), Y(tempvallist~=kk,:), settingsSet);
                    disp('-----Apply to training data...');
                    %Apply the fitted regression to the calibration data
                    C_hat.Y.cal{j,m,k,kk,zz} = predictFunc(Y(tempvallist~=kk,:),cmodels{j,m,k,kk,zz,1},settingsSet);
                    disp('-----Apply to test data...');
                    %Apply the fitted regression to the validation data
                    C_hat.Y.val{j,m,k,kk,zz} = predictFunc(Y(tempvallist==kk,:),cmodels{j,m,k,kk,zz,1},settingsSet);
                    close all
                    catch
                        disp('Try again')
                    end
                end
                
                
                
                %% Repeat using estimated concentrations (Y_hat)
                Y_hat = X_hat{j,m,k,kk};
                yhatt = Y_hat.datetime;
                keepCols = ~strcmpi(Y_hat.Properties.VariableNames,'datetime');
                Y_hat = Y_hat(:,keepCols);
                [Y_hat, C, t] = alignRefandPod(Y_hat,yhatt,C_ref,ct,settingsSet);
                settingsSet.isRef=false;
                for zz = 1:nClass
                    settingsSet.loops.zz=zz;
                    disp(['-------Fitting and applying model ' settingsSet.classList{zz} ' with estimated Y_hat from: ' settingsSet.modelList{m}]);
                    %Get string representation of functions - this must match the name of a function saved in the directory
                    modelFunc = settingsSet.classList{zz};
                    %Convert this string to a function handle for the regression
                    modelFunc = str2func(modelFunc);
                    %Get the generation function for that regression
                    %Note that the function must be set up correctly - see existing regression functions for an example
                    fitFunc = modelFunc(1);
                    %Get the prediction function for that regression
                    predictFunc = modelFunc(2);
                    %Clear the main regression function for tidyness
                    clear modelFunc
                    
                    %Fit the selected regression
                    %Also returns the estimates and fitted model details
                    %Indices for the regression model array are: (j=nPods,m=nRegs,k=nVal,kk=nReps)
                    disp('-----Fitting...');
                    cmodels{j,m,k,kk,zz,2} = fitFunc(C(tempvallist~=kk,:), Y_hat(tempvallist~=kk,:), settingsSet);
                    
                    %Apply the fitted regression to the calibration data
                    disp('-----Apply to training data...');
                    C_hat.Y_hat.cal{j,m,k,kk,zz} = predictFunc(Y_hat(tempvallist~=kk,:),cmodels{j,m,k,kk,zz,2},settingsSet);
                    
                    %Apply the fitted regression to the validation data
                    disp('-----Apply to test data...');
                    C_hat.Y_hat.val{j,m,k,kk,zz} = predictFunc(Y_hat(tempvallist==kk,:),cmodels{j,m,k,kk,zz,2},settingsSet);
                    
                    close all
                end
            end
        end
    end
end
%Save classification results
classObj = {C_hat, C,valList,settingsSet};
temppath = fullfile(originalPath,'classResults');
save(char(temppath),'classObj','-v7.3');
temppath = fullfile(originalPath,'classModels');
save(char(temppath),'cmodels','-v7.3');

%Plot CDFs 
CatPlotting(originalPath,C_hat,C,valList,settingsSet)
redoPlots(settingsSet)
end






