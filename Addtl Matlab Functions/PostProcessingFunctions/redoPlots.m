function redoPlots(settingsSet)


%% Fix for running once on PC and once on Mac (or vise-versa)
if xor(settingsSet.ispc, ispc)
    disp('Original run was on different OS. Select the folder again to get the right file directory');
    originalPath = uigetdir(pwd,'Select folder with previous analysis outputs');
    assert(~isequal(originalPath,0),'Error: no file selected, run ended'); %Check that file was selected
else
    originalPath = settingsSet.outpath;
end

%% These are the number of reference files, pods, regressions, validations, and folds to evaluate
nRef   = length(settingsSet.fileList.colocation.reference.files.bytes); %Number of reference files
nPods  = size(settingsSet.podList.timezone,1); %Number of unique pods
nModels  = length(settingsSet.modelList); %Number of regression functions
nValidation   = length(settingsSet.valList); %Number of validation functions
%nReps = settingsSet.nFoldRep;  %Number of folds to evaluate
nPlots = length(settingsSet.plotsList); %Number of plotting functions
nStats = length(settingsSet.statsList); %Number of statistical functions to apply

%% Load all of the old fitted data
for j = 1:nPods
    settingsSet.loops.j=j;
    currentPod = settingsSet.podList.podName{j};
    for i = 1:nRef
        settingsSet.loops.i=i;
        if nRef==1
            reffileName = settingsSet.fileList.colocation.reference.files.name;
            pollutant = settingsSet.fileList.colocation.reference.files.pollutants;
        else
            reffileName = settingsSet.fileList.colocation.reference.files.name{i};
            pollutant = settingsSet.fileList.colocation.reference.files.pollutants{i};
        end
        currentRef = split(reffileName,'.');
        currentRef = currentRef{1};
        
        mdlStats = cell(nModels,nValidation,nStats);
        
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
        load(temppath); clear temppath
        
        %Load the data used for fitting models
        temppath = ['FitData_' currentPod '_' currentRef '.mat'];
        temppath = fullfile(originalPath,temppath); %Create file path for estimates
        load(temppath); clear temppath
        Y = fittingStruct.Y;
        t = fittingStruct.t;
        X = fittingStruct.X;
        valList = fittingStruct.valLists;
        
        
        %Load the data used for fitting models
        temppath = ['fittedModels_' currentPod '_' currentRef '.mat'];
        temppath = fullfile(originalPath,temppath); %Create file path for estimates
        load(temppath); clear temppath
                %% --------------------------START VALIDATION SETS LOOP------------------------------
        %Create a vector used to separate calibration and validation data sets
        for k = 1:nValidation
            settingsSet.loops.k=k;
            %% --------------------------START REGRESSIONS LOOP------------------------------
            %Fit regression equations and validate them
            for m = 1:nModels
                settingsSet.loops.m=m;
                %% ------------------------------Determine statistics------------------------------
                disp('-----Running statistical analyses...');
                for mm = 1:nStats
                    %Keep track of the loop number in case it's needed by a sub function
                    settingsSet.loops.mm=mm;
                    %Get string representation of function - this must match the name of a function
                    statFunc = settingsSet.statsList{mm};
                    fprintf('------Applying statistical analysis function %s ...\n',statFunc);
                    
                    %Convert this string to a function handle to feed data to
                    statFunc = str2func(statFunc);
                    
                    %Apply the statistical function m=nRegs,k=nVal,mm=nStats
                    mdlStats{m,k,mm} = statFunc(X, Y, Y_hat, valList{k}, fittedMdls(m,k,:), settingsSet);
                    
                    clear statFunc
                end%loop of common statistics to calculate
            end%loop of regressions
        end%loop of calibration/validation methods
        
        %% ------------------------------Create plots----------------------------------------
        disp('-----Plotting estimates and statistics...');
        for mm = 1:nPlots
            %Keep track of the loop number in case it's needed by a sub function
            settingsSet.loops.mm=mm;
            %Get string representation of function - this must match the name of a function
            plotFunc = settingsSet.plotsList{mm};
            %if ~contains(lower(plotFunc),'box');continue;end
            fprintf('------Running plotting function %s ...\n',plotFunc);
            
            %Convert this string to a function handle to feed data to
            plotFunc = str2func(plotFunc);
            
            %Run the plotting function m=nRegs,k=nVal,kk=nFold
            plotFunc(t, X, Y, Y_hat,valList,mdlStats,settingsSet);
            
            %Save the plots if selected and then close them (reduces memory load and clutter)
            if settingsSet.savePlots && ishandle(1)
                temppath = [currentPod '_' currentRef '_' settingsSet.plotsList{mm}];
                temppath = fullfile(originalPath,temppath);
                saveas(gcf,temppath,'fig');
                saveas(gcf,temppath,'png');
                clear temppath
                close(gcf)
            end
            clear plotFunc
        end%loop of plotting functions
    end%reference loop
    clear currentPod
end%pods loop



end

