%% USAGE NOTES
%{
This code is based on Ricardo Piedraheta's Pod_calibration_generation
code, with attemps to make it more readable and more easily modified.
This was developed on Matlab R2017b, so some unhandled errors may be caused
by the use of different versions.

%}
%% Change Log
%{
    Rev     Date        Revision Notes
    V2.0	12/01/2017  Creation
    V2.1    07/18/2018  Minor tweaks and addition of new functions
    V2.2    09/13/2018  Modified how fitting and prediction with models occurs to improve consistency
%}

%% Clear variables and close old figures
clear variables;
close all;

%% ------------=========Settings for analysis=========------------
%% Change these values to affect what portions of the code is run
settingsSet.loadOldSettings = false;  %Load a settings set structure from an old run
settingsSet.convertOnly = false;  %Stop after converting (unconverted) pod and reference data files into .mat files
settingsSet.generateCal = true;  %Generate fitted models - if no, will skip any data in the "colocation" folder and will not fit new models
settingsSet.applyCal = false;  %Apply calibrations to the pod files in the "field" folder
settingsSet.savePlots = true;  %Save plots as image files as they are created

%% These affect how datetime data is imported by function "dataExtract"
%Custom datetime formats to use when trying to convert non-standard import datetime strings
settingsSet.datetimestrs = {'M/d/yy H:mm:ss','M/d/yy H:mm','yyyy.MM.dd H.m.s'};
%Custom date formats to use when trying to convert non-standard import date strings
settingsSet.datestrings = {'yyyy.MM.dd','MM-dd-yyyy'};
%Custom time formats to use when trying to convert non-standard time strings
settingsSet.timestrings = {'H.m.s'};

%% These change the smoothing of pod and reference data in function "smoothData"
%Time in minutes to average data by for analysis
settingsSet.timeAvg = 1;
%Smoothing method for pod data - 0=median, 1=mean, 2=linear interpolation, 3=smoothing spline, 4=mode
settingsSet.podSmooth = 0;
%Smoothing method for reference data. Values are the same as above
settingsSet.refSmooth = 0;

%% In these lists of columns to extract, be sure that the data will have headers with one match to each entry.  Note that partial matches will be used (e.g.: a column with "CO2" would be selected if you entered "CO" below)
%Name of columns in reference files to try to extract from the full data file
settingsSet.refGas = {'Gasoline','NMHC','CH4','CO1','CO2','NO2'};%,'CONO','NatGas'
%settingsSet.refGas = {'CH4'};%!!!Not tested thoroughly with more than one pollutant

%List of sensor signals to extract from the full pod data file
%Use "allcol" to extract all columns that are not dates
settingsSet.podSensors = {'NO2_B1','CO_B4','H2S_BH','O3_B4','NO_B4','fig2600','fig2602','fig4161','fig2611','e2v2611','MICS2710','MICS5121','MICS2611','MICS5525','co2_NDIR','bl_mocon'}; %
%settingsSet.podSensors = {'fig2600'}; %

%List of environmental sensor signals to extract from the full pod data file
%These are separated from the other sensors in case they need to be treated differently than the other sensors (e.g. when determining interaction terms
settingsSet.envSensors = {'temperature','humidity','bme_P'}; %

%% For function lists, enter function names in the order in which you'd like them to be performed (if it matters)
%Preprocessing functions for reference data
settingsSet.refPreProcess = {'removeNaNs','remove999','sortbyTime','removeDST','podSmooth','removeZeros'};%
%Preprocessing functions for pod data 
settingsSet.podPreProcess = {'joinPodsData','sortbyTime','podSmooth','makeDiscontTElaps','rmWarmup','removeNaNs','podTimeZone','TempC2K','removeOOBtemp','humidrel2abs'};%

%Regression/classification model(s) to evaluate
settingsSet.modelList = {'podNN','podRFR','podGPR','podRidge','podStepLM','linearSensors','fullLinear'};%,'podStepLM','joannaNN'

%Validation set selection functions
%settingsSet.valList = {'timeFold','timeofDayVal','temperatureFold','environClusterVal','concentrationsFold'};%
settingsSet.valList = {'timeFold'};

%Number of 'folds' for validation (e.g. 5 folds ~ 20% dropped for each validation set)
settingsSet.nFolds = 10;
%Number of folds to actually use (up to nFolds) (will be used in order from 1 to nFolds)
settingsSet.nFoldRep = 10;

%Statistics to calculate
settingsSet.statsList = {'podRMSE','podR2','podCorr'};%

%Plotting functions run during model fitting loops
settingsSet.plotsList = {'acfPlot','timeseriesPlot','vsReferencePlot','basicStatsBoxPlots'};%'originalPlot',

%Plotting functions to run after applying calibrations to field data
%(not all normal plotting functions will work correctly with field data)
settingsSet.fieldPlots = {'timeseriesPlot'};%

%Functions to run at the end of everything.  This can be empty
settingsSet.postProcesses = {'VOCClustering_multisource'};%VOCClustering

%% List of currently implemented functions that can be used in the settings lists above
%***Implemented modeling functions for regression and classification problems
%{
Regression:
-fullLinear - fits Y as a linear function of all columns of X
-fullLinInt - same as fullLinear, but also includes interaction terms
-linearSensors - designed to use our knowledge of sensors that are appropriate to the current pollutant and then fits a linear model using those + temperature and humidity
-line1 - fits Y as a linear function only of the first sensor in "podSensors"
-line1T - adds time as an independent predictor and adds an interaction term to allow for changes in sensitivity over time
-line1_loess - same as line1, adds a surface fitted to the residuals with respect to T and Rh and then adds those results to the line1 fit
-line3 - fits the podSensor as a linear function of Y, T, Rh and then inverts the model to predict Y
-line3T - same as line 3, but including an interaction between the temperature and Y
-line4 - fits the podSensor as a linear function of Y, T, Rh, elapsed time, and then inverts the model to predict Y
-line4T - same as line 4, but includes interaction between temperature and Y
-linePCA - maps X into principal components and then fits a simple linear regression with interaction terms
-joannaNN - uses all columns and the settings that Joanna developed for using ANN to regress methane concentrations
-podGPR - Tries to optimize and then apply a gaussian process regression (Kriging) to the data based on all columns of X
-podM5p - Uses a custom function to fit M5' regression trees where each "leaf" has a linear fit at the end instead of a raw numerical prediction
-podNN - Tries to optimize the number of neurons in a two layer neural net, with the option to have 0 in 2nd layer, which is a 1 layer net
-podPLSR - Use partial least squares regression to select relevant parameters from a high dimensionality "X"
-podRFR - Tries to optimize then apply a random forest regression model. Hyperparameters optimized are the minimum number of leaves and number of parameters to select from at each edge
-podStepGLM - Use R^2 (or others with minor modifications) to select terms for a GLM regression with a log-normal distribution
-podStepLM - Use R^2 (or others with minor modifications) to select terms for a linear regression in a stepwise fashion
-smart4T - Similar to "linearSensors" but usese 'fitnlm' to fit a similar equation to the line4T model using all recognized sensors

Classification:
-baggedClassTrees - uses bagged classification trees and allows the user to select how many trees to fit
-clusterAll - tries to cluster data into categories (unsupervised).  Fits to 1:10 clusters and then allows user to specify number of clusters to use as the estimate
-GLM_MultiClass - fits a linear model to a binomially distributed reference value (true/false) for each column of Y for e.g. leak detection or source ID
-LDA_Classifier - fits a multi-class linear discriminant model to predict classes
-patternNet_multiClass - fits a shallow neural network for multi-class classification problems
-podClassTree - uses fitctree to try to create and prune a classification tree
-podkNN_Class - fits a k-nearest neighbor classification model
-podMultinomial - fits a multinomial regression using all of X
-podPatternRecog - uses a neural net to identify patterns and classify data
-SVM_Classifier - fits a multi-class support vector machine classification model
%}
%***Preprocessing functions
%{
-addACF - adds columns with a rolling autocorrelation for each variable
-addDerivatives - adds columns for each variable with the time derivative
-addRollingVar - uses "movvar" to add a rolling variance for each variable
-addSolarAngle - adds a column with solar angle "alpha" (currently using a fixed GPS coordinate)
-addTimeofDay - adds a column "ToD" that ranges from 0 at midnight to 2 at noon in a sine wave-like function
-clusterContin2Categ - uses k-means clustering to convert a matrix of continuous variables to categorical clusters
-easyRbyR0 - NOT QUITE WORKING? Converts voltage to resistance by assuming a voltage divider and then tries to find a "clean air" resistance and divides the calculated resistance by that value 
-humidrel2abs - converts relative humidity to absolute humidity
-joinPodsData - joins the data from all pods together (this means that looping through each pods is essentially repeating the same analysis multiple times)
-makeCategorical - converts each column to a categorical variable
-makePCs - centers and scales each variable and then performs a PCA and allows the user to select the number of PCs to keep. Saves the information necessary to convert new data into the PC space
-makeDiscontTElaps - adds "telapsed" column that ignores gaps longer than "t_gap", which is specified in the function code (currently 5 minutes)
-makeTElapsed - adds "telapsed" column calculated as t - min(t)
-normalizeMat - center and scale each variable by its standard deviation
-plotWavelets - plots the 2D wavelet power spectrum for each sensor or reference value
-podTimeZone - adjusts the pod data to match the reference timezone entered above under "settingsSet.refTZ"
-podSmooth - applies the smoothing method selected by "podSmooth" and "refSmooth" in the settings set
-refTimeZone- same as podTimeZone, but this moves reference data to match the pod timezone and allows reference files to be in different timezones
-RbyR0 - uses a file titled "ResistanceValues.csv" in the "Logs" folder to divide each sensor by its clean air resistance value
-referenceSpikeFilter - similar filtering to that included in Ricardo's code.  Assigns NaN values to points that differ from the point before or after them by more than 2 std deviations as calculated on a 60 minute rolling window
-remove999 - removes values exactly equal to -999
-removeDST - adustst the timeseries to remove daylight savings time (allows for data that bridges the time change
-removeNaNs - remove rows that contain NaN values
-removeOOBtemp - removes unrealistic temperature values (assumes values are in K)
-removeSomeZeros - removes 2/3 of points that are exactly 0 to reduce overtraining on synthetic reference data
-removeToDTrends - fits a smooth function to values vs time of day and then removes the fitted diurnal trend
-removeZeros - adds a small, positive value (determined relative to the median) to values that are exactly zero
-rmWarmup - removes 60 minutes after a break in pod data that is longer than 5x the calculated typical interval
-sortbyTime - sorts the data by the timeseries in case it was imported out of order
-TempC2K - adjusts the "temperature" column to convert from celcius to kelvin
-waveletRemove - Not implemented yet, but want to filter data with SSA
%}
%***Validation functions
%{
-clusterVal - Select data groupings with k-means clustering on X data. Note that clusters will not be the same size and that group 1 will be the largest (most points dropped for validation)
-concentrationsFold - Select and drop percentiles of reference concentration starting with the highest levels
-environClusterVal - Select groups of data by clustering on temperature and humidity values
-prepostVal - Select groups of data from out to in (group 1 gets the first and last x points and group n is the middle)
-randHrsVal - Randomly assign hour blocks of data to folds
-randVal - Randomly select data points for validation while trying to maintain roughly the same number of points in each dataset
-temperatureFold - Select and drop percentiles of temperature starting with the lowest temperatures
-timeFold - Split the data into similarly sized folds based on the time from start to end
-timeofDayVal - Split data into quantiles where fold 1 is centered around noon, and the last fold is centered around midnight
%}
%***Plotting functions
%{
-acfPlot - plots the time lagged correlation of Y and podSensor{1} to check that the timestamps are well aligned (max correlation at t_lag=0)
-basicStatsBoxPlots - makes boxplots of variation in model statistics between different "folds" as calculated by "podRMSE", "podR2", and "podCorr" (for now)
-categoricalPlot - designed to plot the results of a classification problem with categorical results
-histogramsPlot - plots the distribution of estimated and true concentrations
-hourBoxPlot - creates hourly boxplots of concentrations for the reference and estimated data
-originalPlot - Creates a similar plot to that produced by Ricardo's "camp_plotting" function for each fitted model
-residsTrends - Plots the residuals (y_hat-y) versus each covariate, the timestamp, and the reference concentration
-timeofDayPlot - Plots all reference and estimate values by time of day, colored by the "telapsed" variable (if it exists), or by elapsed time calculated as (t - min(t))
-timeseriesPlot - plots the predictions of each calibration/validation fold as well as the reference data over the same period
-vsRefDens - like "vsReferencePlot", but plots density of estimates to see if there is any weird clumping that's hard to see in a scatter plot
-vsReferencePlot - plots predictions versus the reference data
-XYCorrelations - plots correlations between all variables in X and Y
%}
%***Statistical functions
%{
Regression:
-podCorr - Calculates the Spearman's rho correlation between estimates and reference concentrations
-podRMSE - Calculates the classic RMSE value on the validation and calibration datasets for each fold
-podR2 - Calculates the coefficient of determination (R^2) value on the validation and calibration datasets for each fold
-podSkewness - Calculate the "skewness" of the distributions of concentrations for the reference and for the estimates

Classification:
-podSilhouette - Plots the silhouette for each point comparing its similarity within its group to its similarity to other groups. Positive 1 =ideal, -1 = terrible (too many/too few clusters)
%}
%-----------------------------------------------------------------
%% ------------=========End settings for analysis=========------------



%% Begin Body of Code
disp('-----------------------------------------------------')
disp('-----------------Pod Analysis Code-------------------')
disp('---------------Begin Code Version 2.2----------------')
disp('-----------------------------------------------------')

%% Perform some Matlab housekeeping, initial checks, and path additions
disp('Performing system checks and housekeeping...')

%Fix for cross compatibility between OSX and Windows, and record which this
%was run on
settingsSet.ispc = ispc;
if settingsSet.ispc; slash = '\'; else; slash = '/'; end

%Add the subfolder containing other functions
addpath(genpath('Addtl Matlab Functions'));

%Check Matlab version installed is recent enough
assert(~verLessThan('matlab','9.1'),'Version of Matlab must be R2016b or newer!');

%Check that requisite toolboxes needed are installed
prod_inf = ver;
assert(any(strcmp(cellstr(char(prod_inf.Name)), 'Statistics and Machine Learning Toolbox')),'Statistics Toolbox Not Installed!');
assert(any(strcmp(cellstr(char(prod_inf.Name)), 'Econometrics Toolbox')),'Econometrics Toolbox Not Installed!');
%assert(~isempty(which('fsolve')),'Function fsolve does not exist!'); % Can also check if specific functions exist like this
clear prod_inf;

%% Allow the user to select an old settings set (this still allows you to select a new folder of data to analyze)
if settingsSet.loadOldSettings
    disp('Select the old settings set...');
    [file,path] = uigetfile('*.mat');
    assert(~isequal(file,0),'"Load Old Settings" was selected, but no file was selected!');
    tempSet = load(fullfile(path,file));
    settingsSet = tempSet.settingsSet;
    clear tempSet file path
end

%% User selects the folder with data for analysis
%Mostly useful for Mac users who don't get GUI labels :(
disp('Select folder with dataset for analysis');

%Prompt user to select folder w/ pod data
settingsSet.analyzeDir=uigetdir(pwd,'Select folder with dataset for analysis');

%Throw an error if user hits "cancel"
assert(~isequal(settingsSet.analyzeDir,0), 'No data folder selected!');
disp(['Analyzing data in folder: ' settingsSet.analyzeDir '...'])

%% Review directories for files to analyze
%MODIFY THIS FUNCTION TO ALLOW THE LOADING OF NEW INSTRUMENT TYPES OTHER THAN U-PODS AND Y-PODS
[settingsSet.fileList, settingsSet.podList] = getFilesList(settingsSet.analyzeDir);

%% Create a folder for outputs to be saved into
settingsSet.outFolder=['Outputs_' datestr(now,'yymmddHHMMSS')]; %Create a unique name for the save folder
disp(['Creating output folder: ' settingsSet.outFolder '...']);
mkdir(settingsSet.analyzeDir, settingsSet.outFolder)
settingsSet.outpath = [settingsSet.analyzeDir,slash,settingsSet.outFolder]; %Store that file path for use later

%% Save out initial settings for reuse
disp('Saving settings structure...');
settingsPath = fullfile(settingsSet.outpath,'run_settings'); %Create file path for settings to save
save(char(settingsPath),'settingsSet'); %Save out settings


%% ------------------------------Read Pod Inventory and Deployment Log------------------------------
%The deployment log is used to determine when a pod was operational and
%what (if any) reference files that data is associated with
disp('Reading deployment log...');
settingsSet.deployLog = readDeployment(settingsSet.analyzeDir);

%The pod inventory is used to assign headers and therefore determine which columns contain required information.
%At a minimum, each pod should have an enrgy with labels 'temperature', 'humidity', 'datetime' or 'Unix time', and then containing the names of sensors entered above
disp('Reading pod inventory...');
settingsSet.podList = readInventory(settingsSet);



%% ------------------------------Convert Data to .mat Files as Needed------------------------------
%Import the Pod Data
disp('Converting unconverted data files to .mat files...');

%Convert pod files to .mat files
convertPodDatatoMat(settingsSet);

%Convert reference files to .mat files
convertRefDatatoMat(settingsSet);

%End the program if user has selected that they only want to convert files
assert(~settingsSet.convertOnly , 'Finished converting files, and "Convert Only" was selected');



%% These are the number of reference files, pods, regressions, validations, and folds to evaluate
nref   = length(settingsSet.fileList.colocation.reference.files.bytes); %Number of reference files
nPods  = size(settingsSet.podList.timezone,1); %Number of unique pods
nModels  = length(settingsSet.modelList); %Number of regression functions
nValidation   = length(settingsSet.valList); %Number of validation functions
nReps = settingsSet.nFoldRep;  %Number of folds to evaluate
nStats = length(settingsSet.statsList); %Number of statistical functions to apply
nPlots = length(settingsSet.plotsList); %Number of plotting functions

fprintf('*** Total number of loops to evaluate: %d *** \n Beginning...\n',nref*nPods*nModels*nValidation*nReps);


%--------------------------------------------------------------------------------------------------
%----------------------------------------------------------------------------------------------------
%----------------------------------------------------------------------------------------------------
%----------------------------------------------------------------------------------------------------
%--------------------------------------------------------------------------------------------------


%% --------------------------Fit Calibration Equations--------------------------
%% --------------------------START POD LOOP------------------------------
for j = 1:nPods
    
    %If statement checks whether to generate models and exits this loop if not
    if ~settingsSet.generateCal; break; end
    
    %Get current pod name for readability
    currentPod = settingsSet.podList.podName{j};
    
    %Keep track of the loop number in case it's needed by a sub function
    settingsSet.loops.j=j;
    fprintf('---Fitting models for pod: %s ...\n',currentPod);
    
    %% Load Pod Data
    fprintf('--Loading data for %s ...\n', currentPod);
    X_pod = loadPodData(settingsSet.fileList.colocation.pods, currentPod);
    %If no data was found for this pod, skip it
    if size(X_pod,1)==1; continue; end
    
    %Extract just columns needed
    disp('--Extracting important variables from pod data');
    [X_pod, xt] = dataExtract(X_pod, settingsSet, [settingsSet.podSensors settingsSet.envSensors]);
    
    %% --------------------------Pre-Process Pod Data (Filter, normalize, etc)------------------------------
    disp('--Applying selected pod data preprocessing...');
    settingsSet.filtering = 'pod';
    for jj = 1:length(settingsSet.podPreProcess)
        %Keep track of the loop number in case it's needed by a sub function
        settingsSet.loops.jj=jj;
        %Get string representation of function - this must match the name of a regression function
        preprocFunc = settingsSet.podPreProcess{jj};
        fprintf('---Applying pod preprocess function %s ...\n',preprocFunc);
        %Convert this string to a function handle to feed the pod data to
        preprocFunc = str2func(preprocFunc);
        
        %Apply the filter function
        [X_pod, xt] = preprocFunc(X_pod, xt, settingsSet);
        
        %Clear function for next loop
        clear preprocFunc
    end%pod preprocessing loop
    
    
    
    
    %% --------------------------START REFERENCE FILE LOOP------------------------------
    %Note, these reference files will be analyzed independently
    %Only the time and gas concentration will be extracted from each reference file
    %If you want to combine multiple colocations into one calibration, manually append the reference files into a single file
    for i = 1:nref
        
        %Keep track of the loop number in case it's needed by a sub function
        settingsSet.loops.i=i;
        
        %Create empty cell matrices to store fitted models and statistics for each combination
        fittedMdls = cell(nModels,nValidation,nReps);
        mdlStats = cell(nModels,nValidation,nStats);
        Y_hat.cal = cell(nModels,nValidation,nReps);
        Y_hat.val = cell(nModels,nValidation,nReps);
        valList = cell(nValidation,1);
        
        %% ------------------------------Get Reference Data------------------------------
        %Get the reference file to load
        %Indexing is weird if there's only one file
        if nref==1; reffileName = settingsSet.fileList.colocation.reference.files.name;
        else; reffileName = settingsSet.fileList.colocation.reference.files.name{i}; end
        currentRef = split(reffileName,'.');
        currentRef = currentRef{1};
        
        %Load the reference file into memory
        fprintf('-Importing reference file %s ...\n',reffileName);
        Y_ref = loadRefData(settingsSet);
        
        %Extract just the datestring and the gas of interest
        disp('-Extracting important variables from reference data');
        [Y_ref, yt] = dataExtract(Y_ref, settingsSet, settingsSet.refGas);
        
        %If this reference file did not contain any of the specified gases, skip the file
        if size(Y_ref,2)==0
            warning(['No pollutants found in ' reffileName ' continuing onto further reference files...']);
            clear Y_ref yt
            continue
        else
            %Get what pollutant is contained in that reference.  This may behave weirdly if there are multiple pollutants in a reference file
            fun=@(s)~isempty(regexpi(Y_ref.Properties.VariableNames{1},s));
            if strcmpi(settingsSet.refGas{1},'allcol')
                settingsSet.fileList.colocation.reference.files.pollutants{i} = 'all';
            else
                settingsSet.fileList.colocation.reference.files.pollutants{i} = settingsSet.refGas{cellfun(fun,settingsSet.refGas)};
            end
            clear fun
        end
        
        %% --------------------------Pre-process Reference Data (Filter, normalize, etc)--------------------------
        fprintf('-Pre-processing reference file: %s ...\n',reffileName);
        settingsSet.filtering = 'ref';
        for ii = 1:length(settingsSet.refPreProcess)
            settingsSet.loops.ii=ii;
            %Get string representation of function - this must match the name of a filter function
            filtFunc = settingsSet.refPreProcess{ii};
            fprintf('--Applying reference preprocess function %s ...\n',filtFunc);
            %Convert this string to a function handle to feed the pod data to
            filtFunc = str2func(filtFunc);
            
            %Save filtered reference data into Y
            [Y_ref,yt] = filtFunc(Y_ref, yt, settingsSet);
            
            %Clear for next loop
            clear filtFunc
        end%loop for preprocessing reference data
        
        %Match reference and Pod data based on timestamps
        disp('--Joining pod and reference data...');
        [Y, X, t] = alignRefandPod(Y_ref,yt,X_pod,xt,settingsSet);
        
        %Use deployment log to verify that only colocated data is included
        disp('--Checking data against the deployment log...');
        [Y, X, t] = deployLogMatch(Y,X,t,settingsSet,0);
        
        %Skip this run if there is no overlap between data and entries in deployment log
        if(isempty(t))
            warning(['No overlap between data and entries in deployment log for ' reffileName ' and ' currentPod ' this combo will be skipped!']);
            clear X t Y Y_ref yt
            continue
        end
        
        
        
        %% --------------------------START VALIDATION SETS LOOP------------------------------
        %Create a vector used to separate calibration and validation data sets
        for k = 1:nValidation
            %Keep track of the loop number in case it's needed by a sub function
            settingsSet.loops.k=k;
            %Get string representation of validation selection function
            validFunc = settingsSet.valList{k};
            fprintf('----Selecting validation set with function: %s ...\n',validFunc);
            %Convert this string to a function handle for the validation selection function
            validFunc = str2func(validFunc);
            
            %Run that validation function and get the list of points to fit/validate on for each fold
            valList{k} = validFunc(Y, X, t, settingsSet.nFolds);
            %Clear the validation selection function for tidyness
            clear validFunc
            
            
            
            %% --------------------------START REGRESSIONS LOOP------------------------------
            %Fit regression equations and validate them
            for m = 1:nModels
                %Keep track of the loop number in case it's needed by a sub function
                settingsSet.loops.m=m;
                fprintf('-----Fitting model: %s ...\n',settingsSet.modelList{m});
                
                %Get string representation of functions - this must match the name of a function saved in the directory
                modelFunc = settingsSet.modelList{m};
                %Convert this string to a function handle for the regression
                modelFunc = str2func(modelFunc);
                %Get the generation function for that regression
                %Note that the function must be set up correctly - see existing regression functions for an example
                fitFunc = modelFunc(1);
                %Get the prediction function for that regression
                applyFunc = modelFunc(2);
                %Clear the main regression function for tidyness
                clear modelFunc
                
                
                
                %% --------------------------START K-FOLDS LOOP------------------------------
                %For each repetition (fold) of the validation list, select the data and fit a regression to it
                for kk = 1:nReps
                    %Keep track of the loop number in case it's needed by a sub function
                    settingsSet.loops.kk=kk;
                    fprintf('------Using calibration/validation fold #%d ...\n',kk);
                    %Check that there is at least one value in the validation list for this fold
                    if ~any(valList{k}==kk)
                        warning('------No entries in validation list for this fold, skipping!')
                        Y_hat.cal{m,k,kk} = NaN;
                        Y_hat.val{m,k,kk} = NaN;
                        fittedMdls{m,k,kk} = NaN;
                        continue
                    end
                    
                    %% Fit the selected regression
                    %Also returns the estimates and fitted model details
                    %Indices for the regression model array are: (m=nRegs,k=nVal,kk=nFolds)
                    disp('-------Fitting model on calibration data...');
                    fittedMdls{m,k,kk} = fitFunc(Y(valList{k}~=kk,:), X(valList{k}~=kk,:), settingsSet);

                    %% Apply the fitted regression to the calibration data
                    disp('-------Applying the model to calibration data...');
                    Y_hat.cal{m,k,kk} = applyFunc(X(valList{k}~=kk,:),fittedMdls{m,k,kk},settingsSet);
                    
                    %% Apply the fitted regression to the validation data
                    disp('-------Applying the model to validation data...');
                    Y_hat.val{m,k,kk} = applyFunc(X(valList{k}==kk,:),fittedMdls{m,k,kk},settingsSet);
                    
                end %loop of calibration/validation folds
                clear calFunc valFunc
                
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
        
%%------------------------------SHOULD TRAIN A FINAL MODEL USING ALL DATA AFTER LOOKING AT EACH FOLD TO MAKE LIKELY BEST FIT FOR APPLICATION TO REAL FIELD DATA
        
        %% ------------------------------Create plots----------------------------------------
        disp('-----Plotting estimates and statistics...');
        for mm = 1:nPlots
            %Keep track of the loop number in case it's needed by a sub function
            settingsSet.loops.mm=mm;
            %Get string representation of function - this must match the name of a function
            plotFunc = settingsSet.plotsList{mm};
            fprintf('------Running plotting function %s ...\n',plotFunc);
            
            %Convert this string to a function handle to feed data to
            plotFunc = str2func(plotFunc);
            
            %Run the plotting function m=nRegs,k=nVal,kk=nFold
            plotFunc(t, X, Y, Y_hat,valList,mdlStats,settingsSet);
            
            %Save the plots if selected and then close them (reduces memory load and clutter)
            if settingsSet.savePlots && ishandle(1)
                temppath = [currentPod '_' currentRef '_' settingsSet.plotsList{mm}];
                temppath = fullfile(settingsSet.outpath,temppath);
                saveas(gcf,temppath,'jpeg');
                clear temppath
                close(gcf)
            end
            clear plotFunc
        end%loop of plotting functions
        
        
        %% ------------------------------Save info for each pod for future reference------------------------------
        %Save the estimates
        disp('---Saving estimates...');
        temppath = ['Estimates_' currentPod '_' currentRef];
        temppath = fullfile(settingsSet.outpath,temppath); %Create file path for estimates
        save(char(temppath),'Y_hat'); %Save out model estimates
        clear Y_hat
        
        %Save X, Y, and the validation list to make this reproducible
        disp('---Saving data used for fitting...');
        fittingStruct.valLists = valList;
        fittingStruct.Y = Y;
        fittingStruct.X = X;
        fittingStruct.t = t;
        temppath = ['FitData_' currentPod '_' currentRef];
        temppath = fullfile(settingsSet.outpath,temppath); %Create file path for estimates
        save(char(temppath),'fittingStruct'); %Save out assignments
        clear fittingStruct
        
        %Save the fitted model objects
        disp('---Saving fitted regression models...');
        temppath = ['fittedModels_' currentPod '_' currentRef];
        temppath = fullfile(settingsSet.outpath,temppath); %Create file path for fitted coefficients to save
        save(char(temppath),'fittedMdls'); %Save out fitted regression models
        clear fittedMdls currentRef temppath
        
        %Clear variables specific to this pod/reference combination
        clear Y X t valList
        
        %Clear temporary variables specific to each reference file
        clear Y_ref yt refFileName
    end%loop for each reference file
    
    %Clear temporary variables specific to each pod
    clear X_pod xt currentPod
end%loop for each pod

%% Save out final settings structure for future replication/application
disp('Saving settings structure...');
save(char(settingsPath),'settingsSet'); %Save out settings



%--------------------------------------------------------------------------------------------------
%----------------------------------------------------------------------------------------------------
%----------------------------------------------------------------------------------------------------
%----------------------------------------------------------------------------------------------------
%--------------------------------------------------------------------------------------------------

%% If no calibrations were generated on this run, have to load a set of fitted models from elsewhere
%This allows you to apply calibrations from a previous colocation to new data without needing to re-fit the models to the old colocation data
if ~settingsSet.generateCal
    %Remember where to save these new estimates
    newSavePath = settingsSet.outpath;
    
    %Get a list of unique pods in the new analysis folder
    newPodList = settingsSet.podList;
    
    %Save the list of plots for field data
    fieldPlots = settingsSet.fieldPlots;
    
    %Ask the user to select a file with the fitted models to apply
    disp('Select file with previous analysis...');
    settingspath = uigetdir(pwd,'Select folder with previous analysis outputs');
    assert(~isequal(settingsSet.outpath,0),'Error: no file selected, run ended'); %Check that file was selected
    
    %Load the old settings file into memory to make sure we duplicate all
    %of the filtering and other settings
    disp('Loading previously used settings...');
    tempFile = load([settingspath slash 'run_settings.mat']);
    %Extract the regressions used in the previous analysis
    loadedsettingsSet = tempFile.settingsSet;
    
    %Overwrite the old list of "field" files and plots
    loadedsettingsSet.fileList.field = settingsSet.fileList.field;
    loadedsettingsSet.fieldPlots = fieldPlots;
    
    %Get the number of original files, pods, etc. to align with the saved models structure
    nref   = length(loadedsettingsSet.fileList.colocation.reference.files.bytes); %Number of reference files
    nPods  = size(loadedsettingsSet.podList.timezone,1); %Number of unique pods
    nModels  = length(loadedsettingsSet.modelList); %Number of regression functions
    nValidation   = length(loadedsettingsSet.valList); %Number of validation functions
    nReps = loadedsettingsSet.nFoldRep;  %Number of folds to evaluate
    nPlots = length(loadedsettingsSet.fieldPlots);  %Number of field plots
    
    %Overwrite the settings set
    settingsSet = loadedsettingsSet;
    
    %Clear temporary variables
    clear tempFile settingspath loadedsettingsSet fieldPlots
else
    %Use the same paths as were used to generate the calibration
    newSavePath = settingsSet.outpath;
    newPodList = settingsSet.podList;
end%if statement to see if a calibration was generated earlier in this run


%% APPLICATION TO FIELD DATA HAS NOT BEEN FULLY TESTED (YET)
%% --------------------------Apply Fitted Equations to New/Uncolocated Data--------------------------
%This portion of code is only executed if "applyCal" was selected AND there are files in the "field" folder
if ~settingsSet.applyCal || size(settingsSet.fileList.field.pods.files,1)<1
    warning('Either applyCal was not selected, or there is no data in the "Field" folder to apply models to.  Run ended.');
else
    %% ------------------------------Apply models fitted to each pod used in the original calibration------------------------------
    for j = 1:nPods
        settingsSet.loops.j=j;
        currentPod = settingsSet.podList.podName{j};
        
        %% Check if this pod is in the new folder before continuing
        if ~any(strcmpi(settingsSet.podList.podName{j},newPodList.podName))
            %If this pod is not in the new folder, skip it
            warning(['Pod: ' settingsSet.podList.podName{j} ' was used in the original calibration but has no field data, so it was skipped.']);
            continue
        end
        
        %% Load pod data
        fprintf('--Loading data for %s ...\n', currentPod);
        X_field = loadPodData(settingsSet.fileList.field.pods, currentPod);
        
        %If no data was found for this pod in the "field" folder, skip it
        if size(X_field,1)==1; continue; end
        
        %Extract just columns needed
        disp('--Extracting important variables from pod data');
        [X_field, xt] = dataExtract(X_field, settingsSet, [settingsSet.podSensors settingsSet.envSensors]);
        
        
        %% --------Uses the same preprocessing functions as were used in the original calibration--------
        disp('--Applying selected pod data preprocessing...');
        settingsSet.filtering = 'pod';
        for jj = 1:length(settingsSet.podPreProcess)
            %Keep track of the loop number in case it's needed by a sub function
            settingsSet.loops.jj=jj;
            %Get string representation of function - this must match the name of a regression function
            preprocFunc = settingsSet.podPreProcess{jj};
            fprintf('---Applying pod preprocess function %s ...\n',preprocFunc);
            %Convert this string to a function handle to feed the pod data to
            preprocFunc = str2func(preprocFunc);
            
            %Apply the filter function
            [X_field, xt] = preprocFunc(X_field, xt, settingsSet);
            
            %Clear function for next loop
            clear filterFunc
        end%pod preprocessing loop
        
        %Use deployment log to verify that only field data is included.
        %NOTE: xt is passed to deployLogMatch where "Y" would normally be passed because there is no reference data
        disp('--Checking data against the deployment log...');
        [~, X_field, xt] = deployLogMatch(xt,X_field,xt,settingsSet,1);
        
        %Skip this pod/reference if there is no overlap between data and entries in deployment log
        if(isempty(xt))
            warning(['No matching entries in deployment log for these dates for a field deployment of pod: ' currentPod '. This pod will be skipped!']);
            continue
        end
        
        
        %% ------------------------------Each reference file that was used for calibration------------------------------
        for i = 1:nref
            settingsSet.loops.i=i;
            %Get the name of the reference file
            if nref==1; reffileName = settingsSet.fileList.colocation.reference.files.name;
            else; reffileName = settingsSet.fileList.colocation.reference.files.name{i}; end
            currentRef = split(reffileName,'.'); currentRef = currentRef{1};
            
            %Allow the user to skip applying models calibrated against this reference
            apply = input(['Apply models generated using reference: ' currentRef '? (1/0) ']);
            if ~apply;continue;end
            
            %% Load information from the original calibration
            try
                %Use a "try/catch" here in case something broke and the fitted models for this pod/reference combo weren't saved/ were deleted
                disp('--Loading previously fitted models...');
                %Use the same formatting for how models were saved out in the "generation" part of this code
                temppath = ['fittedModels_' currentPod '_' currentRef];
                temppath = fullfile(settingsSet.outpath,temppath);
                %Load the fittedMdls structure
                load(temppath);
                
                disp('--Loading previously generated estimates...');
                temppath = ['Estimates_' currentPod '_' currentRef];
                temppath = fullfile(settingsSet.outpath,temppath);
                %Load the Y_hat structure
                load(temppath);
                
                disp('--Loading data used for fitting...');
                temppath = ['FitData_' currentPod '_' currentRef];
                temppath = fullfile(settingsSet.outpath,temppath);
                %Load the data structure
                load(temppath);
                %Extract the data 
                valList = fittingStruct.valLists;
                Y_fit = fittingStruct.Y;
                X_fit = fittingStruct.X;
                t_fit = fittingStruct.t;
                %Append that data to the field data
                X = [X_field; X_fit];
                t = [xt; t_fit];
                Y = [table(NaN(length(xt),size(Y_fit,2)),'VariableNames',Y_fit.Properties.VariableNames);Y_fit];
                clear fittingStruct X_fit t_fit Y_fit
                
                clear temppath
            catch
                warning(['Problem loading the fitted models and estimates from reference: ' currentRef ' and pod: ' currentPod '. This combination will be skipped.']);
                continue
            end%Try statement for loading fitted models
            
            %Add a field to Y_hat where field estimates can be stored
            Y_hat.field = cell(nModels,nValidation,nReps);
            
            %% ------------------------------Each validation set used in the original calibration------------------------------
            for k = 1:nValidation
                %Allow the user to skip applying this model
                apply = input(['Apply models fitted on validation sets: ' settingsSet.valList{k} '? (1/0) ']);
                if ~apply;continue;end
                
                settingsSet.loops.k=k;
                
                %Add list of -1 to end of validation list to mark field data
                valList{k} = [-1*ones(length(xt),1); valList{k}];
                
                %% ------------------------------Each model that was fitted------------------------------
                for m = 1:nModels
                    %Allow the user to skip applying this model
                    apply = input(['Apply model ' settingsSet.modelList{m} '? (1/0) ']);
                    if ~apply;continue;end
                    
                    settingsSet.loops.m=m;
                    fprintf('--Applying fitted regression model: %s ...\n', settingsSet.modelList{m});
                    
                    %Get string representation of functions - this must match the name of a function saved in the directory
                    modelFunc = settingsSet.modelList{m};
                    %Convert this string to a function handle for the regression
                    modelFunc = str2func(modelFunc);
                    %Get the prediction function for that regression
                    applyFunc = modelFunc(2);
                    %Clear the main regression function for tidyness
                    clear modelFunc
                    
                    
                    %% ------------------------------Models fitted to each fold------------------------------
                    for kk=1:nReps
                        fprintf('--Applying calibrations fitted on validation sets: %s, fold # %i ...\n', settingsSet.valList{k},kk);
                        if isempty(fittedMdls{m,k,kk})
                            warning('Model was not fitted, skipping!');
                            Y_hat.field{m,k,kk} = NaN(size(X(valList{k}==-1,:),1),1);
                            continue
                        end
                        
                        apply = true; %Apply all fitted calibrations
                        %May want to let user decide about which models to apply (this will cause complications)
                        %apply = input(['Apply model ' settingsSet.modelList{m} ' fitted on validation sets: ' settingsSet.valList{k} ', fold #' kk '? (1/0) ']);
                        %if apply
                            Y_hat.field{m,k,kk} = applyFunc(X(valList{k}==-1,:),fittedMdls{m,k,kk},settingsSet);
                        %end
                    end
                end%loop of regressions
            end%loop of validation sets
            
            
            %% ------------------------------Create plots----------------------------------------
            disp('-----Plotting estimates and statistics...');
            for mm = 1:nPlots
                %Keep track of the loop number in case it's needed by a sub function
                settingsSet.loops.mm=mm;
                %Get string representation of function - this must match the name of a function
                plotFunc = settingsSet.fieldPlots{mm};
                fprintf('------Running plotting function %s ...\n',plotFunc);
                
                %Convert this string to a function handle to feed the pod data to
                plotFunc = str2func(plotFunc);
                
                %Run the plotting function m=nRegs,k=nVal,kk=nFold
                plotFunc(t, X, Y, Y_hat,valList,0,settingsSet);
                
                %Save the plots if selected and then close them (reduces memory load and clutter)
                if settingsSet.savePlots && ishandle(1)
                    temppath = [currentPod '_' currentRef '_field_' settingsSet.plotsList{mm}];
                    temppath = fullfile(newSavePath,temppath);
                    saveas(gcf,temppath,'jpeg');
                    clear temppath
                    close(gcf)
                end
                clear plotFunc
            end%loop of plotting functions
            
            
            %% ------------------------------Save info for each pod for future reference------------------------------
            disp('---Saving estimates...');
            currentRef = split(reffileName,'.');
            currentRef = currentRef{1};
            tempfile = ['Estimates_' currentPod '_' currentRef];
            tempfile = fullfile(newSavePath,tempfile); %Create file path for estimates
            save(char(tempfile),'Y_hat'); %Save out model estimates
            
            disp('---Saving field data...');
            fieldStruct.X = X;
            fieldStruct.t = t;
            tempfile = ['FieldData_' currentPod '_' currentRef];
            tempfile = fullfile(newSavePath,tempfile); %Create file path for estimates
            save(char(tempfile),'fieldStruct'); %Save out model estimates
            
            clear tempfile Y_hat fieldStruct X t Y
        end%loop of reference files
        clear X_field xt
    end%loop of pods
    settingsPath = fullfile(newSavePath,'run_settings'); %Create file path for settings to save
    save(char(settingsPath),'settingsSet'); %Save out settings
end%if statement to apply to field data

%% Run any final analysis functions here
disp('Running any post-processing functions...');
for z = 1:length(settingsSet.postProcesses)
    %Get string representation of functions - this must match the name of a function saved in the directory
    postFunc = settingsSet.postProcesses{z};
    disp(['Post Processing Function: ' postFunc '...'])
    %Convert this string to a function handle
    postFunc = str2func(postFunc);
    %Run that function
    postFunc(settingsSet);
    clear postFunc
end