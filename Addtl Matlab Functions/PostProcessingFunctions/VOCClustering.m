function [C_hat, X_hat, X_all] = VOCClustering(settingsSet)
%This function loads a list of categorical class labels from a csv and then
%fits classification algorithms to predict that class using the estimated
%concentrations that were fitted during the main regression
%This was writted for Jacob Thorson's thesis work and hasn't been tested
%with datasets other than his own

%% Fix for running once on PC and once on Mac (or vise-versa)
if xor(settingsSet.ispc, ispc)
    disp('Original run was on different OS. Select the path again to get the right file directory');
    originalPath = uigetdir(pwd,'Select folder with previous analysis outputs');
    assert(~isequal(originalPath,0),'Error: no file selected, run ended'); %Check that file was selected
else
    originalPath = settingsSet.outpath;
end


%% Load the "reference" classification data
%Read in the data
disp('Select CSV containing reference categories')
[file, path] = uigetfile('*.csv');
assert(~isequal(file,0),'No file selected');
filePath = [path file];
opts = detectImportOptions(filePath,'Delimiter',',');
opts = setvartype(opts,opts.VariableNames,[{'char'} repmat({'categorical'},1,size(opts.VariableNames,2)-1)]);
opts.VariableNames{1} = 'datetime';
refData = readtable(filePath, opts);
refData.datetime=datetime(refData.datetime,'InputFormat','M/d/yy H:mm:ss');
refData = refData(~isnat(refData.datetime),:);
C_ref = table(refData.Category,'VariableNames',{'Category'});
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
nClassReps = nReps;
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
g.calval = strings(nPods,nRef,nRegs,nVal,nReps,2);g.calval(:,:,:,:,:,1)='Train';g.calval(:,:,:,:,:,2)='Test';
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
            treg = settingsSet.modelList{m};
            %% Re-name for thesis
            switch treg
                case 'fullLinear'
                    treg='FullLM';
                case 'linearSensors'
                    treg='SelectLM';
                case 'podStepLM'
                    treg='StepLM';
                case 'podRidge'
                    treg='RidgeLM';
                case 'podGPR'
                    treg='GaussProc';
                case 'podRFR'
                    treg='RandFor';
                case 'podNN'
                    treg='NeurNet';
                otherwise
                    treg='Derp';
            end
            g.regs(:,:,m,:,:,:) = treg;
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
        
        %Make plot of reference timeseries
        Y_temp  = table(Y);
        Y_temp.telapsed = ind;
        ct(isundefined(C_ref.Category),:)=[];
        C_ref(isundefined(C_ref.Category),:)=[];
        [Y_temp, C, t_temp] = alignRefandPod(Y_temp,t,C_ref,ct,settingsSet);
        telaps = Y_temp.telapsed;
        Y_temp = Y_temp.Y;
        C = table2array(C);
        t_temp = datenum(t_temp);
        tempvalist = valList{1};
        hr(i,1) = gramm('x',telaps,'y',Y_temp,'color',C);
        hr(i,1).geom_point('alpha',1);
        hr(i,1).set_point_options('base_size',5,'step_size',3,'markers',{'o' 'p' 'd' '^' 'v' '>' '<' 's' 'h' '*' '+' 'x'});
%         if i == nRef
            %hr(i,1).axe_property('YAxis.TickLabelFormat','%.2f');
            hr(i,1).set_names('x','','y',pollutant, 'color','Source');
%         else
%              hr(i,1).axe_property('XTickLabel',{});%
%              hr(i,1).set_names('x','','y',pollutant, 'color','Label');
%         end
        hr(i,1).set_layout_options('Position',[0 (i-1)/nRef 1.0 1/nRef]);
        hr(i,1).axe_property('TickDir','out','XGrid','on','Ygrid','on','GridColor',[0.5 0.5 0.5]);%,'xlim',[(min(t)-1) (max(t)+1)],'ylim',[(min(Y)*0.9) (max(Y)*1.1)]);
        hr(i,1).set_text_options('base_size',12,'facet_scaling',1.0,'label_scaling',1.0);
        hr(i,1).no_legend();
        clear currentRef pollutant currentRef reffileName C Y t valList
    end%reference loop
    %Draw the figure
    figure('Position',get( groot, 'Screensize' ));
    %hr.set_title('Reference Timeseries Colored by Simulated Pollutant Source')
    hr.draw();
    if settingsSet.savePlots && ishandle(1)
        temppath = [currentPod 'ReferenceTimeseries'];
        temppath = fullfile(originalPath,temppath);
        saveas(gcf,temppath,'jpeg');
        clear temppath
        close(gcf)
    end
    
    clear currentPod
end%pods loop
g.pods = categorical(g.pods);
g.regs = categorical(g.regs);
g.poll = categorical(g.poll);
g.vals = categorical(g.vals);
g.calval = categorical(g.calval);

%% Plot what the regression data looked like:

%Estimated Data vs Reference
gr = gramm('x',Y_plot,'y',Y_hat_plot,'color',g.reps,'lightness',g.calval,'subset',g.pods=='YPODE2');
gr.facet_grid(g.regs,g.poll,'column_labels',true,'scale','independent');
gr.geom_point();
gr.geom_abline();
gr.set_point_options('base_size',3,'step_size',3,'markers',{'o' 'p' 'd' '^' 'v' '>' '<' 's' 'h' '*' '+' 'x'});
%gr.set_datetick('x');
gr.set_names('row','R','column','Gas', 'x','Reference','y','Estimate', 'color','Fold','lightness','Set');
gr.axe_property('TickDir','out','XGrid','on','Ygrid','on','GridColor',[0.5 0.5 0.5]);%,'xlim',[(min(t)-1) (max(t)+1)],'ylim',[(min(Y)*0.9) (max(Y)*1.1)]);
%gr.set_layout_options('Position',[0 0 1.0 nRegs/(nRegs+2)]);
gr.set_text_options('base_size',14,'facet_scaling',1);
%Draw the figure
figure('Position',get( groot, 'Screensize' ));
%gr.set_title('Estimates for Pod E2');
gr.draw();
if settingsSet.savePlots && ishandle(1)
    temppath = 'Estimates4Classification';
    temppath = fullfile(originalPath,temppath);
    saveas(gcf,temppath,'jpeg');
    clear temppath
    close(gcf)
end

%Timeseries
gr = gramm('x',ind_plot,'y',Y_hat_plot,'color',g.regs,'marker',g.reps,'lightness',g.calval,'subset',(g.pods=='YPODE2' & g.reps~=3));
gr.facet_grid(g.poll,[],'column_labels',false,'scale','independent');
gr.geom_point('alpha',1);
gr.set_point_options('base_size',3,'step_size',3,'markers',{'o' 'p' 'd' '^' 'v' '>' '<' 's' 'h' '*' '+' 'x'});
%gr.set_datetick('x');
gr.set_names('row','Gas', 'x','Elapsed Time (days)','y','PPM', 'color','Regression','marker','Fold','lightness','Set');
gr.axe_property('TickDir','out','XGrid','on','Ygrid','on','GridColor',[0.5 0.5 0.5]);%,'xlim',[(min(t)-1) (max(t)+1)],'ylim',[(min(Y)*0.9) (max(Y)*1.1)]);
%gr.set_layout_options('Position',[0 0 1.0 nRegs/(nRegs+2)]);
gr.set_text_options('base_size',12,'facet_scaling',1);
%Draw the figure
figure('Position',get( groot, 'Screensize' ));
gr.set_title('Estimates for Pod E2');
gr.draw();

if settingsSet.savePlots && ishandle(1)
    temppath = 'Estimatestimeseries';
    temppath = fullfile(originalPath,temppath);
    saveas(gcf,temppath,'jpeg');
    clear temppath
    close(gcf)
end


%% Try classifying the data
classList = {'SVM_Classifier'};
%,'boostTree_Classifier','podPatternRecog','baggedClassTrees','podkNN_Class','LDA_Classifier',
nClass = length(classList);

%Grouping Variables
C_hat.Y.cal = cell(nPods,nRegs,nVal,nReps,nClass);
C_hat.Y.val = cell(nPods,nRegs,nVal,nReps,nClass);
C_hat.Y_hat.cal = cell(nPods,nRegs,nVal,nReps,nClass);
C_hat.Y_hat.val = cell(nPods,nRegs,nVal,nReps,nClass);
valList = cell(nPods,nRegs,nVal,nReps,nClass);
for j = 1:nPods
    for m=1:nRegs
        for k=1:nVal
            for kk=1:nReps
                %Get reference Y values
                Y = X_all{j,m,k,kk};
                yt = Y.datetime;
                keepCols = ~strcmpi(Y.Properties.VariableNames,'datetime');
                Y = Y(:,keepCols);
                %Y = normalizeMat(Y,yt,settingsSet);
                
                [Y, C, t] = alignRefandPod(Y,yt,C_ref,ct,settingsSet);
                
                valList{j,m,k,kk} = timeFold(C, Y, t, nClassReps);
                %valList{j,m,k,kk} = t>datetime(2018,7,1);
                tempvallist = valList{j,m,k,kk};
                
                %%Try different classification functions
                for zz = 1:nClass
                    %Get string representation of functions - this must match the name of a function saved in the directory
                    modelFunc = classList{zz};
                    %Convert this string to a function handle for the regression
                    modelFunc = str2func(modelFunc);
                    %Get the generation function for that regression
                    %Note that the function must be set up correctly - see existing regression functions for an example
                    calFunc = modelFunc(1);
                    %Get the prediction function for that regression
                    valFunc = modelFunc(2);
                    %Clear the main regression function for tidyness
                    clear modelFunc
                    
                    %Fit the selected regression
                    %Also returns the estimates and fitted model details
                    %Indices for the regression model array are: (i=nPods,m=nRegs,k=nVal,kk=nReps)
                    disp(['-------Fitting and applying model ' classList{zz} ' with reference Y_hat']);
                    cmodel = calFunc(C(tempvallist~=kk,:), Y(tempvallist~=kk,:), settingsSet);
                    
                    %Apply the fitted regression to the calibration data
                    C_hat.Y.cal{j,m,k,kk,zz} = valFunc(Y(tempvallist~=kk,:),cmodel,settingsSet);
                    
                    %Apply the fitted regression to the validation data
                    C_hat.Y.val{j,m,k,kk,zz} = valFunc(Y(tempvallist==kk,:),cmodel,settingsSet);
                    
                    close all
                end
                
                
                
                %% Repeat using estimated concentrations (Y_hat)
                Y_hat = X_hat{j,m,k,kk};
                yhatt = Y_hat.datetime;
                keepCols = ~strcmpi(Y_hat.Properties.VariableNames,'datetime');
                Y_hat = Y_hat(:,keepCols);
                [Y_hat, C, t] = alignRefandPod(Y_hat,yhatt,C_ref,ct,settingsSet);
                
                for zz = 1:nClass
                    %Get string representation of functions - this must match the name of a function saved in the directory
                    modelFunc = classList{zz};
                    %Convert this string to a function handle for the regression
                    modelFunc = str2func(modelFunc);
                    %Get the generation function for that regression
                    %Note that the function must be set up correctly - see existing regression functions for an example
                    calFunc = modelFunc(1);
                    %Get the prediction function for that regression
                    valFunc = modelFunc(2);
                    %Clear the main regression function for tidyness
                    clear modelFunc
                    
                    %Fit the selected regression
                    %Also returns the estimates and fitted model details
                    %Indices for the regression model array are: (j=nPods,m=nRegs,k=nVal,kk=nReps)
                    disp(['-------Fitting and applying model ' classList{zz} ' with estimated Y_hat']);
                    cmodel = calFunc(C(tempvallist~=kk,:), Y_hat(tempvallist~=kk,:), settingsSet);
                    
                    %Apply the fitted regression to the calibration data
                    C_hat.Y_hat.cal{j,m,k,kk,zz} = valFunc(Y_hat(tempvallist~=kk,:),cmodel,settingsSet);
                    
                    
                    Apply the fitted regression to the validation data
                    C_hat.Y_hat.val{j,m,k,kk,zz} = valFunc(Y_hat(tempvallist==kk,:),cmodel,settingsSet);
                    close all
                end
            end
        end
    end
end


%% Plot the classification results of C_hat using Y
cats = categories(table2array(C_ref));
nrefcats = size(cats,1);

for j = 1:nPods
    figure('Position',get( groot, 'Screensize' ));
    for m=1:nRegs
        for k=1:nVal
            for zz = 1:nClass
                for kk=1:settingsSet.loops.kk
                    tempvallist = valList{j,m,k,kk};
                    
                    C_hat_cal = C_hat.Y.cal{j,m,k,kk,zz};
                    C_hat_val = C_hat.Y.val{j,m,k,kk,zz};
                    
                    if kk==1
                        c_plot = [C(tempvallist~=kk,:); C(tempvallist==kk,:)];
                        c_hat = [C_hat_cal; C_hat_val];
                    else
                        c_hat = [c_hat; C_hat_cal; C_hat_val];
                        c_plot = [c_plot; C(tempvallist~=kk,:); C(tempvallist==kk,:)];
                    end
                    
                    
                end
                if ~iscategorical(c_hat)
                    c_hat = categorical(c_hat,cats);
                end
                %Calculate normalized confusion matrix
                %cm=confusionmat(yplot,yhat);
                predcats = categories(c_hat);
                nhatcats = size(predcats,1);
                confmat = zeros(nrefcats, nhatcats);
                countlist = ones(size(c_plot,1),1);
                c_plot = table2array(c_plot);
                for q = 1:nrefcats
                    for xx = 1:nhatcats
                        confmat(q,xx) = sum(countlist(c_plot==cats{q} & c_hat==predcats{xx}));
                    end
                end
                successrate = sum(diag(confmat))/sum(sum(confmat));
                cumcm = sum(confmat,2);
%                 for z = 1:size(confmat,1)
%                     confmat(z,:) = confmat(z,:)/cumcm(z);
%                 end
                
                %%Make the plot
                subplot(settingsSet.loops.m, nClass, (m-1)*nClass+zz)
                imagesc(confmat); colormap('bone');
                for w = 1:size(confmat,1)
                    for ww = 1:size(confmat,2)
                        text(w,ww,num2str(confmat(ww,w),'%5.0f'),'Color','b','HorizontalAlignment','center');
                    end
                end
                text(0,0,['Success Rate: ' num2str(successrate,'%5.2f')])
                colorbar;%colorbar('Ticks',[0,0.25,0.5,0.75,1]);
                yticks(1:length(cats));xticks(1:length(categories(c_hat)));grid on
                
                %Set titles
                if j==1 && m==1 && zz==1
                    temptitle=[settingsSet.podList.podName{j} ', C:' classList{zz} ', R: ' settingsSet.modelList{m}];
                elseif m==1
                    temptitle=['C:' classList{zz} ', R: ' settingsSet.modelList{m}];
                else
                    temptitle=['R: ' settingsSet.modelList{m}];
                end
                title(temptitle,'FontSize',8);
                %Set x labels
                if m==settingsSet.loops.m
                    xticklabels(categories(c_hat))
                    xtickangle(-90)
                    xlabel('Prediction')
                else
                    xticklabels({})
                end
                %Set y labels
                if zz==1
                    yticklabels(categories(c_plot))
                    ylabel('True Category');
                else
                    yticklabels({})
                end
                set(gca,'fontsize',8)
                hold on
                clear yhat yplot
            end
        end
    end
    if settingsSet.savePlots && ishandle(1)
        temppath = [settingsSet.podList.podName{j} 'ReferenceClassifications'];
        temppath = fullfile(originalPath,temppath);
        saveas(gcf,temppath,'jpeg');
        clear temppath
        close(gcf)
    end
end

%% Repeat plots for C_hat from Y_hat
%% All Validation estimates
for j = 1:nPods
    figure('Position',get( groot, 'Screensize' ));
    for m=1:nRegs
        for k=1:nVal
            for zz = 1:nClass
                for kk=1:settingsSet.loops.kk
                    tempvallist = valList{j,m,k,kk};
                    
                    C_hat_val = C_hat.Y_hat.val{j,m,k,kk,zz};
                    
                    if kk==1
                        c_plot = C(tempvallist==kk,:);
                        c_hat = C_hat_val;
                    else
                        c_hat = [c_hat; C_hat_val];
                        c_plot = [c_plot; C(tempvallist==kk,:)];
                    end
                    
                    
                end
                if ~iscategorical(c_hat)
                    c_hat = categorical(c_hat,cats);
                end
                %Calculate normalized confusion matrix
                %cm=confusionmat(yplot,yhat);
                predcats = categories(c_hat);
                nhatcats = size(predcats,1);
                confmat = zeros(nrefcats, nhatcats);
                countlist = ones(size(c_plot,1),1);
                c_plot = table2array(c_plot);
                for q = 1:nrefcats
                    for xx = 1:nhatcats
                        confmat(q,xx) = sum(countlist(c_plot==cats{q} & c_hat==predcats{xx}));
                    end
                end
                successrate = sum(diag(confmat))/sum(sum(confmat));
                cumcm = sum(confmat,2);
                for z = 1:size(confmat,1)
                    confmat(z,:) = confmat(z,:)/cumcm(z);
                end
                
                %%Make the plot
                subplot(settingsSet.loops.m, nClass, (m-1)*nClass+zz)
                imagesc(confmat); colormap('bone');
                for w = 1:size(confmat,1)
                    for ww = 1:size(confmat,2)
                        text(w,ww,num2str(confmat(ww,w),'%5.2f'),'Color','b','HorizontalAlignment','center');
                    end
                end
                text(0,0,['Success Rate: ' num2str(successrate,'%5.2f')])
                colorbar;%colorbar('Ticks',[0,0.25,0.5,0.75,1]);
                yticks(1:length(cats));xticks(1:length(categories(c_hat)));grid on
                
                %Set titles
                if m==1 && zz==1
                    temptitle=[settingsSet.podList.podName{j} ', C:' classList{zz} ', R: ' settingsSet.modelList{m}];
                elseif m==1
                    temptitle=['C:' classList{zz} ', R: ' settingsSet.modelList{m}];
                else
                    temptitle=['R: ' settingsSet.modelList{m}];
                end
                title(temptitle,'FontSize',8);
                %Set x labels
                if m==nRegs
                    xticklabels(categories(c_hat))
                    xtickangle(-90)
                    xlabel('Prediction')
                else
                    xticklabels({})
                end
                %Set y labels
                if zz==1
                    yticklabels(categories(c_plot))
                    ylabel('True Category');
                else
                    yticklabels({})
                end
                set(gca,'fontsize',8)
                hold on
                clear yhat yplot
            end%Classification model loop
        end%Validation sets loop
    end%Regression model loop
    if settingsSet.savePlots && ishandle(1)
        temppath = [settingsSet.podList.podName{j} 'EstValClassifications'];
        disp(['Saving: ' temppath]);
        temppath = fullfile(originalPath,temppath);
        saveas(gcf,temppath,'jpeg');
        clear temppath
        close(gcf)
    end
end%Pod loop
%% All Calibrated estimates
for j = 1:nPods
    figure('Position',get( groot, 'Screensize' ));
    for m=1:nRegs
        for k=1:nVal
            for zz = 1:nClass
                for kk=1:settingsSet.loops.kk
                    tempvallist = valList{j,m,k,kk};
                    
                    C_hat_cal = C_hat.Y_hat.cal{j,m,k,kk,zz};
                    
                    if kk==1
                        c_plot = C(tempvallist~=kk,:);
                        c_hat = C_hat_cal;
                    else
                        c_hat = [c_hat; C_hat_cal];
                        c_plot = [c_plot; C(tempvallist~=kk,:)];
                    end
                    
                    
                end
                if ~iscategorical(c_hat)
                    c_hat = categorical(c_hat,cats);
                end
                %Calculate normalized confusion matrix
                %cm=confusionmat(yplot,yhat);
                predcats = categories(c_hat);
                nhatcats = size(predcats,1);
                confmat = zeros(nrefcats, nhatcats);
                countlist = ones(size(c_plot,1),1);
                c_plot = table2array(c_plot);
                for q = 1:nrefcats
                    for xx = 1:nhatcats
                        confmat(q,xx) = sum(countlist(c_plot==cats{q} & c_hat==predcats{xx}));
                    end
                end
                successrate = sum(diag(confmat))/sum(sum(confmat));
                cumcm = sum(confmat,2);
                for z = 1:size(confmat,1)
                    confmat(z,:) = confmat(z,:)/cumcm(z);
                end
                
                %%Make the plot
                subplot(settingsSet.loops.m, nClass, (m-1)*nClass+zz)
                imagesc(confmat); colormap('bone');
                for w = 1:size(confmat,1)
                    for ww = 1:size(confmat,2)
                        text(w,ww,num2str(confmat(ww,w),'%5.2f'),'Color','b','HorizontalAlignment','center');
                    end
                end
                text(0,0,['Success Rate: ' num2str(successrate,'%5.2f')])
                colorbar;%colorbar('Ticks',[0,0.25,0.5,0.75,1]);
                yticks(1:length(cats));xticks(1:length(categories(c_hat)));grid on
                
                %Set titles
                if m==1 && zz==1
                    temptitle=[settingsSet.podList.podName{j} ', C:' classList{zz} ', R: ' settingsSet.modelList{m}];
                elseif m==1
                    temptitle=['C:' classList{zz} ', R: ' settingsSet.modelList{m}];
                else
                    temptitle=['R: ' settingsSet.modelList{m}];
                end
                title(temptitle,'FontSize',8);
                %Set x labels
                if m==nRegs
                    xticklabels(categories(c_hat))
                    xtickangle(-90)
                    xlabel('Prediction')
                else
                    xticklabels({})
                end
                %Set y labels
                if zz==1
                    yticklabels(categories(c_plot))
                    ylabel('True Category');
                else
                    yticklabels({})
                end
                set(gca,'fontsize',8)
                hold on
                clear yhat yplot
            end%Classification model loop
        end%Validation sets loop
    end%Regression model loop
    if settingsSet.savePlots && ishandle(1)
        temppath = [settingsSet.podList.podName{j} 'CalEstClassifications'];
        disp(['Saving: ' temppath]);
        temppath = fullfile(originalPath,temppath);
        saveas(gcf,temppath,'jpeg');
        clear temppath
        close(gcf)
    end
end%Pod loop

%% Per fold plots
%% Calibrated Classifications
for j = 1:nPods
    for kk=1:settingsSet.loops.kk
        figure('Position',get( groot, 'Screensize' ));
        for m=1:nRegs
            for k=1:nVal
                for zz = 1:nClass
                    
                    tempvallist = valList{j,m,k,kk};
                    
                    %Get estimated and "reference" sources
                    C_hat_cal = C_hat.Y_hat.cal{j,m,k,kk,zz};
                    c_plot = C(tempvallist~=kk,:);
                    c_hat = C_hat_cal;
                    
                    
                    
                    if ~iscategorical(c_hat)
                        c_hat = categorical(c_hat,cats);
                    end
                    %Calculate normalized confusion matrix
                    %cm=confusionmat(yplot,yhat);
                    predcats = categories(c_hat);
                    nhatcats = size(predcats,1);
                    confmat = zeros(nrefcats, nhatcats);
                    countlist = ones(size(c_plot,1),1);
                    c_plot = table2array(c_plot);
                    for q = 1:nrefcats
                        for xx = 1:nhatcats
                            confmat(q,xx) = sum(countlist(c_plot==cats{q} & c_hat==predcats{xx}));
                        end
                    end
                    successrate = sum(diag(confmat))/sum(sum(confmat));
                    cumcm = sum(confmat,2);
%                     for z = 1:size(confmat,1)
%                         confmat(z,:) = confmat(z,:)/cumcm(z);
%                     end
                    
                    %%Make the plot
                    subplot(settingsSet.loops.m, nClass, (m-1)*nClass+zz)
                    imagesc(confmat); colormap('bone');
                    for w = 1:size(confmat,1)
                        for ww = 1:size(confmat,2)
                            text(w,ww,num2str(confmat(ww,w),'%5.0f'),'Color','b','HorizontalAlignment','center');
                        end
                    end
                    text(0,0,['Success Rate: ' num2str(successrate,'%5.2f')])
                    colorbar;%colorbar('Ticks',[0,0.25,0.5,0.75,1]);
                    yticks(1:length(cats));xticks(1:length(categories(c_hat)));grid on
                    
                    %Set titles
                    if m==1 && zz==1
                        temptitle=[settingsSet.podList.podName{j} ', C:' classList{zz} ', R: ' settingsSet.modelList{m}];
                    elseif m==1
                        temptitle=['C:' classList{zz} ', R: ' settingsSet.modelList{m}];
                    else
                        temptitle=['R: ' settingsSet.modelList{m}];
                    end
                    title(temptitle,'FontSize',8);
                    %Set x labels
                    if m==nRegs
                        xticklabels(categories(c_hat))
                        xtickangle(-90)
                        xlabel('Prediction')
                    else
                        xticklabels({})
                    end
                    %Set y labels
                    if zz==1
                        yticklabels(categories(c_plot))
                        ylabel('True Category');
                    else
                        yticklabels({})
                    end
                    set(gca,'fontsize',8)
                    hold on
                    clear yhat yplot
                end%Classification model loop
            end%Validation sets loop
        end%Regression model loop
        if settingsSet.savePlots && ishandle(1)
            temppath = [settingsSet.podList.podName{j} 'fold' num2str(kk) 'EstCalClassifications'];
            disp(['Saving: ' temppath]);
            temppath = fullfile(originalPath,temppath);
            saveas(gcf,temppath,'jpeg');
            clear temppath
            close(gcf)
        end
    end
end%Pod loop

%% Validation Classifications
for j = 1:nPods
    for kk=1:settingsSet.loops.kk
        figure('Position',get( groot, 'Screensize' ));
        for m=1:nRegs
            for k=1:nVal
                for zz = 1:nClass
                    
                    tempvallist = valList{j,m,k,kk};
                    %Get estimated and "reference" sources
                    C_hat_val = C_hat.Y_hat.val{j,m,k,kk,zz};
                    c_plot = C(tempvallist==kk,:);
                    c_hat = C_hat_val;
                    
                    if ~iscategorical(c_hat)
                        c_hat = categorical(c_hat,cats);
                    end
                    %Calculate normalized confusion matrix
                    %cm=confusionmat(yplot,yhat);
                    predcats = categories(c_hat);
                    nhatcats = size(predcats,1);
                    confmat = zeros(nrefcats, nhatcats);
                    countlist = ones(size(c_plot,1),1);
                    c_plot = table2array(c_plot);
                    for q = 1:nrefcats
                        for xx = 1:nhatcats
                            confmat(q,xx) = sum(countlist(c_plot==cats{q} & c_hat==predcats{xx}));
                        end
                    end
                    successrate = sum(diag(confmat))/sum(sum(confmat));
                    cumcm = sum(confmat,2);
%                     for z = 1:size(confmat,1)
%                         confmat(z,:) = confmat(z,:)/cumcm(z);
%                     end
                    
                    %%Make the plot
                    subplot(settingsSet.loops.m, nClass, (m-1)*nClass+zz)
                    imagesc(confmat); colormap('bone');
                    for w = 1:size(confmat,1)
                        for ww = 1:size(confmat,2)
                            text(w,ww,num2str(confmat(ww,w),'%5.0f'),'Color','b','HorizontalAlignment','center');
                        end
                    end
                    text(0,0,['Success Rate: ' num2str(successrate,'%5.2f')])
                    colorbar;%colorbar('Ticks',[0,0.25,0.5,0.75,1]);
                    yticks(1:length(cats));xticks(1:length(categories(c_hat)));grid on
                    
                    %Set titles
                    if m==1 && zz==1
                        temptitle=[settingsSet.podList.podName{j} ', C:' classList{zz} ', R: ' settingsSet.modelList{m}];
                    elseif m==1
                        temptitle=['C:' classList{zz} ', R: ' settingsSet.modelList{m}];
                    else
                        temptitle=['R: ' settingsSet.modelList{m}];
                    end
                    title(temptitle,'FontSize',8);
                    %Set x labels
                    if m==nRegs
                        xticklabels(categories(c_hat))
                        xtickangle(-90)
                        xlabel('Prediction')
                    else
                        xticklabels({})
                    end
                    %Set y labels
                    if zz==1
                        yticklabels(categories(c_plot))
                        ylabel('True Category');
                    else
                        yticklabels({})
                    end
                    set(gca,'fontsize',8)
                    hold on
                    clear yhat yplot
                end%Classification model loop
            end%Validation sets loop
        end%Regression model loop
        if settingsSet.savePlots && ishandle(1)
            temppath = [settingsSet.podList.podName{j} 'fold' num2str(kk) 'EstValClassifications'];
            disp(['Saving: ' temppath]);
            temppath = fullfile(originalPath,temppath);
            saveas(gcf,temppath,'jpeg');
            clear temppath
            close(gcf)
        end
    end
end%Pod loop

end%Function

