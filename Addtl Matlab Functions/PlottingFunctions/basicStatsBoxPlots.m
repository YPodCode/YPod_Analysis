function basicStatsBoxPlots(~,~,Y,~,~,mdlStats,settingsSet)
%Create boxplots of the RMSE that was calculated using the podRMSE function

plottableFuncs = {'podRMSE','podR2','podCorr'};
statnames = {'RMSE','R^2','Spearman''s Rho'};

%Get names of current pod, regression, and validation methods
podName = settingsSet.podList.podName{settingsSet.loops.j};
refName = Y.Properties.VariableNames{1};clear Y

nRegs = length(settingsSet.modelList); %Number of regression functions
nVal = length(settingsSet.valList); %Number of validation functions
nStats = length(settingsSet.statsList); %Number of statistical functions
nFolds = settingsSet.nFoldRep;  %Number of folds to evaluate

%% Allow for the plotting of several different stats whose calculation functions make outputs that work with this
for zz = 1:length(plottableFuncs)
    
    toPlot = plottableFuncs{zz};
    %Find what order the RMSE function was run in
    index = 0;
    for i=1:nStats
        if strcmpi(settingsSet.statsList{i},toPlot)
            %If this function matches the one being plotted, get the index and skip looping
            index = i;
            break
        end
    end
    
    %Check if the function was not run and skip the rest of the loop if so
    if index == 0
        warning(['Stats function ' toPlot ' was not run! ' statnames{zz} ' plotting skipped!']);
        continue
    end
    
    %Extract all values of calculated RMSE
    podStat = zeros(nRegs*nVal*nFolds*2,1);
    folds = zeros(nRegs*nVal*nFolds*2,1);
    regs = strings(nRegs*nVal*nFolds*2,1);
    vals = strings(nRegs*nVal*nFolds*2,1);
    calval = strings(nRegs*nVal*nFolds*2,1);
    ind = 1;
    
    for i = 1:nRegs %Calculated for each model
        for j = 1:nVal %Calculated on each validation set
            %Get just that array of statistics
            tempstats = mdlStats{i,j,index};
            
            for k = 1:nFolds %Calculated on each fold of that validation
                for z = 1:2 %Calculated for calibration data (1) or validation (2)
                    %Get this stat value
                    podStat(ind) = tempstats(k,z);
                    
                    %Also get characteristics for colors/plotting
                    folds(ind) = k;
                    treg = settingsSet.modelList{i};
                    
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
                    regs(ind) = treg;
                    vals(ind) = settingsSet.valList{j};
                    if z==1;calval(ind)='Train';else;calval(ind)='Test';end
                    
                    %Increment the loop index
                    ind = ind+1;
                end
            end
        end
    end
    %Convert to categorical variables for plotting
    regs = categorical(regs);
    vals = categorical(vals);
    calval=categorical(calval);
    
    %% Perform Plotting
    g(1,zz) = gramm('x',regs,'y',podStat,'color',vals,'lightness',calval);
    %g(1,zz).facet_grid(vals,[],'scale','free_y');
    g(1,zz).stat_boxplot();
    g(1,zz).set_names('x','Model', 'y',statnames{zz}, 'color','Validation','lightness','Set');
    g(1,zz).axe_property('TickDir','out', 'XGrid','on','Ygrid','on','GridColor',[0.5 0.5 0.5],'XTickLabelRotation',-45);
    g(1,zz).set_text_options('base_size',14);
    %g(1,zz).set_title(statnames{zz});
    if zz ~= length(plottableFuncs);g(1,zz).no_legend();end
    

end%Loop of plotable functions

%In case none of the statistical functions were run
if exist('g','var')==1
    %Draw the figure
    figure('Position',[50 50 1500 1000]);
    %figure('Position',get( groot, 'Screensize' ));
    g.set_title(['Statistical Variation for Pod: ' podName ', Reference: ' refName]);
    g.draw();
end%if

end%Function

