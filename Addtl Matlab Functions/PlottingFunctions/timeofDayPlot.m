function timeofDayPlot(t,~,Y,Y_hat,valList,~,settingsSet)
%Plots the validation and calibration estimates for each fold against the time of day

%Get names of current pod, regression, and validation methods
podName = settingsSet.podList.podName{settingsSet.loops.j};
refName = Y.Properties.VariableNames{1};

nRegs = length(settingsSet.modelList); %Number of regression functions
nVal = length(settingsSet.valList); %Number of validation functions
nFolds = settingsSet.nFoldRep;  %Number of folds to evaluate
nReps = settingsSet.nFoldRep;  %Number of folds actually used
%nSets = numel(fieldnames(Y_hat)); %Number of estimate sets
nSets = 2;

%% Get reference values together
Y = table2array(Y); %Converts to array for use in plotting
tcol = round(datenum(t) - min(datenum(t)),0);
t = datenum(timeofday(t)); %Convert to the time of day for plotting
y_plot = cell(2,nVal,nFolds); %Dimensions: (i=cal/val, k=nVal, kk=nFolds)
tref_plot = cell(2,nVal,nFolds);
tref_color = cell(2,nVal,nFolds);

%Grouping variables
G.reffold = ones(2,nVal,nFolds)*(nReps+1);
G.refval = strings(2,nVal,nFolds);
G.refcalval = strings(2,nVal,nFolds);

%% Assign values for reference plot
G.refcalval(1,:,:) = 'Cal';
G.refcalval(2,:,:) = 'Val';
for m=1:nRegs
    for k=1:nVal
        %Name of validation
        G.refval(:,k,:) = settingsSet.valList{k};
        for kk=1:nFolds
            %Calibration and validation timeseries for reference cell array
            G.reffold(:,:,kk) = kk;
            %Reference value
            y_plot{1,k,kk} = Y(valList{k}~=kk,1);
            y_plot{2,k,kk} = Y(valList{k}==kk,1);
            %Time of day
            tref_plot{1,k,kk} = t(valList{k}~=kk);
            tref_plot{2,k,kk} = t(valList{k}==kk);
            %Actual datetime for color
            tref_color{1,k,kk} = tcol(valList{k}~=kk);
            tref_color{2,k,kk} = tcol(valList{k}==kk);
        end
    end
end

%Make into categorical values for plotting
G.refval = categorical(G.refval);
G.refcalval = categorical(G.refcalval);

%% Get estimated values together
Y_hat_cal = Y_hat.cal; %Extract calibrated estimates
Y_hat_val = Y_hat.val; %Extract validation estimates
Y_hat = cell(nSets,nRegs,nVal,nFolds);
t_plot = Y_hat; %Have to make cell matrix to hold timeseries for each plot
t_color = t_plot; %For coloring
Y_hat(1,:,:,:) = Y_hat_cal; %Dimensions: (i=cal/val, m=nRegs, k=nVal, kk=nFolds)
Y_hat(2,:,:,:) = Y_hat_val;
clear Y_hat_cal Y_hat_val

%Grouping variables
G.regs = strings(nSets,nRegs,nVal,nFolds);
G.vals = strings(nSets,nRegs,nVal,nFolds);
G.folds = ones(nSets,nRegs,nVal,nFolds)*(nReps+1);
G.calval = strings(nSets,nRegs,nVal,nFolds);

%% Assign values for estimate plot
G.calval(1,:,:,:)='Cal';
G.calval(2,:,:,:)='Val';
for m=1:nRegs
    %Name of regression
    G.regs(:,m,:,:)=settingsSet.modelList{m};
    for k=1:nVal
        %Name of validation
        G.vals(:,:,k,:) = settingsSet.valList{k};
        for kk=1:nFolds
            %Grouping Variable and times for estimate cell array
            G.folds(:,:,:,kk)=kk;
            t_plot{1,m,k,kk} = t(valList{k}~=kk);
            t_plot{2,m,k,kk} = t(valList{k}==kk);
            t_color{1,m,k,kk} = tcol(valList{k}~=kk);
            t_color{2,m,k,kk} = tcol(valList{k}==kk);
        end
    end
end
G.regs = categorical(G.regs);
G.vals = categorical(G.vals);
G.calval = categorical(G.calval);

%% Do plotting
%Reference Values
g(1,1) = gramm('x',tref_plot,'y',y_plot,'color',tref_color,'marker',G.reffold,'size',G.refcalval);% Create a gramm object
g(1,1).facet_grid([],G.refval);
g(1,1).set_continuous_color('colormap','copper');
g(1,1).geom_point();% Plot raw data as points
g(1,1).set_names('x',' ', 'y','Reference', 'column','Validation', 'color','Days','marker','Fold', 'size','Set'); % Set appropriate names for legends
g(1,1).set_point_options('base_size',3,'step_size',3,'markers',{'o' 'p' 'd' '^' 'v' '>' '<' 's' 'h' '*' '+' 'x'});
g(1,1).axe_property('TickDir','out','XLim',[0 1],'XTick',0:1/24:1,'XTickLabel',datestr(hours(0:24),'HH'),'XGrid','on','Ygrid','on','GridColor',[0.5 0.5 0.5]);
%g(1,1).set_title('Reference Concentrations');% Set figure title
%g(1,1).set_datetick('x'); %Make x-axis display dates/times instead of the datenum
g(1,1).set_layout_options('Position',[0 nRegs/(nRegs+2) 1.0 1-nRegs/(nRegs+2)]);

%Estimated Data
g(2,1) = gramm('x',t_plot,'y',Y_hat,'color',t_color,'marker',G.folds,'size',G.calval);
g(2,1).facet_grid(G.regs,G.vals,'column_labels',false,'scale','free_y');
g(2,1).set_continuous_color('colormap','copper');
g(2,1).geom_point('alpha',1);
g(2,1).set_point_options('base_size',3,'step_size',3,'markers',{'o' 'p' 'd' '^' 'v' '>' '<' 's' 'h' '*' '+' 'x'});
g(2,1).set_names('row','M', 'x','Hour', 'y','Concentration', 'column','Validation', 'color','Days','marker','Fold', 'size','Set');
g(2,1).axe_property('TickDir','out','XLim',[0 1],'XTick',0:1/24:1,'XTickLabel',datestr(hours(0:24),'HH'),'XGrid','on','Ygrid','on','GridColor',[0.5 0.5 0.5]);
%g(2,1).set_title('Estimated Concentrations');
%g(2,1).set_datetick('x');
g(2,1).set_layout_options('Position',[0 0 1.0 nRegs/(nRegs+2)]);

%Draw the figure
%figure('Position',[200 200 1000 800]);
figure('Position',get( groot, 'Screensize' ));
g.set_title(['Time of Day Plot for Pod: ' podName ', Reference: ' refName]);
g.draw();


end

