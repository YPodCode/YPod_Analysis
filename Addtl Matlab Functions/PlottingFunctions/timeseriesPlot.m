function timeseriesPlot(t,X,Y,Y_hat,valList,~,settingsSet)
%Plots the validation and calibration estimates for each fold

%Get names of current pod, regression, and validation methods
podName = settingsSet.podList.podName{settingsSet.loops.j};
refName = Y.Properties.VariableNames{1};

nRegs = length(settingsSet.modelList); %Number of regression functions
nVal = length(settingsSet.valList); %Number of validation functions
nFolds = settingsSet.nFolds;  %Number of folds to evaluate
nReps = settingsSet.nFoldRep;  %Number of folds actually used
nSets = numel(fieldnames(Y_hat)); %Number of estimate sets
setList = {'Train','Test','Field'};

%% Get reference values together
Y = table2array(Y); %Converts to array for use in plotting

%Add an elapsed time variable that doesn't include long gaps
[X, ~] = makeDiscontTElaps(X, t, settingsSet);
t = X.telapsed;
y_plot = cell(nVal,nFolds); %Dimensions: (i=cal/val, k=nVal, kk=nFolds)
tref_plot = cell(nVal,nFolds);

%Grouping variables
G.reffold = ones(nVal,nFolds)*(nReps+1);
G.refval = strings(nVal,nFolds);
G.refcalval = strings(nVal,nFolds);

%% Assign values for reference plot
for m=1:nRegs
    for k=1:nVal
        %Name of validation
        G.refval(k,:) = settingsSet.valList{k};
        for kk=1:nFolds
            %Calibration and validation timeseries for reference cell array
            G.reffold(:,kk) = kk;
            y_plot{k,kk} = Y(valList{k}==kk,1);
            tref_plot{k,kk} = t(valList{k}==kk);
        end
    end
end

%Make into categorical values for plotting
G.refval = categorical(G.refval);
G.refcalval = categorical(G.refcalval);

%% Get estimated values together
tempycal = Y_hat.cal;
tempyval = Y_hat.val;

if nSets == 3
    tempyfield = Y_hat.field;
    Y_hat = cell(nSets,nRegs,nVal,nReps);
    Y_hat(3,:,:,:) = tempyfield; %Extract field estimates if they've been generated
else
    Y_hat = cell(nSets,nRegs,nVal,nReps);%Dimensions: (i=cal/val, m=nRegs, k=nVal, kk=nFolds)
end
Y_hat(1,:,:,:) = tempycal; %Extract calibrated estimates 
Y_hat(2,:,:,:) = tempyval; %Extract validation estimates

%Have to make cell matrix to hold timeseries for each plot
t_plot = cell(nSets,nRegs,nVal,nReps); 

%Grouping variables
H.regs = strings(nSets,nRegs,nVal,nReps);
H.vals = strings(nSets,nRegs,nVal,nReps);
H.folds = ones(nSets,nRegs,nVal,nReps)*(nReps+1);
H.calval = strings(nSets,nRegs,nVal,nReps);

%% Assign values for estimate plot
for zz = 1:nSets
    H.calval(zz,:,:,:) = setList{zz};
end

for m=1:nRegs
    %Name of regression
    H.regs(:,m,:,:)=settingsSet.modelList{m};
    for k=1:nVal
        %Name of validation
        H.vals(:,:,k,:) = settingsSet.valList{k};
        for kk=1:nReps
            %Grouping Variable and times for estimate cell array
            H.folds(:,:,:,kk)=kk;
            t_plot{1,m,k,kk} = t(valList{k}~=kk & valList{k}~=-1);
            t_plot{2,m,k,kk} = t(valList{k}==kk);
            if nSets==3
                t_plot{3,m,k,kk} = t(valList{k}==-1);
            end
        end
    end
end
H.regs = categorical(H.regs);
H.vals = categorical(H.vals);
H.calval = categorical(H.calval);


%% Do plotting
%% Reference Values
g(1,1) = gramm('x',tref_plot,'y',y_plot,'color',G.reffold);% Create a gramm object
g(1,1).facet_grid([],G.refval);
g(1,1).geom_point();% Plot raw data as points
g(1,1).set_names('x',' ', 'y','Reference', 'column','Validation', 'color','Fold'); % Set appropriate names for legends
g(1,1).set_point_options('base_size',3,'step_size',3,'markers',{'o' 'p' 'd' '^' 'v' '>' '<' 's' 'h' '*' '+' 'x'});
%g(1,1).set_color_options('map','matlab');
%g(1,1).set_datetick('x'); %Make x-axis display dates/times instead of the datenum
g(1,1).axe_property('TickDir','out','XGrid','on','Ygrid','on','GridColor',[0.5 0.5 0.5]);%,'xlim',[(min(t)-1) (max(t)+1)],'ylim',[(min(Y)*0.9) (max(Y)*1.1)]);
%g(1,1).set_title('Reference Concentrations');% Set figure title
g(1,1).set_layout_options('Position',[0 nRegs/(nRegs+2) 1.0 1-nRegs/(nRegs+2)]);
g(1,1).set_text_options('base_size',10,'facet_scaling',1.2);

%% Estimated Data
g(2,1) = gramm('x',t_plot,'y',Y_hat,'color',H.folds,'lightness',H.calval);
g(2,1).facet_grid(H.regs,H.vals,'column_labels',false,'scale','free_y');
g(2,1).geom_point('alpha',1);
g(2,1).set_point_options('base_size',3,'step_size',3,'markers',{'o' 'p' 'd' '^' 'v' '>' '<' 's' 'h' '*' '+' 'x'});
%g(2,1).set_datetick('x');
g(2,1).set_names('row','M', 'x','Elapsed (days)', 'y','Estimate', 'column','Validation', 'color','Fold', 'Lightness','Set');
g(2,1).axe_property('TickDir','out','XGrid','on','Ygrid','on','GridColor',[0.5 0.5 0.5]);%,'xlim',[(min(t)-1) (max(t)+1)],'ylim',[(min(Y)*0.9) (max(Y)*1.1)]);
%g(2,1).set_title('Estimated Concentrations');
g(2,1).set_layout_options('Position',[0 0 1.0 nRegs/(nRegs+2)]);
g(2,1).set_text_options('base_size',10,'facet_scaling',1.0);

%% Draw the figure
%figure('Position',[200 200 1000 800]);
figure('Position',get( groot, 'Screensize' ));
g.set_title(['Timeseries for Pod: ' podName ', Reference: ' refName]);
g.draw();


end

