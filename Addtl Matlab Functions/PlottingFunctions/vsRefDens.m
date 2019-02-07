function vsRefDens(~,~,Y,Y_hat,valList,~,settingsSet)
%Plots the validation and calibration estimates for each fold versus the
%reference values.  Plotted as density to see "clumping" of the data

%Get names of current pod, regression, and validation methods
podName = settingsSet.podList.podName{settingsSet.loops.j};
refName = Y.Properties.VariableNames{1};

nRegs = length(settingsSet.modelList); %Number of regression functions
nVal = length(settingsSet.valList); %Number of validation functions
nFolds = settingsSet.nFoldRep;  %Number of folds to evaluate

%% Get estimated values together
Y_hat_cal = Y_hat.cal; %Extract calibrated estimates
Y_hat_val = Y_hat.val; %Extract validation estimates
Y_hat = cell(2,nRegs,nVal,nFolds);
y_plot = Y_hat; %Have to make cell matrix to hold timeseries for each plot
Y_hat(1,:,:,:) = Y_hat_cal; %Dimensions: (i=cal/val, m=nRegs, k=nVal, kk=nFolds)
Y_hat(2,:,:,:) = Y_hat_val;
clear Y_hat_cal Y_hat_val

%Grouping variables
G.regs = strings(2,nRegs,nVal,nFolds);
G.vals = strings(2,nRegs,nVal,nFolds);
G.folds = ones(2,nRegs,nVal,nFolds);
G.calval = strings(2,nRegs,nVal,nFolds);

%% Get reference values together
Y = table2array(Y); %Converts to array for use in plotting

%% Assign values for grouping
G.calval(1,:,:,:)='Train';
G.calval(2,:,:,:)='Test';
for m=1:nRegs
    %Name of regression
    G.regs(:,m,:,:)=settingsSet.modelList{m};
    for k=1:nVal
        %Name of validation
        G.vals(:,:,k,:) = settingsSet.valList{k};
        for kk=1:nFolds
            %Grouping Variable and times for estimate cell array
            G.folds(:,:,:,kk)=kk;
            y_plot{1,m,k,kk} = Y(valList{k}~=kk);
            y_plot{2,m,k,kk} = Y(valList{k}==kk);
            
        end
    end
end

%Make into categorical values for plotting
G.regs = categorical(G.regs);
G.vals = categorical(G.vals);
G.calval = categorical(G.calval);

%% Do plotting
%Estimated Data
g = gramm('x',y_plot,'y',Y_hat);
g.facet_grid(G.regs,G.vals);
g.stat_bin2d('nbins',[40 40],'geom','image');
%g.set_point_options('base_size',5,'step_size',2,'markers',{'o' '>' 'p' 'd' '^' 'v' '>' '<' 's' 'h' '*' '+' 'x'});
g.set_names('row','M','column','Validation','x','Reference','y','Estimate');
g.axe_property('TickDir','out','XGrid','on','Ygrid','on','GridColor',[0.5 0.5 0.5]);
g.set_title('Estimated Concentrations');
g.geom_abline();

%Draw the figure
figure('Position',[200 200 1000 800]);
g.set_title(['Results for Pod: ' podName ', Reference: ' refName]);
g.draw();


end

