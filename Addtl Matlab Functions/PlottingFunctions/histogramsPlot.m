function histogramsPlot(~,~,Y,Y_hat,valList,~,settingsSet)
%Plots the histogram of reference values vs a histogram of the estimated values

%Get names of current pod, regression, and validation methods
podName = settingsSet.podList.podName{settingsSet.loops.j};
refName = Y.Properties.VariableNames{1};

nRegs = length(settingsSet.modelList); %Number of regression functions
nVal = length(settingsSet.valList); %Number of validation functions
nFolds = settingsSet.nFoldRep;  %Number of folds to evaluate
%nReps = settingsSet.nFoldRep;  %Number of folds actually used
%nSets = numel(fieldnames(Y_hat)); %Number of estimate sets

%% Get reference values together
Y = table2array(Y); %Converts to array for use in plotting

%% Get estimated values together
Y_hat_cal = Y_hat.cal; %Extract calibrated estimates
Y_hat_val = Y_hat.val; %Extract validation estimates

%% Grouping Variables
G.refs = strings(2,2,nRegs,nFolds);
G.refs(1,:,:,:) = 'Ref';
G.refs(2,:,:,:) = 'Est';
G.refs = categorical(G.refs);

G.sets = strings(2,2,nRegs,nFolds);
G.sets(:,1,:,:) = 'Train';
G.sets(:,2,:,:) = 'Test';
G.sets = categorical(G.sets);

G.folds = ones(2,2,nRegs,nFolds);
for z = 1:nFolds
    G.folds(:,:,:,z) = z;
end

G.regs = strings(2,2,nRegs,nFolds);
for z = 1:nRegs
    G.regs(:,:,z,:) = settingsSet.modelList{z};
end
G.regs = categorical(G.regs);

%% Loop and make plots
y_plot = cell(2,2,nRegs,nFolds);
for k = 1:nVal
    for m = 1:nRegs
        for kk = 1:nFolds
            y_plot{1,1,m,kk} = Y(valList{k}~=k,1);
            y_plot{1,2,m,kk} = Y(valList{k}==k,1);
            
            y_plot{2,1,m,kk} = Y_hat_cal{m,k,kk};
            y_plot{2,2,m,kk} = Y_hat_val{m,k,kk};
        end
    end
    g(1,k) = gramm('x',y_plot,'color',G.regs,'linestyle',G.refs,'lightness',G.sets);
    %g(1,k).stat_bin('geom','line');
    g(1,k).stat_density();
    g(1,k).set_line_options('styles',{'-' ':' '--' '-.'});
    %g(1,k).set_color_options('map','d3_20');
    if k~=nVal
        g(1,k).no_legend();
        g(1,k).facet_grid(G.folds,[],'row_labels',false);
    else
        g(1,k).facet_grid(G.folds,[],'row_labels',true);
    end
    g(1,k).set_names('x','Concentration','y','Density','row','Fold', 'color','Reg:', 'linestyle','Set','lightness','Set');
    g(1,k).set_title(settingsSet.valList{k});
end

%% Draw the figure
figure('Position',get( groot, 'Screensize' ));
g.set_title(['Distributions for Pod: ' podName ', Reference: ' refName]);
g.draw();

end

