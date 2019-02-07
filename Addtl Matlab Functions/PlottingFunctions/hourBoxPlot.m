function hourBoxPlot(t,~,Y,Y_hat,valList,~,settingsSet)
%Plots the validation and calibration estimates for each fold

%Get names of current pod, regression, and validation methods
podName = settingsSet.podList.podName{settingsSet.loops.j};
refName = Y.Properties.VariableNames{1};

nRegs = length(settingsSet.modelList); %Number of regression functions
nVal = length(settingsSet.valList); %Number of validation functions
nFolds = settingsSet.nFoldRep;  %Number of folds to evaluate
nReps = settingsSet.nFoldRep;  %Number of folds actually used
nSets = 2;

%% Get reference values together
Y = table2array(Y); %Converts to array for use in plotting
t = hour(t); %Convert to datenums for plotting

%% Get estimated values together
Y_hat_cal = Y_hat.cal; %Extract calibrated estimates
Y_hat_val = Y_hat.val; %Extract validation estimates
Y_hat = cell(nSets,nRegs,nVal,nFolds);
Y_hat(1,:,:,:) = Y_hat_cal; %Dimensions: (i=cal/val, m=nRegs, k=nVal, kk=nFolds)
Y_hat(2,:,:,:) = Y_hat_val;
clear Y_hat_cal Y_hat_val

y_hat_plot = zeros(0,1);
y_plot = zeros(0,1);
t_plot = zeros(0,1);
calval = strings(0,1);
valids = strings(0,1);
regs = strings(0,1);
for m=1:nRegs
    for k=1:nVal
        for kk=1:nFolds
            %Get the fitted data for this model/validation/fold
            tempy_cal = Y_hat{1,m,k,kk};
            tempy_val = Y_hat{2,m,k,kk};
            
            %Append calibrated data
            y_hat_plot = [y_hat_plot; tempy_cal];
            y_plot = [y_plot; Y(valList{k}~=kk,1)];
            t_plot = [t_plot; t(valList{k}~=kk,1)];
            calval = [calval; repmat('Train',size(tempy_cal,1),1)];
            valids = [valids; repmat(settingsSet.valList{k},size(tempy_cal,1),1)];
            regs = [regs; repmat(settingsSet.modelList{m},size(tempy_cal,1),1)];
            
            %Append validation data
            y_hat_plot = [y_hat_plot; tempy_val];
            y_plot = [y_plot; Y(valList{k}==kk,1)];
            t_plot = [t_plot; t(valList{k}==kk,1)];
            calval = [calval; repmat('Test',size(tempy_val,1),1)];
            valids = [valids; repmat(settingsSet.valList{k},size(tempy_val,1),1)];
            regs = [regs; repmat(settingsSet.modelList{m},size(tempy_val,1),1)];
        end
    end
end
calval = categorical(calval);
valids = categorical(valids);
regs = categorical(regs);

%% Do plotting
%Reference Values
g(1,1) = gramm('x',t_plot,'y',y_plot,'color',calval);% Create a gramm object
g(1,1).facet_grid([],valids);
g(1,1).stat_boxplot();% Plot raw data as points
g(1,1).set_names('x',' ', 'y','Reference', 'column','Validation', 'color','Set', 'lightness','Set'); % Set appropriate names for legends
g(1,1).axe_property('TickDir','out','XLim',[0 24],'XTick',0:23,'XTickLabel',split(num2str(0:23),'  '),...
    'XGrid','on','Ygrid','on','GridColor',[0.5 0.5 0.5]);%,'xlim',[(min(t)-1) (max(t)+1)],'ylim',[(min(Y)*0.9) (max(Y)*1.1)]);
g(1,1).set_layout_options('Position',[0 nRegs/(nRegs+2) 1.0 1-nRegs/(nRegs+2)]);

%Estimated Data
g(2,1) = gramm('x',t_plot,'y',y_hat_plot,'color',calval);
g(2,1).facet_grid(regs,valids,'column_labels',false,'scale','free_y');
g(2,1).stat_boxplot();
g(2,1).set_names('row','M', 'x','Hour', 'y','Estimate', 'column','Validation', 'color','Set');
g(2,1).axe_property('TickDir','out','XLim',[0 24],'XTick',0:23,'XTickLabel',split(num2str(0:23),'  '),...
    'XGrid','on','Ygrid','on','GridColor',[0.5 0.5 0.5]);%,'xlim',[(min(t)-1) (max(t)+1)],'ylim',[(min(Y)*0.9) (max(Y)*1.1)]);
g(2,1).set_layout_options('Position',[0 0 1.0 nRegs/(nRegs+2)]);
g(2,1).set_text_options('facet_scaling',1.0);

%Draw the figure
%figure('Position',[200 200 1000 800]);
figure('Position',get( groot, 'Screensize' ));
g.set_title(['Hourly Concentrations for Pod: ' podName ', Reference: ' refName]);
g.draw();


end

