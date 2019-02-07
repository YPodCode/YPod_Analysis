function residsTrends(t,X,Y,Y_hat,valList,~,settingsSet)
%Plots the residuals versus each of the different predictor variables in X

%Get names of current pod, regression, and validation methods
podName = settingsSet.podList.podName{settingsSet.loops.j};
refName = Y.Properties.VariableNames{1};

nRegs = length(settingsSet.modelList); %Number of regression functions
nVal = length(settingsSet.valList); %Number of validation functions
nFolds = settingsSet.nFoldRep;  %Number of folds to evaluate
nSets = 2; %Number of estimate sets

%% Get estimated values together
Y_hat_cal = Y_hat.cal; %Extract calibrated estimates
Y_hat_val = Y_hat.val; %Extract validation estimates
Y_hat = cell(nSets,nRegs,nVal,nFolds); %Initialize estimates cell array
resids_plot = Y_hat; %Initialize array to hold residuals for each estimate
Y_hat(1,:,:,:) = Y_hat_cal; %Dimensions: (i=cal/val/field, m=nRegs, k=nVal, kk=nFolds,zz=nVars)
Y_hat(2,:,:,:) = Y_hat_val;
clear Y_hat_cal Y_hat_val

%% Get reference values together
X.datetime = datenum(t); %Add the datetime
X = [X Y]; %Join X and Y
covNames = X.Properties.VariableNames;
X = table2array(X);

%Make array to hold all covariates
nVars = size(X,2);  %Number of predictor variables in X

cov_plot = cell(nSets,nRegs,nVal,nFolds,nVars); %Hold all covariates
Y = table2array(Y); %Converts Y to array for use in calculating residuals

%Grouping variables
G.regs = strings(nSets,nRegs,nVal,nFolds);
G.vals = strings(nSets,nRegs,nVal,nFolds);
G.folds  =  ones(nSets,nRegs,nVal,nFolds);
G.calval=strings(nSets,nRegs,nVal,nFolds);
G.covs = strings(nSets,nRegs,nVal,nFolds,nVars);


%% Assign values for grouping
for yy = 1:nSets
    switch yy
        case 1; G.calval(yy,:,:,:,:)='Cal';
        case 2; G.calval(yy,:,:,:,:)='Val';    
    end
    
    for m=1:nRegs
        %Name of regression
        G.regs(:,m,:,:,:)=settingsSet.modelList{m};
        for k=1:nVal
            %Name of validation
            G.vals(:,:,k,:,:) = settingsSet.valList{k};
            for kk=1:nFolds
                %Grouping Variable and times for estimate cell array
                G.folds(:,:,:,kk,:)=kk;
                %Calculate residuals for each estimate
                resids_plot{1,m,k,kk} = Y_hat{1,m,k,kk} - Y(valList{k}~=kk);
                resids_plot{2,m,k,kk} = Y_hat{2,m,k,kk} - Y(valList{k}==kk);
                for zz=1:nVars
                    %Covariates
                    G.covs(:,:,:,:,zz) = covNames{zz};
                    cov_plot{1,m,k,kk,zz} = X(valList{k}~=kk,zz);
                    cov_plot{2,m,k,kk,zz} = X(valList{k}==kk,zz);
                end
                
            end
        end
    end
end


%Make into categorical values for plotting
G.regs = categorical(G.regs);
G.vals = categorical(G.vals);
G.calval = categorical(G.calval);
G.covs = categorical(G.covs);

%% Do plotting
%Plot PDF of residuals
g(1,1) = gramm('x',resids_plot,'color',G.vals,'lightness',G.calval);
g(1,1).set_names('x','Residual','y','density','row','Model','color','Validation','lightness','Cal/Val');
%g(1,1).stat_bin('geom','stacked_bar');
g(1,1).stat_density();
g(1,1).facet_grid(G.regs,[],'row_labels',false,'scale','fixed');
g(1,1).no_legend();
g(1,1).axe_property('XGrid','on','Ygrid','off','GridColor',[0.5 0.5 0.5]);
g(1,1).coord_flip();
%Then plot residuals versus each covariate including time and reference values
for i = 1:nVars
    g(1,i+1) = gramm('x',cov_plot(:,:,:,:,i),'y',resids_plot,'marker',G.folds,'lightness',G.calval,'color',G.vals);
    
    g(1,i+1).geom_point('alpha',1);
    g(1,i+1).set_point_options('base_size',3, 'markers',{'o' 'p' 'd' '^' 'v' '>' '<' 's' 'h' '*' '+' 'x'});
    g(1,i+1).set_names('row','M','x',covNames{i},'y','Residual',...
            'color','Validation','marker','Fold','lightness','Cal/Val');
    g(1,i+1).axe_property('TickDir','out','XGrid','on','Ygrid','on','GridColor',[0.5 0.5 0.5]);
    g(1,i+1).geom_hline('yintercept',0);
    
    if strcmpi(covNames{i},'datetime')
        g(1,i+1).set_datetick('x');
        %g(1,i+1).stat_glm();
        %g(1,i+1).stat_smooth('method','loess','lambda',0.2);
    else
        %g(1,i+1).stat_smooth('method','loess','lambda',0.2);
    end
    
    if i<nVars
        g(1,i+1).facet_grid(G.regs,[],'row_labels',false,'scale','fixed');
        g(1,i+1).no_legend();
    else
        g(1,i+1).facet_grid(G.regs,[],'row_labels',true,'scale','fixed');
    end
end


%Draw the figure
figure('Position',get( groot, 'Screensize' ));
g.set_title(['Correlation of Residuals with Predictors for Pod: ' podName ', Gas: ' refName]);
g.draw();


end

