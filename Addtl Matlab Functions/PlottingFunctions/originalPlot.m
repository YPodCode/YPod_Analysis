function originalPlot(t,X,Y,y_plot,valList,~,settingsSet)
%Creates a similar plot to that produced by Ricardo's "camp_plotting" function

%Get names of current pod, regression, and validation methods
podName = settingsSet.podList.podName{settingsSet.loops.j};
refName = Y.Properties.VariableNames{1};

nRegs = length(settingsSet.modelList); %Number of regression functions
nVal = length(settingsSet.valList); %Number of validation functions
nFolds = settingsSet.nFoldRep;  %Number of folds to evaluate
calVal = 2; calvallist = {'Train','Test'};
refEst = 2;

%% Get reference values together
Y = table2array(Y); %Converts to array for use in plotting
t = datenum(t); %Convert to a datenum for plotting

%% Get estimated values together
%Extract calibrated estimates
Y_hat_cal = y_plot.cal;
%Extract validation estimates
Y_hat_val = y_plot.val; 
%Dimensions: (z=ref/est, i=cal/val, m=nRegs, k=nVal, kk=nFolds)
G.calVal = strings(calVal,nVal,nFolds);
G.fold = ones(calVal,nVal,nFolds);
G.valid = strings(calVal,nVal,nFolds);

H.refest = strings(refEst,calVal,nVal,nFolds);H.refest(1,:,:,:)='Ref';H.refest(2,:,:,:)='Est';H.refest=categorical(H.refest);
H.calVal = strings(refEst,calVal,nVal,nFolds);
H.fold = ones(refEst,calVal,nVal,nFolds);
H.valid = strings(refEst,calVal,nVal,nFolds);

%Find columns with covariates
t_col = 0; h_col = 0;
for i =1:size(X,2)
    if strcmpi(X.Properties.VariableNames{i},'temperature')
        t_col = i;
    elseif strcmpi(X.Properties.VariableNames{i},'humidity')
        h_col = i;
    end
end

%Calibration vs validation data
for zz = 1:calVal
    G.calVal(zz,:,:)=calvallist{zz};
    H.calVal(:,zz,:,:)=calvallist{zz};    
end
H.calVal=categorical(H.calVal);G.calVal=categorical(G.calVal);

%Get data and other plotting/grouping variables
for m=1:nRegs
    %Initialize temporary variables
    %For Y vs Y_hat
    y_hat_temp = cell(calVal,nVal,nFolds);
    y_temp = cell(calVal,nVal,nFolds);
    %For Timeseries
    t_temp = cell(refEst,calVal,nVal,nFolds);
    y_plot_temp = cell(refEst,calVal,nVal,nFolds);
    %For residuals
    resids_temp = cell(calVal,nVal,nFolds);
    %For covariates
    humid_temp = cell(calVal,nVal,nFolds);
    temp_temp = cell(calVal,nVal,nFolds);
    time_temp = cell(calVal,nVal,nFolds);
    for k=1:nVal
        G.valid(:,k,:)=settingsSet.valList{k};
        H.valid(:,:,k,:)=settingsSet.valList{k};
        for kk=1:nFolds
            G.fold(:,k,:)=kk;
            H.fold(:,:,k,:)=kk;
            
            for zz = 1:calVal
                if zz==1
                    tplot = t(valList{k}~=kk);
                    Yplot = Y(valList{k}~=kk);
                    hplot = table2array(X(valList{k}~=kk,h_col));
                    templot = table2array(X(valList{k}~=kk,t_col));
                    Yhatplot = Y_hat_cal{m,k,kk};
                elseif zz==2
                    tplot = t(valList{k}==kk);
                    Yplot = Y(valList{k}==kk);
                    hplot = table2array(X(valList{k}==kk,h_col));
                    templot = table2array(X(valList{k}==kk,t_col));
                    Yhatplot = Y_hat_val{m,k,kk};
                end
                
                %Timeseries for plots
                t_temp{1,zz,k,kk} = tplot;
                t_temp{2,zz,k,kk} = tplot;
                
                %Reference values for plots
                y_plot_temp{1,zz,k,kk} = Yplot;
                
                %Reference values for other plots
                y_temp{zz,k,kk} = Yplot;
                
                %Residuals
                resids_temp{zz,k,kk} = Yhatplot - Yplot;
                
                %Covariates
                humid_temp{zz,k,kk} = hplot;
                
                temp_temp{zz,k,kk} = templot;
                
                time_temp{zz,k,kk} = tplot;
            end
        end
    end
    %Convert to categorical variables
    H.valid = categorical(H.valid);G.valid = categorical(G.valid);
    
    %Estimate values for plots
    y_hat_temp(1,:,:) = Y_hat_cal(m,:,:);
    y_hat_temp(2,:,:) = Y_hat_val(m,:,:);
    y_plot_temp(2,1,:,:) = Y_hat_cal(m,:,:);
    y_plot_temp(2,2,:,:) = Y_hat_val(m,:,:);
    
    %% Make plots
    %Timeseries
    g(1,1) = gramm('x',t_temp,'y',y_plot_temp,...
        'color',H.refest,'lightness',H.calVal,'marker',H.fold);
    g(1,1).set_names('x',' ', 'y','Concentration',...
        'color','Source','marker','Fold', 'marker','Set');
    g(1,1).geom_point();
    g(1,1).set_point_options('base_size',3,'markers',{'o' 'p' 'd' '^' 'v' '>' '<' 's' 'h' '*' '+' 'x'});
    g(1,1).set_datetick('x');
    g(1,1).axe_property('TickDir','out','XGrid','on','Ygrid','on','GridColor',[0.5 0.5 0.5]);%,'xlim',[(min(t)-1) (max(t)+1)],'ylim',[(min(Y)*0.9) (max(Y)*1.1)]);
    g(1,1).set_layout_options('Position',[0 0.5 0.7 0.5]);%[left bottom width height]
    
    %Versus Reference
    g(1,2) = gramm('x',y_temp,'y',y_hat_temp, 'lightness',G.calVal,'marker',G.fold);
    g(1,2).set_names('x','Concentration', 'y','Estimate', 'lightness','Set');
    g(1,2).geom_point('alpha',1);
    g(1,2).set_point_options('base_size',3,'markers',{'o' 'p' 'd' '^' 'v' '>' '<' 's' 'h' '*' '+' 'x'});
    g(1,2).axe_property('TickDir','out','XGrid','on','Ygrid','on','GridColor',[0.5 0.5 0.5]);%,'xlim',[(min(t)-1) (max(t)+1)],'ylim',[(min(Y)*0.9) (max(Y)*1.1)]);
    g(1,2).set_layout_options('Position',[0.7 0.5 0.3 0.5]);%[left bottom width height]
    g(1,2).geom_abline();
    g(1,2).stat_glm();
    
    %PDF of residuals
    g(2,1) = gramm('x',resids_temp,'color',G.valid,'lightness',G.calVal);
    g(2,1).set_names('x','Residual','y','Frequency','color','Validation','lightness','Cal/Val');
    %g(2,1).stat_bin('geom','stacked_bar');
    g(2,1).stat_density();
    g(2,1).no_legend();
    g(2,1).set_layout_options('Position',[0.0 0.0 0.11 0.5]);%[left bottom width height]
    g(2,1).axe_property('XGrid','on','Ygrid','off','GridColor',[0.5 0.5 0.5]);
    g(2,1).coord_flip();
    g(2,1).no_legend();
    
    %Residuals vs covariates
    %Versus Reference
    g(2,2) = gramm('x',y_temp,'y',resids_temp, 'lightness',G.calVal,'marker',G.fold);
    g(2,2).set_names('x','Reference', 'y',' ', 'lightness','Set');
    g(2,2).geom_point('alpha',1);
    g(2,2).set_point_options('base_size',3,'markers',{'o' 'p' 'd' '^' 'v' '>' '<' 's' 'h' '*' '+' 'x'});
    g(2,2).axe_property('TickDir','out','XGrid','on','Ygrid','on','GridColor',[0.5 0.5 0.5]);%,'xlim',[(min(t)-1) (max(t)+1)],'ylim',[(min(Y)*0.9) (max(Y)*1.1)]);
    g(2,2).set_layout_options('Position',[0.11 0.0 0.22 0.5]);%[left bottom width height]
    g(2,2).geom_hline('yintercept',0);
    g(2,2).stat_glm();
    g(2,2).no_legend();
    
    %Versus Humidity
    g(3,2) = gramm('x',humid_temp,'y',resids_temp, 'lightness',G.calVal,'marker',G.fold);
    g(3,2).set_names('x','Humidity', 'y',' ', 'lightness','Set');
    g(3,2).geom_point('alpha',1);
    g(3,2).set_point_options('base_size',3,'markers',{'o' 'p' 'd' '^' 'v' '>' '<' 's' 'h' '*' '+' 'x'});
    g(3,2).axe_property('TickDir','out','XGrid','on','Ygrid','on','GridColor',[0.5 0.5 0.5]);%,'xlim',[(min(t)-1) (max(t)+1)],'ylim',[(min(Y)*0.9) (max(Y)*1.1)]);
    g(3,2).set_layout_options('Position',[0.33 0.0 0.22 0.5]);%[left bottom width height]
    g(3,2).geom_hline('yintercept',0);
    g(3,2).stat_glm();
    g(3,2).no_legend();
    
    %Versus Temperature
    g(4,2) = gramm('x',temp_temp,'y',resids_temp, 'lightness',G.calVal,'marker',G.fold);
    g(4,2).set_names('x','Temperature', 'y',' ', 'lightness','Set');
    g(4,2).geom_point('alpha',1);
    g(4,2).set_point_options('base_size',3,'markers',{'o' 'p' 'd' '^' 'v' '>' '<' 's' 'h' '*' '+' 'x'});
    g(4,2).axe_property('TickDir','out','XGrid','on','Ygrid','on','GridColor',[0.5 0.5 0.5]);%,'xlim',[(min(t)-1) (max(t)+1)],'ylim',[(min(Y)*0.9) (max(Y)*1.1)]);
    g(4,2).set_layout_options('Position',[0.55 0.0 0.22 0.5]);%[left bottom width height]
    g(4,2).geom_hline('yintercept',0);
    g(4,2).stat_glm();
    g(4,2).no_legend();
    
    %Versus Time
    g(5,2) = gramm('x',time_temp,'y',resids_temp, 'lightness',G.calVal,'marker',G.fold);
    g(5,2).set_names('x','Time', 'y',' ', 'lightness','Set');
    g(5,2).set_datetick('x');
    g(5,2).geom_point('alpha',1);
    g(5,2).set_point_options('base_size',3,'markers',{'o' 'p' 'd' '^' 'v' '>' '<' 's' 'h' '*' '+' 'x'});
    g(5,2).axe_property('TickDir','out','XGrid','on','Ygrid','on','GridColor',[0.5 0.5 0.5]);%,'xlim',[(min(t)-1) (max(t)+1)],'ylim',[(min(Y)*0.9) (max(Y)*1.1)]);
    g(5,2).set_layout_options('Position',[0.77 0.0 0.23 0.5]);%[left bottom width height]
    g(5,2).geom_hline('yintercept',0);
    g(5,2).stat_glm();
    
    
    
    %Draw the figure
    %figure('Position',[200 200 1000 800]);
    warning('off','stats:LinearModel:RankDefDesignMat');
    figure('Position',get( groot, 'Screensize' ));
    g.set_title(['Results for Pod: ' podName ', Reference: ' refName ', Model: ' settingsSet.modelList{m}]);
    g.draw();
    
    if settingsSet.savePlots
        temppath = [podName '_' refName '_' settingsSet.modelList{m} '_originalPlot'];
        temppath = fullfile(settingsSet.outpath,temppath);
        saveas(gcf,temppath,'jpeg');
        clear temppath
        close(gcf)
    end
    
end
warning('on','stats:LinearModel:RankDefDesignMat');
end

