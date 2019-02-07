function XYCorrelations(t,X,Y,~,~,~,settingsSet)


%% Get pod and reference info for titles
currentPod = settingsSet.podList.podName{settingsSet.loops.j};

nref   = length(settingsSet.fileList.colocation.reference.files.bytes); %Number of reference files
if nref==1; reffileName = settingsSet.fileList.colocation.reference.files.name;
else; reffileName = settingsSet.fileList.colocation.reference.files.name{settingsSet.loops.i}; end
currentRef = split(reffileName,'.');
currentRef = currentRef{1};

%% Gather data and calculate correlations
Xtemp = X;
Ytemp = Y;
Ytemp.temperature = X.temperature;
Ytemp.humidity = X.humidity;
Ytemp.datetime = datenum(t);
XNames = Xtemp.Properties.VariableNames;
YNames = Ytemp.Properties.VariableNames;
Xtemp = table2array(Xtemp);
Ytemp = table2array(Ytemp);

npred = size(Xtemp,2);
nrefs = size(Ytemp,2);
pear = zeros(npred,nrefs);
kendall = zeros(npred,nrefs);
spear = zeros(npred,nrefs);

for zz = 1:npred
    for yy = 1:nrefs
        pear(zz,yy) = corr(Xtemp(:,zz),Ytemp(:,yy),'Type','Pearson');
        kendall(zz,yy) = corr(Xtemp(:,zz),Ytemp(:,yy),'Type','Kendall');
        spear(zz,yy) = corr(Xtemp(:,zz),Ytemp(:,yy),'Type','Spearman');
    end
end


%% Make color plots
figure('Position',get( groot, 'Screensize' ));
pr = subplot(1,3,1);
imagesc((pear));
colormap(pr, jet(20));colorbar('Ticks',[-1,-0.5,0,0.5,1]);
xticks(1:nrefs);xticklabels(YNames);xtickangle(45);
yticks(1:npred);yticklabels(XNames);
grid on
title('Pearson Correlations');ylabel('Sensor');xlabel('Reference')
ax=gca;
ax.FontSize = 8;

kd = subplot(1,3,2);
imagesc((kendall));
colormap(kd, jet(20));colorbar('Ticks',[-1,-0.5,0,0.5,1]);
xticks(1:nrefs);xticklabels(YNames);xtickangle(45);
yticks(1:npred);yticklabels(XNames);
grid on
title('Kendalls Tau Coefficient');xlabel('Reference')
ax=gca;
ax.FontSize = 8;

sp = subplot(1,3,3);
imagesc((spear));
colormap(sp, jet(20));colorbar('Ticks',[-1,-0.5,0,0.5,1]);
xticks(1:nrefs);xticklabels(YNames);xtickangle(45);
yticks(1:npred);yticklabels(XNames);
grid on
title('Spearmans Rho');xlabel('Reference')
ax=gca;
ax.FontSize = 8;

temppath = [currentPod '_' currentRef '_Correlation_Colorplots'];
temppath = fullfile(settingsSet.outpath,temppath);
saveas(gcf,temppath,'jpeg');
clear temppath
close(gcf)

%% Plot scatters
nplot = sum(sum((abs(spear)>0.5),2)>0);
iplot = 1;
nfigs = 1;
hf = figure('Position',get( groot, 'Screensize' ));
for zz = 1:npred
    if npred<=6
        nrow = npred;
    elseif zz<=(npred-6)
        nrow = 6;
    elseif iplot>(nrefs*nrow)
        nrow = npred-zz+1;
    end
    for yy = 1:nrefs
        
        % Save plots periodically
        if iplot>(nrefs*nrow)
            temppath = [currentPod '_' currentRef '_XYCorrelations' num2str(nfigs)];
            temppath = fullfile(settingsSet.outpath,temppath);
            saveas(gcf,temppath,'jpeg');
            clear temppath
            close(gcf)
            
            figure('Position',get(groot, 'Screensize' ));
            iplot = 1;
            nfigs = nfigs+1;
        end

        %ha(iplot) = subplot(nrow,nrefs,iplot);
        subplot(nrow,nrefs,iplot);
        if isnan(spear(zz,yy))
            plotcol = [1 1 1];
        else
            colind = spear(zz,yy);
            if colind==0;colind=1;end
            colmap = jet(100);
            plotcol = colmap(round(colind*50+50,0),:);
        end
        scatter(Ytemp(:,yy),Xtemp(:,zz),'.','MarkerEdgeColor',plotcol);

        
        if mod(iplot,nrefs)==1
            ylabel(XNames{zz})
        else
            yticklabels('')
        end
        
        if iplot > (nrow*nrefs-nrefs)
            xlabel(YNames{yy})
        else
            xticklabels('')
        end
        iplot = iplot+1;
    end
end
end
