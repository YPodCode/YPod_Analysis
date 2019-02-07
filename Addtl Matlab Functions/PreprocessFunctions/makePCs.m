function [X,t] = makePCs(X, t, settingsSet)
%{
This function normalizes X and then transforms the X matrix into principal
components for use in later regressions.  It does not modify the timeseries
vector, so this should be added to the X matrix before applying this
function if time should be included as a variable in the PCA
%}

%Check for existing PCA for this pod
filename = [settingsSet.podList.podName{settingsSet.loops.j} 'PCAsave.mat'];
regPath = fullfile(settingsSet.outpath,filename);
    
%% If PCA has already been computed for this pod, can skip the SVD, which can be computationally expensive
if exist(regPath,'file')==2
    %Load the previous analysis
    load(regPath);
    
    %Extract important values
    coeff = PCA.coeff;
    keepPCs = PCA.keep;
    varMeanMat = PCA.varMeanMat;
    
    %Convert X to matrix
    X = table2array(X);
    
    %Normalize X using original scaling factors
    normX = X;
    for i = 1:size(X,2)
        if varMeanMat(i,2)~=0
            tempX = X(:,i);
            meanx = varMeanMat(i,1);
            stdx = varMeanMat(i,2);
            tempX = (tempX-meanx)/stdx;
            normX(:,i) = tempX;
        end
    end
    
    %Remove columns with no variance or that caused an error when normalizing
    normX(:,varMeanMat(:,2)==0)=[];
    
    %Calculate the new principal components
    score = normX*(coeff')^-1;
    
    %Use the same number of PCs as before
    score = score(:,1:keepPCs);
    
else
    %% If there is no PCA file in the current output folder
    %Get the variable names for reporting errors
    varNames = X.Properties.VariableNames;
    nvars = length(varNames);
    
    %Convert X to matrix
    X = table2array(X);
    
    %Create matrix to hold normalized values
    normX = X;
    varMeanMat = zeros(size(X,2),2);
    
    %Loop through each column in X and normalize it
    for i = 1:size(X,2)
        
        %Analyze each column separately
        tempX = X(:,i);
        fprintf('Normalizing column %s \n',varNames{i});
        
        try
            %Calculate the mean and std deviation
            meanx = mean(tempX,'omitnan');
            stdx = std(tempX,'omitnan');
            
            %Normalize that array
            tempX = (tempX-meanx)/stdx;
            
            %Store mean and standard deviation for use later
            varMeanMat(i,1) = meanx;
            varMeanMat(i,2) = stdx;
        catch err
            warning('Could not normalize column %s',varNames{i});
            varMeanMat(i,1) = 0;
            varMeanMat(i,2) = 0;
        end
        
        %Assign into the normalized X matrix
        normX(:,i) = tempX;
        
        clear tempX meanx stdx
    end %loop normalizing each column of X

    %Remove columns with no variance or that caused an error when normalizing
    normX(:,varMeanMat(:,2)==0)=[];
    
    %Perform the PCA Analysis using the SVD algorithm
    [coeff,score,~,~,explained,~]=pca(normX,'Centered',false);
    
    nplot = min(20,size(X,2));
    
    %Plot the variance explained by each component
    figure('Position',[100 100 1000 500]);  subplot(1,2,1)
    scatter(1:nplot,explained(1:nplot));ylabel('% of Variance');
    yyaxis right; scatter(1:1:nplot,cumsum(explained(1:nplot)));ylabel('Cumulative Explained');
    title('Variance Explained for Each PC');xlabel('PC #');
    xticks([1:nplot]); grid on

    %Plot the PCs
    subplot(1,2,2)
    npcs = min(size(score,2),20);
    xpoints = 1:size(score,1);
    colors = bone(npcs+1);
    plot(xpoints,score(:,1),'Color',colors(1,:));
    hold on
    for j = 2:npcs
        plot(xpoints,score(:,j),'Color',colors(j,:));
        hold on
    end
    legend(split(num2str(1:npcs),'  '),'Location','eastoutside');
    
    %% Allow the user to determine how many PCs they would like to keep
    keepPCs = input('How many PCs would you like to use?   ');
    
    %Save image from PCs
    if settingsSet.savePlots
        currentPod = settingsSet.podList.podName{settingsSet.loops.j};
        temppath = [currentPod '_PC_Variance'];
        temppath = fullfile(settingsSet.outpath,temppath);
        saveas(gcf,temppath,'jpeg');
        clear temppath
        close(gcf)
    end
    if keepPCs>nvars;keepPCs=nvars;end % Don't allow the user to select more PCs than possible
    if keepPCs<=0;keepPCs=nvars;end %If they select a negative number, give them all of the PCs
    
    %Plot the contribution of each variable to the selected PCs
    sortC = sum(abs(coeff(:,1:keepPCs)),2);
    [~, sortC] = sort(sortC);
    figure('Position',[400 200 1200 800]); 
    subplot(1,2,1);
    barh(coeff(sortC,1:keepPCs),'histc'); colormap('hot');
    yticks(1:nvars);yticklabels(varNames(sortC)); grid('on'); ylim([1,nvars+1])
    xlabel('Weight of Variable');legend(split(num2str(1:keepPCs),'  '),'Location','southeast');
    title('Variable Contribution to Each PC');
    
    subplot(1,2,2);
    barh(coeff(sortC,1:keepPCs)','histc'); colormap('jet');
    yticks(1:nvars);yticklabels(split(num2str(1:keepPCs),'  ')); grid('on'); ylim([1,keepPCs+1]); ylabel('PC #')
    xlabel('Weight of Variable');legend(varNames(sortC),'Location','eastoutside');
    title('Variable Weights for Selected PCs');
    
    %Save image from PCs
    if settingsSet.savePlots
        temppath = [currentPod '_PC_Contribs'];
        temppath = fullfile(settingsSet.outpath,temppath);
        saveas(gcf,temppath,'jpeg');
        clear temppath
        close(gcf)
    end
    
    %Make structure with important PC info for applying to future data
    PCA.coeff = coeff;
    PCA.varMeanMat = varMeanMat;
    PCA.keep = keepPCs;
    
    %Save out PCA info for applying to future data
    save(char(regPath),'PCA');
    
end %If statement that tries to load the SVD from a previous run

%Write over X
X = score(:,1:keepPCs);
X = array2table(X);

end %Function





