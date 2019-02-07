function [X,t] = podPMF(X, ~, settingsSet)
%{
This function normalizes X and then changes your X matrix into principal
components for use in later regressions.  It does not modify the timeseries
vector, so this should be added to the X matrix before applying this
function if time should be included as a variable in the PCA
%}

%Remove bad first rows for now
X(1:100,:) = [];

%Check for existing PCA for this pod
filename = [settingsSet.podList.podName{settingsSet.loops.j} 'PMFsave.mat'];
regPath = fullfile(settingsSet.outpath,filename);

%% If there is no PCA file in the current output folder
%Get the variable names for reporting errors
varNames = X.Properties.VariableNames;
nvars = length(varNames);

%Convert X to matrix
X = table2array(X);

%Remove bad variables
torem = any(~isfinite(X),2);
X(torem,:) = []; clear torem

%Create matrix to hold normalized values
normX = X;
varMeanMat = zeros(size(X,2),2);
torem = false(size(X,2),1);

%Loop through each column in X and normalize it
for i = 1:size(X,2)
    
    %Analyze each column separately
    fprintf('Normalizing column %s \n',varNames{i});
    tempX = X(:,i);
    try
        %Calculate the mean and std deviation
        meanx = median(tempX,'omitnan');
        stdx = std(tempX,'omitnan');
        
        %Columns with zero variance are not useful
        assert(stdx~=0,['No variance, column ' varNames{i} ' skipped...']);
        
        %Normalize that array
        tempX = (tempX-meanx)/stdx;
        
        %Store mean and standard deviation for use later
        varMeanMat(i,1) = meanx;
        varMeanMat(i,2) = stdx;
        
        %Smooth the normalized array
        tempX = smooth(tempX,(100/size(tempX,1)),'lowess');
    catch err
        warning('Could not normalize column %s',varNames{i});
        torem(i) = true;
        tempX = zeros(size(X,1),1);
    end
    
    %Assign into the normalized X matrix
    normX(:,i) = tempX;
    
    clear tempX meanx stdx
end %loop normalizing each column of X

%Remove columns that could not be normalized
normX(:,torem) = [];
X(:,torem) = [];
varMeanMat(torem,:) = [];
varNames(torem) = [];
nvars = size(X,2)-1;

%Do PMF with different number of factors
D = zeros(nvars,1);
for i = 1:nvars
    disp(['Computing PMF with ' num2str(i) ' factors...']);
    [~,~,D(i)] = nnmf(normX,i,'replicates',5);
end

%Plot the results
[M,I]=min(D);
figure;plot(1:nvars,D);
set(gca, 'YScale', 'log')
title('RMSE Explained for N Factors');xlabel('Number of Factors');ylabel('Log RMSE');
grid on
displaytext = ['\leftarrow Min. RMSE: ' num2str(round(M,3)) ' with ' num2str(I) ' factors.'];
text(I,M,displaytext);

%Let the user select how many factors to use
nkeep = input('How many factors to use? ');

%Re-run PMF with that number of factors
[W,H,~] = nnmf(X,nkeep, 'replicates',5);

%Plot the Factors
colors=jet(nkeep);
fb = figure; %('Position',[400 100 400 200*nkeep])
xpoints = 1:size(W,1);
for j = 1:size(W,2)
    subplot(size(W,2),1,j);
    plot(xpoints,W(:,j),'Color',colors(j,:))
    title(['Factor #' num2str(j)])
end

%Plot the contribution of each variable to the selected PCs
% sortC = sum(abs(coeff(:,1:keepPCs)),2);
% [~, sortC] = sort(sortC);
figure('Position',[400 200 1200 800]);
subplot(1,2,1);
barh(H','histc'); colormap('hot');
yticks(1:nvars);yticklabels(varNames); grid('on'); ylim([1,nvars+1])
xlabel('Weight of Variable');legend(split(num2str(1:nkeep),'  '),'Location','southeast');
title('Variable Contribution to Each Factor');

subplot(1,2,2);
barh(H,'histc'); colormap('jet');
yticks(1:nvars);yticklabels(split(num2str(1:nkeep),'  ')); grid('on'); ylim([1,nkeep+1]); ylabel('Factor #')
xlabel('Weight of Variable');legend(varNames,'Location','southeast');
title('Variable Weights for Selected PCs');

%Close figures for tidyness
%close(fa);
%close(fb);
%close(fc);
    

end %Function





