function func = clusterAll(a)
%Warning: this function uses all columns of X to cluster - this can be
%extremely computationally expensive if X has many dimensions.  Try
%applying PCA or other dimensional reduction tools before applying this
%method to a large dataset
switch a
    case 1; func = @clusterAllGen;
    case 2; func = @clusterAllApply;
    case 3; func = @clusterAllReport;
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function mdlobj = clusterAllGen(Y,X,~)

%Get a matrix for functions
clusterX = table2array(X);

%Maximum number of clusters to try
nmax = 10;
%Create matrix to hold within group sum of squares values
all_wss = zeros(nmax,1);

%Plot for analysis
nwide = ceil(nmax/4);
if nmax>=4;nhigh=4;else;nhigh=nmax;end
height = 250*nhigh;
width = 400*nwide;
figure('Position',[400 100 width height]);
xpoints = 1:size(clusterX,1);
warning('off','stats:kmeans:MissingDataRemoved')
for k = 1:nmax%Include one cluster (base case)
    %Tries 5 replicates for each k to account for random initialization of centers
    [groups, ~, sumd] = kmeans(clusterX,k,'Replicates',5); 
    
    %Calculate within groups sum of squares
    all_wss(k) = sum(sumd);
    
    %Plot for visualization
    subplot(nhigh,nwide,k);
    gscatter(xpoints,table2array(Y),groups);
    title(['K = ' num2str(k) ' WSS = ' num2str(round(all_wss(k),2))]);
    
    clear sumd
end% loop that tries different # of centers

figure;
plot(1:nmax,all_wss,'-o');
title('Clusters WSS')
xlabel('Number of Clusters')
ylabel('Within Group Sum of Squares')

%Allow the user to determine how many clusters they would like to keep
keepK = input('How many clusters would you like to use?   ');
if isempty(keepK);keepK = input('Try again: how many clusters would you like to use?   ');
%mdl get the cluster centers, and y_hat gets the cluster assignments for
%each row of X.  Use 10 replicates to ensure good centering of clusters
[y_hat, mdl, ~] = kmeans(clusterX,keepK,'Replicates',10);
warning('on','stats:kmeans:MissingDataRemoved')

%Make things easier if Y is a categorical variable
if iscategorical(table2array(Y))
    catlist = categories(y_hat);
else
    catlist = {};
end
mdlobj = {mdl,catlist};
end%fitting function
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = clusterAllApply(X,mdlobj,~)

%"mdl" contains the locations of each cluster center and a list of categories (if Y was categorical)
C=mdlobj{1};
catlist = mdlobj{2};

%The size of this matrix is the number of clusters
k_centers = size(C,1);

%This will give a warning that it didn't converge, which is correct (we don't
%want it to iterate and find new centers, we want to assign to the fitted
%centers), so we temporarily turn off that warning
warning('off','stats:kmeans:FailedToConverge')
[y_hat] = kmeans(table2array(X),k_centers,'MaxIter',1,'Start',C);
warning('on','stats:kmeans:FailedToConverge')

%Make comparisons to the original Y easier if it's categorical
if ~isempty(catlist)
    y_hat = categorical(y_hat);
    %Assign the original fitted categories to y_hat
    y_hat = setcats(y_hat,catlist);
end

end%application function
%--------------------------------------------------------------------------

%-------------Report relevant stats (coefficients, etc) about the model-------------
function clusterAllReport(fittedMdl,mdlStats,settingsSet)
fittedMdl
end

