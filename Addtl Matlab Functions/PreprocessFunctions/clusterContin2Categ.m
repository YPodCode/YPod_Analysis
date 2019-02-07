function [X,t] = clusterContin2Categ(X, t, ~)
%Uses knn clustering to convert a matrix of continuous variables to categorical clusters

%Get a matrix for functions
clusterX = table2array(X);

%Normalize the matrix to improve clustering
for j = 1:size(clusterX,2)
    m = median(clusterX(:,j));
    d = std(clusterX(:,j));
    clusterX(:,j) = (clusterX(:,j)-m)/d;
end

%Maximum number of clusters to try
nmax = 12;
%Create matrix to hold within group sum of squares values
all_wss = zeros(nmax,1);

%Plot for analysis
nwide = ceil(nmax/4);
if nmax>=4;nhigh=4;else;nhigh=nmax;end
height = 250*nhigh;
width = 400*nwide;
figure('Position',[400 100 width height]);
%xpoints = 1:size(clusterX,1);
xpoints = datenum(t);
warning('off','stats:kmeans:MissingDataRemoved')
for k = 1:nmax%Include one cluster (base case)
    %Tries 5 replicates for each k to account for random initialization of centers
    [groups, ~, sumd] = kmeans(clusterX,k,'Replicates',5); 
    
    %Calculate within groups sum of squares
    all_wss(k) = sum(sumd);
    
    %Plot for visualization
    subplot(nhigh,nwide,k);
    mrkers = {'o','+','*','.','x','s','d','^','v','p','h'};
    for j = 1:size(clusterX,2)
        gscatter(xpoints,clusterX(:,j),groups,[],mrkers{mod(j,11)+1});
        title(['K = ' num2str(k) ' WSS = ' num2str(round(all_wss(k),2))]);
        hold on
    end
    
    
    clear sumd
end% loop that tries different # of centers

figure;
plot(1:nmax,all_wss,'-o');
title('Clusters WSS')
xlabel('Number of Clusters')
ylabel('Within Group Sum of Squares')

%Allow the user to determine how many clusters they would like to keep
keepK = input('How many clusters would you like to use?   ');
%mdl get the cluster centers, and y_hat gets the cluster assignments for
%each row of X.  Use 10 replicates to ensure good centering of clusters
rng(101); %THIS FIXES THE RANDOM NUMBER SEED FOR REPEATABLE RESULTS
[X, ~, ~] = kmeans(clusterX,keepK,'Replicates',10);
warning('on','stats:kmeans:MissingDataRemoved')

%Make X into a categorical type variable
X = categorical(X);

end