function statObj = podSilhouette(X, Y, y_hat, valList, ~, settingsSet)
%The silhouette function compares the similarity of a point to other points
%within its group to its similarity to points outside of its group.  It
%ranges from -1 to 1, where 1 is a perfect match, and -1 is a terrible
%match.  Poor results may indicate too many or too few groupings

%Convert to arrays
X = table2array(X);
Y = table2array(Y);

%Get the number of folds that were validated on
nFolds = settingsSet.nFoldRep;

%Get the index of the current validation and regression to get the right estimates
k = settingsSet.loops.k;
m=settingsSet.loops.m;

%Loop through and plot silhouette plots (this is very slow with large datasets)
s=zeros(size(X,1),nFolds);
figure
%First Silhouette of Input Data
subplot(1,nFolds+1,1)
silhouette(X,Y);
title('Input Groupings')
xticks(linspace(-1,1,5)); grid on

%Then of estimates
for i=1:nFolds
    subplot(1,nFolds+1,i+1)
    silhouette(X(valList~=i,:),y_hat.cal{m,k,i});
    title(['Fold: ' num2str(i)])
    xticks(linspace(-1,1,5)); grid on
end

statObj = s;

end

