function valList = clusterVal(~, X, ~, n)
%Select blocks by kmeans clustering.  NOTE: groupings may vary
%wildly in the quantity of points selected, and group 1 will be the
%largest (validation set), and group n will be the smallest

warning('off','stats:kmeans:FailedToConvergeRep')

%"Normalize" X so all columns have similar variability
clusterX = table2array(X);
for i = 1:size(clusterX,2)
    meanx = mean(clusterX(:,i),'omitnan');
    stdx = std(clusterX(:,i),'omitnan');
    clusterX(:,i) = (clusterX(:,i) - meanx)/stdx;
end
%clusterX = normc(clusterX);

%Tries 5 replicates for each k to account for random initialization of centers
rng(1); %Set seed for reproducibility
[groups, ~, ~] = kmeans(clusterX,n,'Replicates',5);

%Sort by quantity in each group
counts = zeros(n,1);
for i = 1:n
    counts(i) = sum(groups==i);
end

[~,I] = sort(counts,'descend');

%Sort by quantity in each group (group 1 is largest, group "n" is smallest)
valList = groups;
for i = 1:n
    valList(groups==I(i))=i;
end

warning('on','stats:kmeans:FailedToConvergeRep')

end