function valList = environClusterVal(~, X, ~, n)
%Select blocks by kmeans clustering on temperature and humidity. 
%NOTE: groupings may vary in the quantity of points selected, and 
%group 1 will be the most extreme conditions, and n will be the middle of
%the temperature and humidity ranges

warning('off','stats:kmeans:FailedToConvergeRep')

%"Normalize" X so all columns have similar variability
clusterX = zscore([X.temperature X.humidity]);

%Tries 5 replicates for each k to account for random initialization of centers
rng(1); %Set seed for reproducibility
[groups, c, ~] = kmeans(clusterX,n,'Replicates',10);

%Sort by "weirdness" of each group (distance of center from 0)
[~,I]=sort(sqrt(sum(c.^2,2)),'descend');

%Sort by quantity in each group (group 1 is largest, group "n" is smallest)
valList = groups;
for i = 1:n
    valList(groups==I(i))=i;
end

%Plot the results of clustering
figure;
gscatter(X.temperature,X.humidity,valList);
xlabel('Temperature');ylabel('Humidity');
legend(split(num2str(1:n),'  '),'Location','eastoutside');
title('Groups Identified by Clustering');

warning('on','stats:kmeans:FailedToConvergeRep')

end