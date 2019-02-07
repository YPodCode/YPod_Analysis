function [X,t] = addRollingVar(X, t, ~)
%Add a rolling variance measure spanning the past 5 minutes

%Calculate typical deviation between points for window
a = t(1:end-1);
b = t(2:end);
delta = median(b-a);

%Use 5 minute window
k = floor(minutes(5)/delta);

%Get array to hold values
varArray = table2array(X);

%Use moving variance function for each variable
for i = 1:size(varArray,2)
    varArray(:,i) = movvar(varArray(:,i),k,'omitnan');
end

%Assign nice variable names (e.g.: humidity -> humidity_var)
varArray = array2table(varArray);
for i = 1:size(varArray,2)
    varArray.Properties.VariableNames{i} = [X.Properties.VariableNames{i} '_var'];
end

%Join the two arrays
X = [X varArray];

end