function [X,t] = addDerivatives(X, t, ~)
%{
This function adds columns for each existing column's derivative.
Note: this does not do anything to handle gaps in the data, which could
cause anomolous values.  It also does this to every variable, so you may
need to remove some extra columns if you don't want every variable
%}

%Get the variable names for cleaner code
varNames = X.Properties.VariableNames;

%Get matrices to do work on
tempX = table2array(X);
X_dt = tempX(2:end,:);
tempt = posixtime(t);

%Loop through each column and each row in X
for i = 1:size(X,2)
    for j = 2:size(X,1)
        %Calculate the change from the previous / time difference between the two rows
        X_dt(j-1,i) = (tempX(j,i)-tempX(j-1,i))/(tempt(j)-tempt(j-1));
        
    end
    %Save the variable name
    varNames{i} = [varNames{i} '_dt'];
end

%Convert into table with new names
X_dt = array2table(X_dt,'VariableNames',varNames);
%Remove first row of X and t to make dimensions match
X = X(2:end,:);
t = t(2:end);
%Join tables and return
X = [X X_dt];
end