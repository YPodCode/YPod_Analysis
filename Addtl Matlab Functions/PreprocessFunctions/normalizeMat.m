function [X,t] = normalizeMat(X, t, ~)
%{
This function normalizes and centers data by subtracting the mean and then
dividing by the standard deviation
%}

%Get the variable names for cleaner code
varNames = X.Properties.VariableNames;

%Loop through each column in X
for i = 1:size(X,2)
    
    %Convert the column to an array for functionality
    tempX = table2array(X(:,i));
    fprintf('Normalizing column %s \n',varNames{i});
    
    try
        %Calculate the mean and std deviation
        meanx = mean(tempX,'omitnan');
        fprintf('Mean is %6.4f \n',meanx);
        stdx = std(tempX,'omitnan');
        fprintf('Std dev is %6.4f \n',stdx);
        
        %Normalize that array
        tempX = (tempX-meanx)/stdx;
    catch err
        warning('Could not normalize column %s',varNames{i});
    end
    
    %Re-assign into X
    X(:,i) = array2table(tempX);
    
    clear tempX meanx stdx
end

end