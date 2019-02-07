function [X,t] = removeNaNs(X, t, ~)
%Remove rows with NaN values in any column of X. Also remove columns of X
%with all "NaN" values

%Convert to table for boolean operators
tempX = table2array(X);

%Remove columns of X that are entirely NaN first
badcols = all(isnan(tempX),1);
X(:,badcols) = [];

%Remove rows that contain any NaN values
boolList = any(isnan(tempX(:,~badcols)),2);
X(boolList,:)=[];
t(boolList,:)=[];

end

