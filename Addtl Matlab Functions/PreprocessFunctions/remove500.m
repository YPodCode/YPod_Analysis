function [X,t] = remove500(X, t, ~)
%Remove rows with values greater than 500

%Convert to table for boolean operators
tempX = table2array(X);

%Check if any values in each row are greater than 500
boolList = any(tempX>=500,2);

%Remove those rows
X(boolList==1,:)=[];
t(boolList==1,:)=[];

end

