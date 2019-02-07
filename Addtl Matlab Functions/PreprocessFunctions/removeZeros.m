function [X,t] = removeZeros(X, t, settingsSet)

%Convert to table for boolean operators
tempX = table2array(X);

%Check if any values in each row are exactly -999
boolList = tempX==0;

%Get median value
medx = median(tempX);

%Replace zeros with small, constant value
for i = 1:size(X,2)
    if medx(i)==0
        tempX(boolList(:,i),i) = ones(sum(boolList(:,i)),1).*(1e-3);
    else
        tempX(boolList(:,i),i) = ones(sum(boolList(:,i)),1).*(medx(i)/1000);
    end
end
% %Remove those rows
% X(boolList==1,:)=[];
% t(boolList==1,:)=[];

X = array2table(tempX,'VariableNames',X.Properties.VariableNames);

end

