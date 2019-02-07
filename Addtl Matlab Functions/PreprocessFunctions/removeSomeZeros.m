function [X,t] = removeSomeZeros(X, t, ~)
%{
Reduces the number of points that are exactly zero to reduce over-training
on them
%}

%Convert to array for math
tempx = table2array(X);

%Get indices of zeros
tempx = tempx==0;
indices = [1:size(X,1)]';
indices = indices(tempx);

%Randomize selection of zeros
randselect = rand(size(indices,1),1);

%Remove 2/3 of the zero values
randselect = randselect<0.67;
indices = indices(randselect);

%Remove those points
X(indices,:)=[];
t(indices,:)=[];

end%function