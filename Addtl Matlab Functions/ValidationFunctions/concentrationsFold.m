function valList = concentrationsFold(Y, ~, ~, n)
%Select percentiles of data by concentration, starting by dropping the highest levels

refConc = table2array(Y(:,1));

%Number of points to validate on
n_val = floor(size(refConc,1)/n);

%Get ordered list of indices based on datetime column (in case t is not sorted)
[~, ordert] = sort(-1*refConc);

%Initialize list of outputs where the number = the fold to use that as validation
%(i.e. rows with valList=1 are validation for the 1st fold)
%Initialize the list as all in the last fold, so if the list is not perfectly divisible by "n", the last fold will be slightly larger, but every point will be assigned into a validation fold
valList = ones(size(Y,1),1)*n;
%Initialize a variable to hold the index of the starting index in each range
thisStart = 1;
for i = 1:n
    %Starting point
    thisEnd = n_val*i;
    
    %Using the sorted list, select n_val points for validation
    valList(ordert(thisStart:thisEnd)) = i;

    %Assign the end of this fold as the start of the next
    thisStart = thisEnd+1;
end

end