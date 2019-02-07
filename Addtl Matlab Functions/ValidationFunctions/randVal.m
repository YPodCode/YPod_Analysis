function valList = randVal(Y, ~, ~, n)
%Randomly select rows to drop for each fold of validation

%Randomly select rows
rng(1); %THIS FIXES THE RANDOM NUMBER SEED FOR REPEATABLE RESULTS
dropList = randperm(size(Y,1));

%Number of points to validate on
n_val = floor((size(Y,1)-1)/n);

%Initialize list of outputs where the number = the fold to use that as validation
%(i.e. rows with valList=1 are validation for the 1st fold)
valList = ones(size(Y,1),1)*n;

for i = 1:n
    %Starting point
    start = n_val*(i-1)+1;
    
    %Using the random list, select n_val points for validation
    valList(dropList(start:start+n_val)) = i;
end

%Assign any unassigned values randomly (this does mean that some validation sets will be longer than others)
valList(valList==0) = randi(n,length(valList(valList==0)),1);
end

