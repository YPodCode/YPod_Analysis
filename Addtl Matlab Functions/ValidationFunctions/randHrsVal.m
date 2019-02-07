function valList = randHrsVal(~, X, t, n)
%Select random hour blocks for each fold

%Group data points by the hour
hrs = dateshift(t,'start','hour');
G = unique(hrs);

%Scramble the order of hours to select
rng(1); %Set seed for reproducibility
I = randperm(size(G,1));
G = G(I); clear I
hrsperfold = floor(size(G,1)/n);

%Initialize list of outputs where the number = the fold to use that as validation
%(i.e. rows with valList=1 are validation for the 1st fold)
%Initialize the list as all in the last fold, so if the list is not perfectly divisible by "n", the last fold will be slightly larger, but every point will be assigned into a validation fold
valList = ones(size(X,1),1)*n;

%Loop through each fold and assign values into it if they match the scrambled list of hours
groupn = 1;
for i = 1:n
    for j = 1:hrsperfold
        valList(hrs==G(groupn))=i;
        groupn=groupn+1;
    end
end

end