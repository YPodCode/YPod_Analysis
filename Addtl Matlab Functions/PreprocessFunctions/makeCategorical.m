function [X,t] = makeCategorical(X, t, ~)
%Convert values into the "categorical" type to allow for categorical
%functions (classification trees, multinomial regression, etc).  This
%assumes that categories are simply numbered and that "0" is the reference
%category.

%Loop through each column of X and convert it into a categorical array that
%is not ordinal (has no logical order to it)
for i = 1:size(X,2)
    %Only need to convert columns that aren't already categorical 
    if ~iscategorical(X(:,i))
        %Convert this to a table
        tempX = table2array(X(:,i));
        %Convert that table to categorical
        tempX = categorical(tempX,'Ordinal',false);
        %Get the created categories
        xcats = categories(tempX);
        tempX = reordercats(tempX,[xcats(2:end);xcats(1)]);
        %Reorder assuming that "0" is the reference category
        X.(X.Properties.VariableNames{i}) = tempX;
    end
end

end

