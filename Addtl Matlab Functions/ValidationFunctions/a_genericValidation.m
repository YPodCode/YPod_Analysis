function valList = a_genericValidation(Y, X, t, n)
%Creates a list that is as long as X,Y,t and has values from 1:n.  Each set
%of data associated with a number in 1:n will be used as the holdout or
%validation dataset in a validation. These validation sets can select data
%for different folds using any criteria and do not have to be the same size

%An example validation selection method that would completely randomly
%assign values to a fold.  This would not guarantee that each validation
%set is the same size (which is okay)
valList = randi(n,size(X,1),1);

end