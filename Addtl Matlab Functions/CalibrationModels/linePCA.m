function func = linePCA(a)
%This model creates principal components from X and then fits a linear
%model to it including interaction terms.  NOTE: Data does NOT need to
%already be in PCA form when it is passed to this function
switch a
    case 1; func = @PCA_LinearInteractGen;
    case 2; func = @PCA_LinearInteractApply;
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function mdl = PCA_LinearInteractGen(Y,X,settingsSet)

X = makePCs(X, 1, settingsSet);
X = table2array(X);
mdl = cell(size(Y,2),1);
for i = 1:size(Y,2)
    y = table2array(Y(:,i));
    
    tempmdl =fitlm(X,y,'linear');
    mdl{i} = compact(tempmdl);
    clear tempmdl
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = PCA_LinearInteractApply(X,mdl,settingsSet)

X = makePCs(X, 1, settingsSet);
X = table2array(X);
y_hat = zeros(size(X,1),length(mdl));
for i = 1:length(mdl)
    y_hat(:,i) = predict(mdl{i},X);
end

end
%--------------------------------------------------------------------------
