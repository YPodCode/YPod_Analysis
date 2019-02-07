function func = podLasso(a)
%This fits a purely linear model using all columns of X as predictors and
%also includes interaction terms between each variable
switch a
    case 1; func = @lassoFit;
    case 2; func = @lassoApply;
    case 3; func = @lassoReport;
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function mdl = lassoFit(Y,X,settingsSet)

xnames = X.Properties.VariableNames;
X = table2array(X);
mdl = cell(size(Y,2),1);
tcol=0;hcol=0;
normMat = zeros(size(X,2),2);
for i = 1:size(X,2)
    minx = min(X(:,i),[],'omitnan');
    maxx = max(X(:,i),[],'omitnan');
    normMat(i,1) = minx;
    normMat(i,2) = maxx;
    X(:,i) = 2*(X(:,i)-minx)./(maxx-minx)-1;
    %Find temperature and humidity columns
    currentCol = xnames{i};
    if any(regexpi(currentCol,'temperature'))
        tcol = i;
        continue;
    elseif any(regexpi(currentCol,'humidity'))
        hcol = i;
        continue;
    end
end
%Extract the temperature and humidity columns
tempDat = X(:,tcol);
humDat = X(:,hcol);

%Copy X for manipulation
X_int = X;
for i = 1:size(X,2)
    currentCol = xnames{i};
    if any(regexpi(currentCol,'temperature')) || any(regexpi(currentCol,'humidity'))
        continue;
    end
    %Calculate interaction terms
    t_int = X(:,i).*tempDat; tn = {[currentCol '_t']};
    h_int = X(:,i).*humDat; hn = {[currentCol '_h']};
    th_int =X(:,i).*tempDat.*humDat; thn = {[currentCol '_th']};
    %Append data
    X_int = [X_int t_int h_int th_int];
    %Keep track of names
    xnames = [xnames tn hn thn];
end


currentPod = settingsSet.podList.podName{settingsSet.loops.j};
nref   = length(settingsSet.fileList.colocation.reference.files.bytes);
if nref==1; reffileName = settingsSet.fileList.colocation.reference.files.name;
else; reffileName = settingsSet.fileList.colocation.reference.files.name{settingsSet.loops.i}; end
currentRef = split(reffileName,'.');
currentRef = currentRef{1};


for i = 1:size(Y,2)
    y = table2array(Y(:,i));
    [B,FitInfo] = lasso(X_int, y,'CV',5,'PredictorNames',xnames);
    idxLambda1SE = FitInfo.Index1SE;
    sparseModelPredictors = FitInfo.PredictorNames(B(:,idxLambda1SE)~=0);
    mdl{i} = lasso(X_int, y);
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = lassoApply(X,mdl,~)

X = table2array(X);
y_hat = zeros(size(X,1),length(mdl));
for i = 1:length(mdl)
    y_hat(:,i) = predict(mdl{i},X);
end
end
%--------------------------------------------------------------------------

%-------------Report relevant stats (coefficients, etc) about the model-------------
function lassoReport(fittedMdl,mdlStats,settingsSet)
fittedMdl
end