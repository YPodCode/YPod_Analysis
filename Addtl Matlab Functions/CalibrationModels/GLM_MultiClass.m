function func = GLM_MultiClass(a)
%% Fits a GLM with binomial distribution to predict the presence of each class in Y.
switch a
    case 1; func = @podGLMMCGen;
    case 2; func = @podGLMMCApply;
    case 3; func = @podGLMMCReport;
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function glmClass = podGLMMCGen(Y,X,settingsSet)

xnames = X.Properties.VariableNames;

normMat = zeros(size(X,2),2);
for i = 1:size(X,2)
    tempX = table2array(X(:,i));
    minx = min(tempX,[],'omitnan');
    maxx = max(tempX,[],'omitnan');
    normMat(i,1) = minx;
    normMat(i,2) = maxx;
    
    if minx == maxx
        tempX = zeros(size(tempX));
    else
        tempX = (tempX-minx)./(maxx-minx);
    end
    
    X.(xnames{i}) = tempX;
end

if length(settingsSet.fileList.colocation.reference.files.bytes)==1; reffileName = settingsSet.fileList.colocation.reference.files.name;
else; reffileName = settingsSet.fileList.colocation.reference.files.name{settingsSet.loops.i}; end
currentRef = split(reffileName,'.');
currentRef = currentRef{1};

%This is a workaround for my VOC stuff because models were fit in a few
%ways that was weird
% if settingsSet.isRef
%     setName = 'ref';
% else
%     setName = 'est';
% end
clasName = 'raw';
glmClass = cell(size(Y,2),1);
warning('off')%,'stats:glmfit:BadScaling'

%Fit a predictive model for each class
figure('Position',[50 50 250*size(Y,2) 800])
for i = 1:size(Y,2)
%     varNames = [xnames ynames(i)];
%     glmClass{i} = fitglm(X,Y(:,i),'linear',...
%         'Distribution','binomial','VarNames',varNames);
    XY = [X Y(:,i)];
    disp(['Fitting binomial GLM to: ' Y.Properties.VariableNames{i}])
    rng(763)
    Mdl= stepwiseglm(XY, 'constant',... %Start with a interaction model and subtract from there
        'ResponseVar',Y.Properties.VariableNames{i},... %Make sure we use the right response variable
        'Criterion','bic',... %Use bayesian information criteria to select variables
        'lower','constant','Verbose',0,...
        'upper','interactions',... %Maximum complexity is a full interactions model
        'Distribution','binomial');%Assume Y follows gamma distribution %Use the R^2 to select additional terms
    
    if settingsSet.loops.kk==1
        subplot(1,size(Y,2),i)
        imp = abs(Mdl.Coefficients.tStat);
        bar(imp);
        title([Y.Properties.VariableNames{i} ' Terms']);
        ylabel('Magnitude of t-Statistics');
        xlabel('Predictors');
        h = gca;
        xticks(h,1:length(Mdl.Coefficients.Properties.RowNames));
        grid on
        h.XTickLabel = Mdl.Coefficients.Properties.RowNames;
        h.XTickLabelRotation = 45;
        h.TickLabelInterpreter = 'none';
    end
    %Keep a compact version of the model
    glmClass{i} = compact(Mdl);
end
try
    if settingsSet.loops.kk==1
        temppath = [settingsSet.podList.podName{settingsSet.loops.j} currentRef setName '_GLMCoeff'];
        temppath = fullfile(settingsSet.outpath,temppath);
        saveas(gcf,temppath,'jpeg');
        clear temppath
        close(gcf)
    end
catch
    warning('Image not saved!');
end

glmClass{end+1} = normMat;

warning('on')%,'stats:glmfit:BadScaling'
end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = podGLMMCApply(X,glmClass,~)

%Normalize
normMat = glmClass{end};

for i = 1:size(X,2)
    tempX = table2array(X(:,i));
    minx = normMat(i,1);
    maxx = normMat(i,2);
    
    if minx == maxx
        tempX = zeros(size(tempX));
    else
        tempX = (tempX-minx)./(maxx-minx);
    end
    
    X.(X.Properties.VariableNames{i}) = tempX;
end

%Initialize y_hat matrix
y_hat = zeros(size(X,1),length(glmClass)-1);
%X = table2array(X);

for i = 1:length(glmClass)-1
    %Make predictions on new data
    y_hat(:,i) = predict(glmClass{i},X);
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function podGLMMCReport(glmClass,~)
try
    glmClass
catch err
    disp('Error reporting the GLM classification model');
end

end