function func = joannaNN(a)
%This model creates principal components from X and then fits a linear
%model to it including interaction terms.  NOTE: Data does NOT need to
%already be in PCA form when it is passed to this function
switch a
    case 1; func = @JONNfit;
    case 2; func = @JONNapply;
end

end
%--------------------------------------------------------------------------

%-----------------------------FIT FUNC-------------------------------------
function mdls = JONNfit(Y,X,settingsSet)

%%% Get the time array
%Training.unixtime = X.telapsed*60*60*24;

%% Use cell array formatting for inputs to network
% Training.unixtime = X.telapsed*60*60*24;
% X = table2array(X);
% Y = table2array(Y);
% x = cell(1,size(X,1));
% y_t = cell(1,size(X,1));
% for i = 1:size(X,1)
%     x{i} = X(i,:)';
%     y_t{i} = Y(i,:)';
% end
%% Use matrix formatting
x = table2array(X)';

%Check for existing GPR for this pod
if length(settingsSet.fileList.colocation.reference.files.bytes)==1; reffileName = settingsSet.fileList.colocation.reference.files.name;
else; reffileName = settingsSet.fileList.colocation.reference.files.name{settingsSet.loops.i}; end
currentRef = split(reffileName,'.');
currentRef = currentRef{1};

filename = [settingsSet.podList.podName{settingsSet.loops.j} currentRef 'NNsave.mat'];
regPath = fullfile(settingsSet.outpath,filename);

%% If NN has already been optimized for this pod, can skip the optimization, which is really slow
if exist(regPath,'file')==2
    %Load the previous analysis
    load(regPath);
else
    %Initialize a structure to hold the parameters for the fitted kernel
    nnstruct = zeros(2,size(Y,2));
    

    for i=1:size(Y,2)
        %% Get current y column
        y_t = table2array(Y(:,i))';
        
        nLayer1 = optimizableVariable('nLayer1',[1,min(size(X,2),40)],'Type','integer');%
        nLayer2 = optimizableVariable('nLayer2',[0,min(size(X,2),40)],'Type','integer');
        hyperparametersRF = [nLayer1; nLayer2];
        C = [y_t;x];
        rng(1)
        results = bayesopt(@(params)oobErrNN(params,C),hyperparametersRF,...
            'AcquisitionFunctionName','expected-improvement-plus','Verbose',0);
        
        bestHyperparameters = results.XAtMinObjective;
        
        nnstruct(1,i) = bestHyperparameters.nLayer1;
        nnstruct(2,i) =  bestHyperparameters.nLayer2;
        
        close all
    end
    %save the fitted hyper parameters for later runs
    save(char(regPath),'nnstruct');
end


%% Grow a neural net for each column of Y
mdls = cell(size(Y,2),1);
for i = 1:size(Y,2)
    %% Get current y column
    y_t = table2array(Y(:,i))';
    
    %% Choose a Training Function
    % For a list of all training functions type: help nntrain
    % 'trainlm' is usually fastest.
    % 'trainbr' takes longer but may be better for challenging problems.
    % 'trainscg' uses less memory. Suitable in low memory situations.
    tfunc = 'trainbr';% Bayesian Regularization backpropagation.
    
    %% Create the net
    %Define the number of nodes in each hidden layer
    n1 = nnstruct(1,i);
    n2 = nnstruct(2,i);
    if n2==0
        hiddenLayerSize = n1;
    else
        hiddenLayerSize = [n1 n2];
    end
    
    %Define the  training function
    trainFcn = tfunc;
    
    % Create a Fitting Network
    net = fitnet(hiddenLayerSize,trainFcn);
    
    %% Define training characteristics of the net
    net.trainParam.epochs = 5000; %1000 Maximum number of epochs to train
    net.trainParam.goal = 0.001; %0 Performance goal
    net.trainParam.mu = .5 ; %0.005 Marquardt adjustment parameter
    net.trainParam.max_fail  = 10;        %   0  Maximum validation failures
    %net.trainParam.mu_dec  = ;          % 0.1  Decrease factor for mu
    %net.trainParam.mu_inc  = ;           % 10  Increase factor for mu
    %net.trainParam.mu_max   = ;        % 1e10  Maximum value for mu
    %net.trainParam.min_grad  = ;       % 1e-7  Minimum performance gradient
    %net.trainParam.show   = ;           %  25  Epochs between displays
    %net.trainParam.showCommandLine = ; % false  Generate command-line output
    %net.trainParam.showWindow = true;      % true  Show training GUI
    %net.trainParam.time  = inf ;            % inf  Maximum time to train in seconds
    
    %% Choose Input and Output Pre/Post-Processing Functions
    % For a list of all processing functions type: help nnprocess
    net.input.processFcns = {'removeconstantrows','mapminmax'};
    net.output.processFcns = {'removeconstantrows','mapminmax'};
    
    
    %% Setup Division of Data for Training, Validation, Testing
    % % For a list of all data division functions type: help nndivide
    net.divideFcn = 'divideint';  % "help nndivision" for details
    % net.divideMode = 'sample';  % Divide up every sample
    % net.divideParam.trainRatio = 50/100;
    % net.divideParam.valRatio = 15/100;
    
    %% Choose a Performance Function
    % For a list of all performance functions type: help nnperformance
    net.performFcn = 'mse';  % Mean Squared Error
    
    %% Choose Plot Functions
    % For a list of all plot functions type: help nnplot
    net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
        'plotregression', 'plotfit'};
    %net.trainParam.mu_inc = 1;
    
    %% Train the Network
    rng(1)
    [net,tr] = train(net,x,y_t);
    
    %% Save the model
    mdls{i} = net;
end
end
%--------------------------------------------------------------------------

function oobErr = oobErrNN(params,C)
%oobErroobErrNN Trains neural net and estimates validation error
%   oobErr trains a random forest of 300 regression trees using the
%   predictor data in X and the parameter specification in params, and then
%   returns the out-of-bag quantile error based on the median. X is a table
%   and params is an array of OptimizableVariable objects corresponding to
%   the minimum leaf size and number of predictors to sample at each node.

% yname = C.Properties.VariableNames{end};
% randomForest = TreeBagger(300,C,yname,'Method','regression',...
%     'OOBPrediction','on','MinLeafSize',params.minLS,...
%     'NumPredictorstoSample',params.numPTS);
% oobErr = oobQuantileError(randomForest);


%% Number of hidden layers:
n1 = params.nLayer1;
%Size of those layers
n2 = params.nLayer2;
% Bayesian Regularization backpropagation.
tfunc = 'trainbr';

%% Create the net
%Define the number of nodes in each hidden layer
if n2==0
    hiddenLayerSize = n1;
else
    hiddenLayerSize = [n1 n2];
end

%Define the  training function
trainFcn = tfunc;

%% Create a Fitting Network
net = fitnet(hiddenLayerSize,trainFcn);

%% Define training characteristics of the net
net.trainParam.epochs = 5000; %1000 Maximum number of epochs to train
net.trainParam.goal = 0.001; %0 Performance goal
net.trainParam.mu = .5 ; %0.005 Marquardt adjustment parameter
net.trainParam.max_fail  = 10;        %   0  Maximum validation failures
%Choose Input and Output Pre/Post-Processing Functions
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};
%Setup Division of Data for Training, Validation, Testing
net.divideFcn = 'divideblock';  % "help nndivision" for details
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.1;
net.divideParam.testRatio = 0.2;
%Choose a Performance Function
net.performFcn = 'mse';  % Mean Squared Error
% %Choose Plot Functions
% net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
%     'plotregression', 'plotfit'};

%% Train the Network
y_t = C(1,:);
x = C(2:end,:);
rng(1)
[net,tr] = train(net,x,y_t);

%% Test the Network
y_hat = net(x);
% e = gsubtract(y_t,y_hat);
% performance = perform(net,y_t,y_hat);

%% Recalculate Training, Validation and Test Performance
% trainTargets = y_t .* tr.trainMask{1};
% valTargets = y_t .* tr.valMask{1};
testTargets = y_t .* tr.testMask{1};
% trainPerformance = perform(net,trainTargets,y_hat);
% valPerformance = perform(net,valTargets,y_hat);
oobErr = perform(net,testTargets,y_hat);

end



%---------------------------APPLY------------------------------------------
function y_hat = JONNapply(X,mdls,~)

x = table2array(X)';

y_hat = zeros(length(mdls),size(X,1));
for i=1:length(mdls)
    net = mdls{i};
    y_hat(i,:) = net(x);
end
y_hat = y_hat';
end
%--------------------------------------------------------------------------

