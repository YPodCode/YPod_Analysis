function func = podNN(a)
%This model creates principal components from X and then fits a linear
%model to it including interaction terms.  NOTE: Data does NOT need to
%already be in PCA form when it is passed to this function
switch a
    case 1; func = @NNFit;
    case 2; func = @NNPredict;
end

end
%--------------------------------------------------------------------------

%-----------------------------FIT FUNC-------------------------------------
function mdls = NNFit(Y,X,settingsSet)
%Convert from table for functions
x = table2array(X);

%Add some lagged variables if the matrix is relatively small
sensorList = contains(X.Properties.VariableNames,settingsSet.podSensors,'IgnoreCase',true);
if sum(sensorList)<5
    %Decide how many minutes to lag.  This is arbitrary.  If you change
    %this, make sure to fix the predict function to match.
    nLags = ceil(minutes(5)/minutes(settingsSet.timeAvg));
    
    %Get Environmental and other non-sensor Variables
    envX = x(nLags+1:end,~sensorList);
    
    %Get sensor variables to lag
    toLagX = x(:,sensorList);
    
    %Make Lag matrix
    toLagX = lagmatrix(toLagX,0:nLags);
    
    %Have to remove rows of NaNs
    toLagX = toLagX(nLags+1:end,:);
    
    %Re-combine lagged sensors and environmental parameters
    x = [toLagX, envX];
    
    %Add zeros to the start to avoid size mismatch
    x = [zeros(nLags,size(x,2));x];
    
end

%Neural nets want columns to be instances for some reason
x = x';

%Check for existing optimized network topology for this pod
if length(settingsSet.fileList.colocation.reference.files.bytes)==1; reffileName = settingsSet.fileList.colocation.reference.files.name;
else; reffileName = settingsSet.fileList.colocation.reference.files.name{settingsSet.loops.i}; end
currentRef = split(reffileName,'.');
currentRef = currentRef{1};

filename = [settingsSet.podList.podName{settingsSet.loops.j} currentRef 'NNsave.mat'];
regPath = fullfile(settingsSet.outpath,filename);

%% If NN has already been optimized for this pod, can skip the optimization, which can be really slow
if exist(regPath,'file')==2
    %Load the previous analysis
    load(regPath);
else
    %Initialize a structure to hold the parameters for the fitted kernel
    nnstruct = zeros(2,size(Y,2));
    
    for i=1:size(Y,2)
        %% Get current y column
        y_t = table2array(Y(:,i))';
        
        nLayer1 = optimizableVariable('nLayer1',[1,min(size(x,1),100)],'Type','integer');%
        nLayer2 = optimizableVariable('nLayer2',[0,min(size(x,1),100)],'Type','integer');
        hyperparametersRF = [nLayer1; nLayer2];
        
        %Warning: if this fails, try removing the "UseParallel" parameter
        rng(1)
        results = bayesopt(@(params)NNLossFnc(params, y_t, x),hyperparametersRF,...
            'AcquisitionFunctionName','expected-improvement-plus',...
            'Verbose',0,'UseParallel', true,'NumSeedPoints',10,'MaxObjectiveEvaluations',50);
        
        %Get the best observed parameters
        bestHyperparameters = results.XAtMinObjective;
        
        nnstruct(1,i) = bestHyperparameters.nLayer1;
        nnstruct(2,i) =  bestHyperparameters.nLayer2;
        
        close all
    end
    %save the fitted hyper parameters for later runs
    save(char(regPath),'nnstruct');
end


%% Grow a neural net for each column of Y using optimized parameters
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

    
    %% Choose Input and Output Pre/Post-Processing Functions
    % For a list of all processing functions type: help nnprocess
    net.input.processFcns = {'removeconstantrows','mapminmax'};
    net.output.processFcns = {'removeconstantrows','mapminmax'};
    
    
    %% Setup Division of Data for Training, Validation, Testing
    %Make training, testing, and validation indices
    setList = repmat([ones(1,30) ones(1,5).*2 ones(1,5).*3],ceil(size(x,2)/40),1);
    setList = setList(1:size(x,2));
    indList = 1:size(x,2);
    trainList = indList(setList==1);
    valList = indList(setList==2);
    testList = indList(setList==3);
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = trainList;
    net.divideParam.valInd = valList;
    net.divideParam.testInd = testList;
    
    %% Choose a Performance Function
    % For a list of all performance functions type: help nnperformance
    net.performFcn = 'mse';  % Mean Squared Error
    
    %% Choose Plot Functions
    % For a list of all plot functions type: help nnplot
    net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
        'plotregression', 'plotfit'};
    %net.trainParam.mu_inc = 1;
    
    %% Train the Network
    rng(999)
    net = train(net,x,y_t);
    
    %% Save the model
    mdls{i} = net;
end
end
%--------------------------------------------------------------------------

function testErr = NNLossFnc(params,y_t,x)
%This is the loss function that is optimized

%% Create a Fitting Network
%Number of hidden layers:
n1 = params.nLayer1;
%Size of those layers
n2 = params.nLayer2;
if n2==0
    hiddenLayerSize = n1;
else
    hiddenLayerSize = [n1 n2];
end
% Bayesian Regularization backpropagation.
trainFcn = 'trainbr';
net = fitnet(hiddenLayerSize,trainFcn);

%% Define training characteristics of the net
net.trainParam.epochs = 5000; %1000 Maximum number of epochs to train
net.trainParam.goal = 0.001; %0 Performance goal
net.trainParam.mu = .5 ; %0.005 Marquardt adjustment parameter
net.trainParam.max_fail  = 10;        %   0  Maximum validation failures
%Choose Input and Output Pre/Post-Processing Functions
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

%Make training, testing, and validation indices.  Divided to be first half, and subsequent quarters
setList = [ones(1,ceil(size(x,2)*0.5)) ones(1,ceil(size(x,2)*0.25))*2 ones(1,ceil(size(x,2)*0.25))*3];
setList = setList(1:size(x,2));
indList = 1:size(x,2);
trainList = indList(setList==1);
valList = indList(setList==2);
testList = indList(setList==3);
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainList;
net.divideParam.valInd = valList;
net.divideParam.testInd = testList;

%Choose a Performance Function
net.performFcn = 'mse';  % Mean Squared Error

rng(999)
[net,tr] = train(net,x,y_t);

%Test model
tInd = tr.testInd;
outputs = net(x(:,tInd));
testErr = perform(net, y_t(:,tInd),outputs);

end



%---------------------------APPLY------------------------------------------
function y_hat = NNPredict(X,mdls,settingsSet)

%Convert from table for functions
x = table2array(X);

%Add some lagged variables if the matrix is relatively small
sensorList = contains(X.Properties.VariableNames,settingsSet.podSensors,'IgnoreCase',true);
if sum(sensorList)<5
    %Decide how many lags to make.  This is arbitrary
    nLags = ceil(minutes(5)/minutes(settingsSet.timeAvg));
    
    %Get Environmental Variables
    envX = x(nLags+1:end,~sensorList);
    
    %Get sensor variables to lag
    toLagX = x(:,sensorList);
    
    %Make Lag matrix
    toLagX = lagmatrix(toLagX,0:nLags);
    
    %Have to remove rows of NaNs
    toLagX = toLagX(nLags+1:end,:);
    
    %Re-combine lagged sensors and environmental parameters
    x = [toLagX, envX];
    
    %Add zeros to the start to avoid size mismatch
    x = [zeros(nLags,size(x,2));x];
end

%Neural nets want columns to be instances for some reason
x = x';

y_hat = zeros(length(mdls),size(X,1));
for i=1:length(mdls)
    net = mdls{i};
    y_hat(i,:) = net(x);
end
y_hat = y_hat';
end
%--------------------------------------------------------------------------

