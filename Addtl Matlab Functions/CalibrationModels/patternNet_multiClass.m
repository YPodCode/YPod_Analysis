function func = patternNet_multiClass(a)
%% Uses the settings and code from the Classification learner app on a kNN model
switch a
    case 1; func = @patternMultiGen;
    case 2; func = @patternMultiApply;
    case 3; func = @patternMultiReport;
end

end

%%-------------------------------------------------------------------------

%%-------------------------------------------------------------------------
function net = patternMultiGen(Y,X,settingsSet)
%% Make the data match what is expected
nninputmat = table2array(X);
nntargetmat = table2array(Y);
x = nninputmat';
t = nntargetmat';

%Check for existing optimization for this pod and reference
if length(settingsSet.fileList.colocation.reference.files.bytes)==1; reffileName = settingsSet.fileList.colocation.reference.files.name;
else; reffileName = settingsSet.fileList.colocation.reference.files.name{settingsSet.loops.i}; end
currentRef = split(reffileName,'.');
currentRef = currentRef{1};

%This is a workaround for my VOC stuff because models were fit in a few
%ways that was weird
% if settingsSet.isRef
%     clasName = 'ref';
% else
%     clasName = 'est';
% end
clasName = 'raw';
filename = [settingsSet.podList.podName{settingsSet.loops.j} clasName currentRef '_NNCsave.mat'];
regPath = fullfile(settingsSet.outpath,filename);

%% If NNC has already been optimized for this pod, can skip the optimization, which is really slow
if exist(regPath,'file')==2
    %Load the previous analysis
    load(regPath);
else
    nLayer1 = optimizableVariable('nLayer1',[1,min(size(X,2),40)],'Type','integer');%
    nLayer2 = optimizableVariable('nLayer2',[0,min(size(X,2),40)],'Type','integer');
    transFunc = optimizableVariable('transFunc',{'tansig','logsig','purelin','softmax'},'Type','categorical');
    hyperparametersNN = [nLayer1; nLayer2; transFunc];
    
    %Run bayesian optimization of a NN
    rng(1)
    results = bayesopt(@(params)oobErrNN(params,x,t),hyperparametersNN,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'Verbose',1,'UseParallel', false,'NumSeedPoints',10,'MaxObjectiveEvaluations',50);
    close all
    
    %Get the optimized hyperparameters
    nncstruct = results.XAtMinEstimatedObjective;
    
    %save the fitted parameters for later runs
    save(char(regPath),'nncstruct');
end


%% Solve a Pattern Recognition Problem with a Neural Network
% Create a Pattern Recognition Network with a variable number of neurons
% For a list of all training functions type: help nntrain

%Get the nn layer sizes
nL1 = nncstruct.nLayer1;
nL2 = nncstruct.nLayer2;
if nL2 ==0
    hiddenLayerSize = nL1;
else
    hiddenLayerSize = [nL1 nL2];
end

%Make training, testing, and validation indices
setList = repmat([ones(1,30) ones(1,15).*2 ones(1,5).*3],ceil(size(x,2)/50),1);
setList = setList(1:size(x,2));
indList = 1:size(x,2);
trainList = indList(setList==1);
valList = indList(setList==2);
testList = indList(setList==3);

%Make the net
net = patternnet(hiddenLayerSize, 'trainscg');
net.input.processFcns = {'mapstd'};
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainList;
net.divideParam.valInd = valList;
net.divideParam.testInd = testList;
net.performFcn = 'crossentropy';  % Cross-Entropy
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotconfusion', 'plotroc'};
for i = 1:length(net.layers)
    net.layers{i}.transferFcn = char(nncstruct.transFunc);
end

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'crossentropy';  % Cross-Entropy

%Set the activation function
for i = 1:length(net.layers)
    net.layers{i}.transferFcn = char(nncstruct.transFunc);
end

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotconfusion', 'plotroc'};

% Train the Network
rng(123)
net = train(net,x,t);

%Report performance
outputs = net(x);
testErr = perform(net, t, outputs)

close all

end
function testErr = oobErrNN(params, x, t)
%Get the nn layer sizes
nL1 = params.nLayer1;
nL2 = params.nLayer2;
if nL2 ==0
    hiddenLayerSize = nL1;
else
    hiddenLayerSize = [nL1 nL2];
end
   
%Make training, testing, and validation indices
%ntrain = size(C,1) - floor(size(C,1)/3);
setList = repmat([ones(1,30) ones(1,15).*2 ones(1,20).*3],1,ceil(size(x,2)/65));
setList = setList(1:size(x,2));
indList = 1:size(x,2);
trainList = indList(setList==1);
valList = indList(setList==2);
testList = indList(setList==3);

%Make the net
net = patternnet(hiddenLayerSize, 'trainscg');
net.input.processFcns = {'mapstd'};
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainList;
net.divideParam.valInd = valList;
net.divideParam.testInd = testList;
net.performFcn = 'crossentropy';  % Cross-Entropy
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotconfusion', 'plotroc'};
for i = 1:length(net.layers)
    net.layers{i}.transferFcn = char(params.transFunc);
end

% Train the Network
rng(123)
[net, tr] = train(net,x,t);

%Test model
tInd = tr.testInd;
outputs = net(x(:,tInd));
%testErr = perform(net, t(:,tInd),outputs);

%Calculate the F1 measure and define error = 1-F1
% cm = zeros(size(t,1));
% for i = 1:size(cm,1)
%     for j = 1:size(cm,2)
%         targets = t(i,tInd);
%         estimates = outputs(j,:);
%         cm(i,j) = sum(targets==1 & estimates>=0.5);
%     end
% end
F1 = zeros(size(t,1),1);
for i =1:size(t,1)
    tp = sum(t(i,tInd)==1 & outputs(i,:)>=0.5);
    P_hat = sum(outputs(i,:)>=0.5);
    P = sum(t(i,:)==1);
    
    if P==0%Don't count times where no true value was actually present
        F1(i) = NaN;
        continue
    end 
    if P_hat==0
        precision = 0;
    else
        precision = tp/P_hat;
    end
    if tp ==0
        F1(i) = 0;
        continue
    end
    recall = tp/P;
    F1(i) = 2 * precision * recall / (precision + recall);
end
testErr = 1-mean(F1,'omitnan');


end
%%-------------------------------------------------------------------------

%%-------------------------------------------------------------------------
function y_hat = patternMultiApply(X,net,~)

%Transpose the data to match expected
nninputmat = table2array(X);
x = nninputmat';

%Make predictions on new data
y = net(x);
%Transpose to match formatting (rows = instances)
y_hat = y';

end
%%-------------------------------------------------------------------------

%%-------------------------------------------------------------------------
function patternMultiReport(mdlstruct,~)
try
    net = mdlstruct{1};
    view(net)
catch err
    disp('Error reporting the kNN Classification model');
end

end
