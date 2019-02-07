function func = line1(a)

switch a
    case 1; func = @line1Gen;
    case 2; func = @line1Apply;
    case 3; func = @line1Report;

end

end

%--------------------------------------------------------------------------
function mdlobj = line1Gen(Y,X,settingsSet)

columnNames = X.Properties.VariableNames;
foundCol = 0;

%Assume that the first sensor is the one to model
mainSensor = settingsSet.podSensors{1}; 

%Assume that the first column of Y is the modeled pollutant
pollutant = Y.Properties.VariableNames{1};

%Find the column containing the sensor for analysis
for i = 1:length(columnNames)
    if any(regexpi(columnNames{i},mainSensor))
        foundCol=foundCol+1; %Keep track of how many columns matching this sensor were found
        mainSensor = columnNames{i}; %Get the real name of the sensor
    end
end
assert(foundCol == 1,'Could not find a unique sensor column');

C=[Y,X(:,mainSensor)]; %Join into a temporary table

% %Sensor response as function of gas concentration
% modelSpec = [mainSensor '~' pollutant]; 
% 
% %Fit the model
% mdl = fitlm(C,modelSpec); 
% 
% %Get the estimates of coefficients
% coeffs = mdl.Coefficients.Estimate'; 
% 
% %Invert the model (concentration~Figaro)
% mdlinv = @(p,sens) ((sens-p(1))./p(2)); 
% 
% %Get the estimated concentrations
% y_hat = mdlinv(coeffs,C.(mainSensor)); 

%Fit the model
modelSpec = [pollutant '~' mainSensor]; 
mdl = fitlm(C,modelSpec); 
mdl = compact(mdl); %Compact the model to reduce size

mdlobj = {mdl, mainSensor};
end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = line1Apply(X,mdlobj,settingsSet)

%Get fitted model components
mdl = mdlobj{1};
mainSensor = mdlobj{2};

% %Get the estimates of coefficients
% coeffs = mdl.Coefficients.Estimate'; 
% 
% %Invert the model (concentration~sensor)
% mdlinv = @(p,sens) ((sens-p(1))./p(2)); 
% 
% %Get the estimated concentrations
% y_hat = mdlinv(coeffs,X.(mainSensor));

%Predict
y_hat = predict(mdl,X.(mainSensor));

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function line1Report(fittedMdl,~)
try
    fittedMdl.Coefficients.Estimate'
catch err
    disp('Error reporting this model');
end

end
%--------------------------------------------------------------------------