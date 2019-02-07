function func = line1T(a)

switch a
    case 1; func = @line1TGen;
    case 2; func = @line1TApply;
    case 3; func = @line1TReport;

end

end

%--------------------------------------------------------------------------
function mdl  = line1TGen(Y,X,settingsSet)

columnNames = X.Properties.VariableNames;
foundCol = 0;
mainSensor = settingsSet.podSensors{1}; %Assume that the first sensor is the one to model
pollutant = Y.Properties.VariableNames{1};

%Find the column containing the sensor for analysis
for i = 1:length(columnNames)
    if any(regexpi(columnNames{i},mainSensor))
        foundCol=foundCol+1; %Keep track of how many columns matching this sensor were found
        mainSensor = columnNames{i}; %Get the real name of the sensor
    end
end
assert(foundCol == 1,'Could not find a unique sensor column');

%Join into a temporary table
C=[Y,X(:,mainSensor)]; 
C.telapsed = X.telapsed;

%Sensor response as function of gas concentration and time elapsed
modelSpec = [mainSensor '~' pollutant ' + telapsed'];
%Fitted:   mainSensor = 'p(1) + pollutant.*p(2) + p(3)*telapsed
%Inverted: pollutant = (mainSensor - p(1) - p(3).*telapsed)./(p(2))

%Fit the model
mdl = fitlm(C,modelSpec);  
mdl = compact(mdl);

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = line1TApply(X,mdl,settingsSet)
columnNames = X.Properties.VariableNames;
foundCol = 0;
mainSensor = settingsSet.podSensors{1}; %Assume that the first sensor is the one to model

%Find the column containing the sensor for analysis
for i = 1:length(columnNames)
    if any(regexpi(columnNames{i},mainSensor))
        foundCol=foundCol+1; %Keep track of how many columns matching this sensor were found
        mainSensor = columnNames{i}; %Get the real name of the sensor
    end
end
assert(foundCol == 1,'Could not find a unique sensor column');

%Get the data together
C = X(:,mainSensor);
C.telapsed = X.telapsed;

%Get the fitted estimates of coefficients
coeffs = mdl.Coefficients.Estimate'; 

%Invert the model (concentration~sensor+time)
mdlinv = @(p,sens,telaps) ((sens - p(1) - p(3).*telaps)./(p(2)));

%Get the estimated concentrations
y_hat = mdlinv(coeffs,C.(mainSensor),C.telapsed); 

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function line1TReport(mdl,~)
try
    figure;
    subplot(2,2,1);plotResiduals(mdl);
    subplot(2,2,2);plotDiagnostics(mdl,'cookd');
    subplot(2,2,3);plotResiduals(mdl,'probability');
    subplot(2,2,4);plotResiduals(mdl,'lagged');
    plotSlice(mdl);
catch
    disp('Error reporting this model');
end

end
%--------------------------------------------------------------------------