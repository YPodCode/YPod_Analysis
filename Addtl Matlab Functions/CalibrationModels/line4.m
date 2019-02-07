function func = line4(a)
%Note: This function assumes that your data has columns named
%"temperature", "humidity", and "telapsed".  It will break if they are not
%in your pod data.

switch a
    case 1; func = @line4Gen;
    case 2; func = @line4Apply;
    case 3; func = @line4Report;
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function mdlstruct = line4Gen(Y,X,settingsSet)

columnNames = X.Properties.VariableNames;
foundCol = 0;
%Assume that the first sensor is the one to model as the primary sensor
mainSensor = settingsSet.podSensors{1}; 
%First column of Y is fitted pollutant
pollutant = Y.Properties.VariableNames{1}; 

%Find the column containing the sensor for analysis
for i = 1:length(columnNames)
    if any(regexpi(columnNames{i},mainSensor))
        foundCol=foundCol+1; %Keep track of how many columns matching this sensor were found
        sensorData = X(:,i); %Extract that data into its own table
        mainSensor = columnNames{i}; %Get the real name of the sensor column
    end
end
assert(foundCol == 1,['Could not find a unique column for sensor: ' mainSensor]);

C=[Y,sensorData]; %Join into a temporary table
C.temperature = X.temperature; %Add the temperature column
C.humidity = X.humidity; %Add the humidity column
C.telapsed = X.telapsed; %Add the elapsed time

%Sensor response as function of gas concentration
modelSpec = [mainSensor '~' pollutant '+ temperature + humidity + telapsed'];
%Fitted:   mainSensor = 'p(1) + p(2)*pollutant + p(3)*temp + p(4)*hum +p(5)*telaps'
%Inverted: pollutant = '(mainSensor-p(1) - p(3).*temp - p(4).*hum - p(5).*telaps)/(p(2))'

%Fit the model
mdl = fitlm(C,modelSpec);  
mdl = compact(mdl);

mdlstruct = {mdl, mainSensor};

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = line4Apply(X,mdlstruct,settingsSet)

%Get the column name and model
mdl = mdlstruct{1};
mainSensor = mdlstruct{2}; 


%Collect the predictor variables
C=X(:,mainSensor); %Main sensor data
C.temperature = X.temperature; %Add the temperature column
C.humidity = X.humidity; %Add the humidity column
C.telapsed = X.telapsed; %Add the elapsed time

%Get the previously fitted coefficients
coeffs = mdl.Coefficients.Estimate';

%The inverted model is below:
mdlinv = @(p,sens,temp,hum,telaps) ((sens-p(1)-p(3).*temp-p(4).*hum-p(5).*telaps)./p(2));

%Predict new concentrations
y_hat = mdlinv(coeffs,C.(mainSensor),C.temperature,C.humidity,C.telapsed);

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function line4Report(mdl,~)
try
    figure;
    subplot(2,2,1);plotResiduals(mdl);
    subplot(2,2,2);plotDiagnostics(mdl,'cookd');
    subplot(2,2,3);plotResiduals(mdl,'probability');
    subplot(2,2,4);plotResiduals(mdl,'lagged');
    plotSlice(mdl);
catch err
    disp('Error reporting this model');
end

end
%--------------------------------------------------------------------------
