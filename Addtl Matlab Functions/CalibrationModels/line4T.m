function func = line4T(a)
%Note: This function assumes that your data has columns named
%"temperature", "humidity", and "telapsed".  It will break if they are not
%in your pod data.

switch a
    case 1; func = @line4TGen;
    case 2; func = @line4TApply;
    case 3; func = @line4TReport;
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function mdlobj = line4TGen(Y,X,settingsSet)


mainSensor = settingsSet.podSensors{1}; %Assume that the first sensor is the one to model as the primary sensor
%First column of Y is fitted pollutant
pollutant = Y.Properties.VariableNames{1}; 

%Find the column containing the sensor for analysis
columnNames = X.Properties.VariableNames;
mainSensorIndex = contains(columnNames,mainSensor,'IgnoreCase',true);
mainSensor = columnNames{mainSensorIndex};
sensorData = X(:,mainSensorIndex);

if sum(mainSensorIndex) > 1
    warning(['Did not find a unique column for sensor: ' mainSensor]);
    %Scale multiple sensors and then average them to keep model invertable
    sensorData = zscore(table2array(sensorData));
    sensorData = mean(sensorData,2);
    sensorData = array2table(sensorData,'VariableNames',{mainSensor});
else
    assert(sum(mainSensorIndex) == 1,['Did not find a column for sensor: ' mainSensor])
end

%Join into a temporary table
C=[Y(:,pollutant),sensorData];  %Join into a temporary table
C.temperature = X.temperature; %Add the temperature column
C.humidity = X.humidity; %Add the humidity column
C.telapsed = X.telapsed; %Add the elapsed time

%Sensor response as function of gas concentration
modelSpec = [mainSensor '~' pollutant '+ temperature + humidity + telapsed + temperature:' pollutant];
%Fitted:   mainSensor = 'p(1) + pollutant.*p(2) + p(3)*temperature + p(4)*humidity +p(5)*telapsed + pollutant.*p(6)*T'
%Inverted: pollutant = '(v-p(1) - p(3).*temperature - p(4).*humidity - p(5).*telapsed)/(p(2) + p(6)*temperature)'

%Fit the model
mdl = fitlm(C,modelSpec);  
mdl = compact(mdl);

mdlobj = {mdl, mainSensor};
end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = line4TApply(X,mdlobj,~)

%Get fitted model components
mdl = mdlobj{1};
mainSensor = mdlobj{2};

%Find the column containing the sensor for analysis
columnNames = X.Properties.VariableNames;
mainSensorIndex = contains(columnNames,mainSensor,'IgnoreCase',true);
mainSensor = columnNames{mainSensorIndex};
sensorData = X(:,mainSensorIndex);

if sum(mainSensorIndex) > 1
    warning(['Did not find a unique column for sensor: ' mainSensor]);
    %Scale multiple sensors and then average them to keep model invertable
    sensorData = zscore(table2array(sensorData));
    sensorData = mean(sensorData,2);
    sensorData = array2table(sensorData,'VariableNames',{mainSensor});
else
    assert(sum(mainSensorIndex) == 1,['Did not find a column for sensor: ' mainSensor])
end

%Collect the predictor variables
C=sensorData; %Main sensor data
C.temperature = X.temperature; %Add the temperature column
C.humidity = X.humidity; %Add the humidity column
C.telapsed = X.telapsed; %Add the elapsed time

%Get the previously fitted coefficients
coeffs = mdl.Coefficients.Estimate';

%The inverted model is below:
mdlinv = @(p,sens,temp,hum,telaps) ((sens - p(1) - p(3).*temp - p(4).*hum - p(5).*telaps)./(p(2) + p(6).*temp));

%Predict new concentrations
y_hat = mdlinv(coeffs,C.(mainSensor),C.temperature,C.humidity,C.telapsed);

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function line4TReport(mdlobj,~)
mdl = mdlobj{1};
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
