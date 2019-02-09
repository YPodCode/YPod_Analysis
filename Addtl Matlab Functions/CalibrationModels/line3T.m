function func = line3T(a)
%Fit the sensor response as a function of pollutant, temperature, humidity, and include a pollutant*temperature interaction term
%Note: This function assumes that your data has columns named
%"temperature" and "humidity".  It will break if they are not
%in your pod data.

switch a
    case 1; func = @line3TGen;
    case 2; func = @line3TApply;
    case 3; func = @line3TReport;
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function mdlstruct = line3TGen(Y,X,settingsSet)

%Assume that the first sensor is the one to model
columnNames = X.Properties.VariableNames;
mainSensor = settingsSet.podSensors{1}; 
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
C=[Y(:,1),sensorData]; 
%Add the temperature column
C.temperature = X.temperature;
%Add the humidity column
C.humidity = X.humidity; 

%Sensor response as function of pollutant concentration
modelSpec = [mainSensor '~' pollutant ' + temperature + humidity + temperature:' pollutant];
%fitted:  mainSensor = 'p(1) + pollutant.*p(2) + p(3)*temperature + p(4)*humidity + pollutant.*temperature+p(5)' 
%inverted:  pollutant = '(mainSensor - p(1) - p(3).*temperature - p(4).*humidity )/(p(5)*temperature+p(2))' 

%Fit the model
mdl = fitlm(C,modelSpec);
mdl = compact(mdl);

mdlstruct = {mdl, mainSensor};

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = line3TApply(X,mdlstruct,~)

%Saved model parameters
mdl = mdlstruct{1};
mainSensor=mdlstruct{2};

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
C=sensorData;
C.temperature = X.temperature; %Add the temperature column
C.humidity = X.humidity; %Add the humidity column

%Get the previously fitted estimates of coefficients
coeffs = mdl.Coefficients.Estimate';

%Invert the model (concentration~Figaro+Temperature+Humidity)
mdlinv = @(p,sens,temp,hum) ((sens - p(1) - p(3).*temp - p(4).*hum)./(p(5).*temp + p(2))); 

%Get the estimated concentrations
y_hat = mdlinv(coeffs,C.(mainSensor),C.temperature,C.humidity);

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function line3TReport(mdl,~)
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
