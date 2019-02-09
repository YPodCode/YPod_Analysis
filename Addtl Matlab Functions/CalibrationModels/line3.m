function func = line3(a)
%Fit the sensor response as a function of pollutant, temperature, and humidity
%Note: This function assumes that your data has columns named
%"temperature" and "humidity".  It will break if they are not
%in your pod data.

switch a
    case 1; func = @line3Gen;
    case 2; func = @line3Apply;
    case 3; func = @line3Report;
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function mdlstruct = line3Gen(Y,X,settingsSet)

columnNames = X.Properties.VariableNames;
foundCol = 0;
mainSensor = settingsSet.podSensors{1}; %Assume that the first sensor is the one to model

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
C=[Y(:,1),sensorData]; %Join into a temporary table
C.temperature = X.temperature; %Add the temperature column
C.humidity = X.humidity; %Add the humidity column

%Sensor response as function of gas concentration
modelSpec = [mainSensor '~' Y.Properties.VariableNames{1} '+ temperature + humidity'];

%%Fit the model
mdl = fitlm(C,modelSpec);  

%Fit the model
mdl = fitlm(C,modelSpec);
mdl = compact(mdl);

mdlstruct = {mdl, mainSensor};

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = line3Apply(X,mdlstruct,settingsSet)

%Get the column name and model
mdl = mdlstruct{1};
mainSensor = mdlstruct{2}; 

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
C=sensorData; %Join into a temporary table
C.temperature = X.temperature; %Add the temperature column
C.humidity = X.humidity; %Add the humidity column

coeffs = mdl.Coefficients.Estimate'; %Get the estimates of coefficients

mdlinv = @(p,sens,temp,hum) ((sens-p(1)-p(3).*temp-p(4).*hum)./p(2)); %Invert the model (concentration~Figaro+Temperature+Humidity)

y_hat = mdlinv(coeffs,C.(mainSensor),C.temperature,C.humidity); %Get the estimated concentrations

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function line3Report(fittedMdl,~)
try
    fittedMdl.Coefficients.Estimate'
catch err
    disp('Error reporting this model');
end

end
%--------------------------------------------------------------------------
