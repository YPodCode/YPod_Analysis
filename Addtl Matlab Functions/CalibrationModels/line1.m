function func = line1(a)

switch a
    case 1; func = @line1Gen;
    case 2; func = @line1Apply;
    case 3; func = @line1Report;

end

end

%--------------------------------------------------------------------------
function mdlobj = line1Gen(Y,X,settingsSet)

%Assume that the first sensor is the one to model
mainSensor = settingsSet.podSensors{1}; 

%Assume that the first column of Y is the modeled pollutant
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
C=[Y(:,1),sensorData];  %Join into a temporary table

%Fit the model
modelSpec = [pollutant '~' mainSensor]; 
mdl = fitlm(C,modelSpec); 
mdl = compact(mdl); %Compact the model to reduce size

mdlobj = {mdl, mainSensor};
end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = line1Apply(X,mdlobj,~)

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

%Predict
y_hat = predict(mdl,sensorData);

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