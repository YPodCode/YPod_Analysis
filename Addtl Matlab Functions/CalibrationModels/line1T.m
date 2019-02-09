function func = line1T(a)

switch a
    case 1; func = @line1TGen;
    case 2; func = @line1TApply;
    case 3; func = @line1TReport;

end

end

%--------------------------------------------------------------------------
function mdl  = line1TGen(Y,X,settingsSet)
%Assume that the first sensor is the one to model
mainSensor = settingsSet.podSensors{1};
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
%Assume that the first sensor is the one to model
mainSensor = settingsSet.podSensors{1}; 

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