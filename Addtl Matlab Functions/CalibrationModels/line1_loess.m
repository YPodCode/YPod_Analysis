function func = line1_loess(a)

switch a
    case 1; func = @l1loessGen;
    case 2; func = @l1loessApply;
    case 3; func = @l1loessReport;
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function mdl= l1loessGen(Y,X,settingsSet)


mainSensor = settingsSet.podSensors{1}; %Assume that the first sensor is the one to model
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
C.temperature = X.temperature; %Add the temperature column
C.humidity = X.humidity; %Add the humidity column

%Normalize temperature and humidity for fitting (mostly for residual surface fit)
normMat = zeros(2,2);
[C.temperature,tm,tsd]=zscore(C.temperature);
[C.humidity,hm,hsd]=zscore(C.humidity);
normMat(1,1) = tm;
normMat(1,2) = tsd;
normMat(2,1) = hm;
normMat(2,2) = hsd;

%Sensor response as linear function of gas concentration
modelSpec = [pollutant '~' mainSensor];

%Fit the model
linmdl = fitlm(C,modelSpec); 
linmdl = compact(linmdl);

%Get the estimated concentrations
y_hat_lin = predict(linmdl,C(:,mainSensor));

%Now fit a local spline to the residuals as plotted against temperature and humidity
%Calculate the residuals as Y - y_hat
res_lin = table2array(Y(:,1)) - y_hat_lin; 

%Fit a smooth model based on temp and rh
smoothfit = fit([C.temperature, C.humidity],res_lin,'lowess','span',0.05); 

%Export both models
mdl = {linmdl, smoothfit, normMat, mainSensor};
end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = l1loessApply(X,mdl,~)

%Extract the two models from the cell array "mdl"
linmdl = mdl{1};
smoothfit = mdl{2};
normMat = mdl{3};
mainSensor = mdl{4};

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

%Make a temporary data table
C = sensorData;
C.temperature = X.temperature; %Add the temperature column
C.humidity = X.humidity; %Add the humidity column

%Normalize temperature and humidity for fitting
if normMat(1,2)==0
    C.temperature=zeros(size(C,1),1);
else
    C.temperature = (C.temperature-normMat(1,1))/normMat(1,2); %Normalize the temperature column
end
%Normalize temperature and humidity for fitting
if normMat(2,2)==0
    C.humidity=zeros(size(C,1),1);
else
    C.humidity = (C.humidity-normMat(2,1))/normMat(2,2); %Normalize the humidity column
end


%Get the estimated concentrations
y_hat_lin = predict(linmdl,C(:,mainSensor));

%Get the estimated residuals
y_hat_smooth = smoothfit([C.temperature, C.humidity]); 

%Get a final estimate
y_hat = y_hat_lin + y_hat_smooth;

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function l1loessReport(mdl,~)
linmdl = mdl{1};
try
    linmdl.Coefficients.Estimate'
catch err
    disp('Error reporting this model');
end

end