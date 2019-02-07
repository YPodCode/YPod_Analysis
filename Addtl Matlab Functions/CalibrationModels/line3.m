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
for i = 1:length(columnNames)
    if any(regexpi(columnNames{i},mainSensor))
        foundCol=foundCol+1; %Keep track of how many columns matching this sensor were found
        sensorData = X(:,i); %Extract that data into its own table
        mainSensor = columnNames{i}; %Get the real name of the sensor
    end
end
assert(foundCol == 1,['Could not find a unique column for sensor: ' mainSensor]);

C=[Y,sensorData]; %Join into a temporary table
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

C=X(:,mainSensor); %Join into a temporary table
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
