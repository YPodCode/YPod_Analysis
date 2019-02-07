function [X,t] = easyRbyR0(X, t, settingsSet)
%{
Attempt to convert voltage measurements to relative resistance
measurements.  This assumes that the voltage divider is similar to that in
the Y-Pod for the two figaro sensors (but this will only scale values if
those assumptions are not true)
%}

variableNames = X.Properties.VariableNames;

inverseSensors = {'2611'}; %Sensors where the clean air value is the minimum instead of the maximum
resistanceSensors = {'2611','2600','2602','5121'};%Resistive (MOx) sensors that a voltage divider is used for

%Loop through each sensor and try to find the R0 value to use
for j = 1:length(variableNames)
    
    %Set R0 to zero to make sure it is found (or throw an error if it isn't)
    R0 = 0;
    invFlag = 1;
    currentVar = variableNames{j};
    
    %Only do this for resistive sensors (not electrochemical, PID, temperature, etc sensors)
    isSensor = 0;
    for i = 1:length(resistanceSensors)
        if any(regexpi(currentVar,resistanceSensors{i}))
            isSensor = 1;
            break
        end
    end
    if isSensor==0; continue;end
    
    %Get the vector of voltage measurements in millivolts
    tempVolt = table2array(X(:,j)) * (0.188);
    %Assume that this is in a voltage divider with 5V input and a 2kOhm load resistor
    tempR = (5000*2000)./(tempVolt) - 2000;
    
    %Check to see if this sensor has an inverse response - NOTE: this will find any subset match, so if you have Fig2600_15 and the log has Fig2600_1, it will use that value!
    for i = 1:length(inverseSensors)
        if any(regexpi(currentVar,inverseSensors{i}))
            invFlag = -1;
            break
        end
    end
    
    %Find times that we're at the right temperature
    tgood = abs(X.temperature - 298) < 1;
    R0 = max(tempR(tgood).*invFlag);
    
    %If there were no good temperature ranges, allow to continue but don't
    %correct
    if sum(tgood)==0
        R0=1;
        warning(['Sensor: ' currentVar ' was not divided by a base resistance (no good temperature found).  This may be caused by temperatures not being in Celsius']);
    end
    
    %Put R/R0 into the old variable
    X.(currentVar) = tempR./R0;
    
end

end