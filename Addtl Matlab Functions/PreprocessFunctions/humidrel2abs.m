function [X,t] = humidrel2abs(X, t, ~)
%Convert relative humidity to partial pressure of water.  Requires "temperature" and "humidity" columns.  Can also accept "pressure" column

%Find humidity column.  Uses 1st column containing string "humidity"
foundH = false;
if ~ismember('humidity', X.Properties.VariableNames)
    for ii=1:size(X,2)
        currentCol = X.Properties.VariableNames{ii};
        if any(regexpi(currentCol,'humidity'))
            X.Properties.VariableNames{ii} = 'humidity';
            foundH = true;
            break
        end
    end
else
    foundH = true;
end

%Start by checking that temperature and humidity columns are in X
if ismember('temperature', X.Properties.VariableNames) && foundH
    
    %convert Humidity in X from relative to absolute
    temp = X.temperature;
    rh = X.humidity;
    
    %make sure the temperature and humidity that go in the function are reasonable and otherwise remove them
    keep = (temp<=370)&(temp>=263)&(rh<=100)&(rh>=0);
    if any(keep)
        X(~keep,:)=[];
        t(~keep==1,:)=[];
        
        %Check whether to use pressure values from pod
        if ismember('pressure', X.Properties.VariableNames)
            %Remove unreasonable pressure values
            keep = X.pressure>0;
            if any(keep)
                X(~keep,:)=[];
                t(~keep==1,:)=[];
                %Calculates partial pressure of water
                absHumtemp = convert_humidity(X.pressure,X.temperature,X.humidity, 'relative humidity','partial pressure','Murphy&Koop2005');
            end
        else
            %Calculates partial pressure of water.  Assumes average sea level atmospheric pressure.
            absHumtemp = convert_humidity(101320.75,X.temperature,X.humidity, 'relative humidity','partial pressure','Murphy&Koop2005');
        end%if checking if pressure is in X
        
        %Overwrite the humidity column
        X.humidity = absHumtemp;
    else
        warning('No allowable temperatures and/or humidities, RH conversion skipped!');
    end%if checking values of T & RH
    
else
    warning('Temperature and/or Humidty columns for RH conversion do not exist! Humidity not converted!');
end%if checking temperature and humidity exist in X


end%function

