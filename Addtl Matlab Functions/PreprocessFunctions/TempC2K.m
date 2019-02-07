function [X,t] = TempC2K(X, t, settingsSet)
%{
Convert temperatures in X from Celcius to Kelvin
%}

foundT = false;
if ~ismember('temperature', X.Properties.VariableNames)
    for ii=1:size(X,2)
        %Find temperature column.  Uses 1st column containing string "temperature"
        currentCol = X.Properties.VariableNames{ii};
        if any(regexpi(currentCol,'temperature'))
            X.Properties.VariableNames{ii} = 'temperature';
            foundT = true;
            break
        end
    end
else
    foundT = true;
end

if foundT
    %Convert temperature to K
    X.temperature = X.temperature+273.15;
else
    %If there are no temperatures, add in a dummy variable to let the code run
    warning('NO TEMPERATURE COLUMN LOCATED, DUMMY COLUMN ADDED WITH VALUES OF -1');
    X.temperature = ones(size(X,1),1).*-1;
end

end