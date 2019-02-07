function [X,t] = plotWavelets(X, t, settingsSet)

try
    %Check that the wavelet analysis package is installed
    prod_inf = ver;
    if ~any(strcmp(cellstr(char(prod_inf.Name)),'Wavelet Toolbox'))
        error('Wavelet Toolbox Not Installed!');
    end
    
    
    for i = 1:size(X,2)
        variableName = X.Properties.VariableNames{i};
        %Analyze sensor signals
        for j = 1:length(settingsSet.podSensors)
            sensorName = settingsSet.podSensors{j};
            if any(regexpi(variableName,sensorName))
                %Get that column as an array
                tempX = table2array(X(:,i));
                %Calculate sampling period
                t_sample = mean(diff(t));
                %Plot continuous wavelet transform
                figure('Position',[100 100 800 400]);
                %Plot continuous wave power spectrum w/o NaN values 
                %(this may cause distorted charts depending on how many NaNs your data has)
                cwt(tempX(~isnan(tempX)),t_sample);
                title(['Wavelet Power Spectrum for ' sensorName]);
            end
        end
        %Also analyze reference signals
        for j = 1:length(settingsSet.refGas)
            gasName = settingsSet.refGas{j};
            if any(regexpi(variableName,gasName))
                %Get that column as an array
                tempX = table2array(X(:,i));
                %Calculate sampling period
                t_sample = mean(diff(t));
                %Plot continuous wavelet transform
                figure('Position',[100 100 800 400]);
                cwt(tempX(~isnan(tempX)),t_sample);
                title(['Wavelet Power Spectrum for ' gasName]);
            end
        end
    end
        
catch err
    warning('Wavelet Toolbox Not Installed!  Wavelet analysis skipped!');
end

end

