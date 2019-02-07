function [X,t] = RbyR0(X, t, settingsSet)
%{
This function is intended to divide raw sensor values by their "clean air"
or base values.  It looks in the directory selected under the "Logs" folder
for an CSV file titled "ResistanceValues.csv".  This file should be two
columns. The first should contain sensor identifications that match those
in the "Pod Inventory.csv" file.  The second should include a number that
all sensor values should be divided by.  An example line is below:
"Fig2600_1234,500"
%}

%Fix for cross compatibility between OSX and Windows
if ispc == 1; slash = '\'; else; slash = '/'; end

%Check that the file exists
logPath = fullfile(settingsSet.analyzeDir, ['Logs' slash 'ResistanceValues.csv']);
if exist(logPath,'file') ~= 2
    logPath = fullfile(settingsSet.analyzeDir, ['Logs' slash 'ResistanceValues.CSV']);
end
assert(exist(logPath,'file')==2,'Resistance file not found! Check that it is in the "Logs" subfolder and is named "ResistanceValues.csv"');

%Import the list of resistances
opts = detectImportOptions(logPath);
opts.ImportErrorRule = 'omitrow';
R0Log = readtable(logPath, opts, 'ReadVariableNames',false);
R0Log.Properties.VariableNames = {'Sensors' 'Resistance'};

variableNames = X.Properties.VariableNames;

%Loop through each sensor and try to find the R0 value to use
for j = 1:length(variableNames)
    
    %Set R0 to zero to make sure it is found (or throw an error if it isn't)
    R0 = 0;
    currentVar = variableNames{j};
    
    %If an exact match is found for the sensor name, get that resistance value
    if any(strcmp(R0Log.Sensors, currentVar))
        %This will break if it finds more than one exact value
        R0 = R0Log.Resistance(any(strcmp(R0Log.Sensors,currentVar)));
        
        
        %Otherwise try to find a generic match (eg. the sensor is named Fig2600_1, and we want to use a generic R0 for all "Fig2600_*"s)
    else 
        %Make sure that this is actually a pod sensor (not environmental data, etc)
        isSensor = 0;
        for i = 1:length(settingsSet.podSensors)
            if strfind(currentVar,settingsSet.podSensors{i})>0
                isSensor = 1;
            end
        end
        if isSensor==1
            for i = 1:length(R0Log.Sensors)%Use a more generic check - NOTE: this will find any subset match, so if you have Fig2600_15 and the log has Fig2600_1, it will use that value!
                if any(regexpi(currentVar,R0Log.Sensors{i}))
                    R0 = R0Log.Resistance(i);
                end
            end
            
            
            %If this isn't a pod sensor (temperature, humidity, etc) divide by 1
        else
            R0 = 1;
        end
    end
    
    assert(R0~=0,['No applicable resistance found for ' variableNames{j}]);
    %Divide by the selected R0
    X{:,j}=X{:,j}/R0;
end

end