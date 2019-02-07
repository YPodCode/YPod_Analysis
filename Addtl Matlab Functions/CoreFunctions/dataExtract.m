function [X, t] = dataExtract(rawData, settingsSet, colsToKeep)
%This function goes through the raw data and extracts the specified columns
%listed in "colsToKeep", plus a vector of the datetime values.  It will 
%throw an error if it's not clear which column is the "correct" datetime.
%NOTE: If any of the requested column names contains "allcol", every column
%not labeled as date/time will be passed along (e.g. for PCA analysis).
%This may include columns that are all NaN or could otherwise cause issues
%without proper processing later on

%Get list of column names from the loaded file
columnNames = rawData.Properties.VariableNames;

%Initialize selection vectors
keep = false(1,length(columnNames));
foundDateTime = 0;
dateOnly = 0;
timeOnly = 0;
foundCols = zeros(1,length(colsToKeep));

%Loop through reference file columns and selected gases to extract only important columns
for i =1:length(columnNames)
    
    %Get current column name (for code clarity)
    currentCol = columnNames{i};
    
    %% ----Check if column contains date and time info for alignment----
    if strcmpi(currentCol,'datetime')
        
        %Change variable name for easier referencing
        rawData.Properties.VariableNames{i}='datetime';
        
        %If it's already a datetime, make sure it's not imported wrong
        if isdatetime(rawData.datetime)
            flag = median(rawData.datetime,'omitnan');
            assert(flag > datetime(1990,1,1) && flag < datetime(2100,1,1),'Datetime was autoimported and had OOB dates...');
        end
        
        if(~isdatetime(rawData.datetime))
        %% If not in datetime format, try converting
            %Try to convert the input datetime into datetime using default values and then values defined in the settingsSet structure
            try
                %Make temporary vector to try to convert datetime values
                tempdatetime=datetime(rawData.datetime);
                
                %Don't let it import weird dates
                flag = median(tempdatetime,'omitnan');
                assert(flag > datetime(1990,1,1) && flag < datetime(2100,1,1),'Issues reading datetime...');
                
                %If it succeeded, assign the dates into the table and clear temporary variable
                rawData.datetime=tempdatetime;
                clear tempdatetime
            catch err
                warning('Nonstandard date string detected');
                for j=1:length(settingsSet.datetimestrs)
                    try
                        %Don't make it keep trying if it's already converted
                        assert(~isdatetime(rawData.datetime),'Datetime converted succefully');
                        
                        %Same process as before
                        tempdatetime=datetime(rawData.datetime,'InputFormat',settingsSet.datetimestrs{j});
                        flag = median(tempdatetime,'omitnan');
                        
                        %Don't let it import weird dates
                        assert(flag > datetime(1990,1,1) && flag < datetime(2100,1,1),'Issues reading datetime...');
                        
                        %If it succeeded, write to table
                        rawData.datetime=tempdatetime;
                        clear tempdatetime
                    catch err
                    end
                end%Loop of nonstandard datetime strings to try
            end%Try/catch for datetime assignment
            
            %If the conversion didn't work, throw an error
            assert(isdatetime(rawData.datetime),'Datetime column not imported correctly');
        end%Datetime reading
        
        %Increment count of datetime columns found
        foundDateTime=foundDateTime+1;
        
        
    elseif strcmpi(currentCol,'UNIX') || strcmpi(currentCol,'POSIX') 
        %% ----Try Unix/posix timestamps----
    
        %Change variable name for easier referencing
        rawData.Properties.VariableNames{i}='datetime';
        
        %Convert from epoch time
        rawData.datetime = datetime(rawData.datetime,'ConvertFrom','posixtime');
        
        %Increment the foundDateTime - this allows it to throw an error if multiple datetime columns are found
        foundDateTime=foundDateTime+1;
        
        
    elseif strcmpi(currentCol,'date') 
        %% ----Allow for the use of separate time and date columns----
        
        rawData.Properties.VariableNames{i}='date';
        
        %Try to convert the date into datetime using default values and then values defined in the settingsSet structure
        try 
            rawData.date=datetime(rawData.date);
            dateOnly=i;
        catch err
            disp('Nonstandard date string detected, applying specified parse strings');
            for j=1:length(settingsSet.datestrings)
                try
                    rawData.date=datetime(rawData.date,'InputFormat',settingsSet.datestrings{j});
                    dateOnly=i;
                catch err
                end
            end
        end
        
    elseif strcmpi(currentCol,'time')
        rawData.Properties.VariableNames{i}='time';
        try %Try to convert the time into datetime using default values and then values defined in the settingsSet structure
            rawData.time=datetime(rawData.time);
            timeOnly=i;
        catch err
            disp('Nonstandard time stamp detected, applying specified parse strings');
            for j=1:length(settingsSet.timestrings)
                try
                    rawData.time=datetime(rawData.time,'InputFormat',settingsSet.timestrings{j});
                    timeOnly=i;
                catch err
                end
            end
        end
    else
        
        
        %% ----Check if column contains another variable marked to keep----
        %Check if this column was one of those requested
        for ii = 1:length(colsToKeep)
            if any(regexpi(currentCol,colsToKeep{ii}))
                keep(i)=true;
                foundCols(ii)=foundCols(ii)+1;
                %rawData.Properties.VariableNames{i}=colsToKeep{ii};
            elseif any(contains(colsToKeep,'allcol')) && isnumeric(rawData.(columnNames{i}))
                %Keep this data if the user selected to keep all (non-time) columns
                keep(i) = true;
            end
        end
        
    end
end

%If no datetime or unix time variables were found, try to combine date and time variables
if (foundDateTime == 0) && (dateOnly > 0) && (timeOnly >0)
    rawData.datetime = rawData.date + timeofday(rawData.time);
    foundDateTime = 1;
end

%Check that everything was found okay
assert(foundDateTime==1,'Could not identify a singular date column when extracting data!  Only one column may be labeled "datetime" or contain "posix" or "unix"');

%Warn if a data column was not located (don't throw an error)
for i=1:length(foundCols)
    if foundCols(i)~=1
        warning(['Identified ' num2str(foundCols(i)) ' columns for: ' colsToKeep{i}]);
    end
end


%Return only columns that are in the keep list
X=rawData(:,keep);

%Also return vector of datetimes for timeseries analysis
t = rawData.datetime;
end
