function [X,t] = joinPodsData(X, t, settingsSet)
%Loads the data from each pod in the full analysis and joins them together
%into one (large) matrix.  WARNING - DO NOT CORRECT THE ORIGINAL POD's
%TIMEZONE BEFORE THIS FUNCTION, OR THE DATA WILL NOT ALIGN CORRECTLY

%Get the total number of pods being analyzed
if ischar(settingsSet.podList.podName)%Account for only analyzing one Pod (indexing is weird with a list of one)
    nPods=1;
else
    nPods=size(settingsSet.podList.podName,1);
end

%Add the pod name to each variable to keep track of where variables came from
podname = settingsSet.podList.podName{settingsSet.loops.j};
varnames = X.Properties.VariableNames;
for w = 1:length(varnames)
    X.Properties.VariableNames{w} = [varnames{w} podname];
end

%Initialize temporary variable to hold the full joined data
X_temp = X;
X_temp.datetime = t;

%Need to correct for different timezone programming
baseTZ = settingsSet.podList.timezone(settingsSet.loops.j);

%Loop through each pod and load the data
for j = 1:nPods
    %Don't need to re-load the current pod
    if j == settingsSet.loops.j
        continue
    end
    %Load pod data into memory
    podname = settingsSet.podList.podName{j};
    fprintf('--Loading data for %s ...\n', podname);
    X_import = loadPodData(settingsSet.fileList.colocation.pods, settingsSet.podList.podName{j});
    
    %Remove irrelevant columns
    [X_import, xt] = dataExtract(X_import, settingsSet, [settingsSet.podSensors settingsSet.envSensors]);
    
    %Correct for different timezones (moves to match the main pod's timezone)
    tempTZ = settingsSet.podList.timezone(j);
    xt = xt + hours(baseTZ-tempTZ);
    
    %Add pod name to the variables instead of generic matlab changes
    varnames = X_import.Properties.VariableNames;
    for w = 1:size(varnames,2)
        X_import.Properties.VariableNames{w} = [varnames{w} podname];
    end
    
    %Make into one table with datetime as a variable (allows joining)
    X_import.datetime=xt;
    
    %Join with existing data - Note: outerjoin keeps data without an exact
    %match in the datetime column, so there will be many rows with
    %"NaN" in some pods' columns because they are recording at slightly
    %different times.  Align the data before removing actual "NaN"
    %variables caused by data errors
    X_temp = outerjoin(X_temp,X_import,'Keys','datetime','MergeKeys',true);
    
    clear X_import xt
    
end

%Get the appropriate columns to export
t = X_temp.datetime;

%Return the full matrix of X (excluding the datetime column)
keepCols = ~strcmpi(X_temp.Properties.VariableNames,'datetime');
X = X_temp(:,keepCols);

end

