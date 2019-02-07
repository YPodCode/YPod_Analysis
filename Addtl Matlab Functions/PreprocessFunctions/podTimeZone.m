function [X,xt] = podTimeZone(X, xt, settingsSet)
%This function uses the pod inventory file to adjust for differences between
%the timezone in which the pod is recording time and the reference is
%reported in.  Note that this does not account for DST issues


%% Get information about current pod file
j = settingsSet.loops.j;

%Get the pod name
podName = settingsSet.podList.podName{j};
podTZ = hours(settingsSet.podList.timezone(j));

%Clear index variables
clear j

%% Get the deployment log into its own variable for code tidyness
deployLog = settingsSet.deployLog;
%List of flags to keep track of which times have been corrected
corrected = false(size(xt,1),1);
%Flag for matching entries in deployment log
matchlog = false;
%Temporary array to hold corrected times
temp_t = xt;

%% Check each entry in the deployment log
%NOTE: If there are overlapping entries in the deployment log, the first entry will be used for the correction
for i = 1:size(deployLog,1)
    %Join the pod name and type from the deploy log for convenience
    deployPod = deployLog.PodName{i};
    
    %If the pod name matches, try to use this entry
    if strcmp(deployPod,podName) 
        matchlog = true;
    end
    
    %If a match was found in the deployment log
    if matchlog
        %Get the timezone from this entry
        refTZ = hours(deployLog.TimeZoneDeployed(i));
        
        %Make temporary shifted timeseries to compare to log
        t_shift = xt + (refTZ - podTZ);
        
        %Get T/F list of time entries within the time period indicated on this line
        %Window starts 1 hr before deploy log and ends 1 hour after in case of weirdness with daylight savings
        withinTime = isbetween(t_shift,deployLog.Start(i)-hours(1),deployLog.End(i)+hours(1));
        
        withinTime = withinTime & ~corrected;
        
        %Assign shifted times into the temporary time array
        temp_t(withinTime) = t_shift(withinTime);
        
        %Keep track of which ones have been corrected in case of overlapping entries
        corrected(withinTime) = true;
        
        %Reset match flag
        matchlog = false;
    end
    
    %Can skip looping through rest of deployment log if every point has been corrected
    if all(corrected)
        break
    end
end

%% Have to throw away times that were not corrected to avoid creating overlapping times
xt = temp_t(corrected);
X = X(corrected,:);

end%function