function [Y,X,t] = deployLogMatch(Y,X,t,settingsSet,isField)
%Verify that data in Y and X match time ranges in the deployment log

%% Get information about current pod and reference file
i = settingsSet.loops.i;
j = settingsSet.loops.j;

%Get the pod name
podName = settingsSet.podList.podName{j};

%Get the reference file name
%If this is field data, the reference file is 'NA'
if isField
    refFileName='NA';
    %Otherwise get the reference file name
elseif ischar(settingsSet.fileList.colocation.reference.files.name)
    refFileName = settingsSet.fileList.colocation.reference.files.name;
else
    refFileName = settingsSet.fileList.colocation.reference.files.name{i};
end

%Clear index variables
clear i j

%% Get the deployment log into its own variable for code tidyness
deployLog = settingsSet.deployLog;

%Initialize vector for indicating whether to keep data
keep = false(size(X,1),1);

%% Check each entry in the deployment log
for i = 1:size(deployLog,1)
    %Join the pod name and type for convenience
    deployPod = deployLog.PodName{i};
    
    %If the pod name and reference file name both match...
    if strcmp(deployPod,podName) && strcmp(deployLog.ReferenceFile{i}, refFileName)

        %Get T/F list of time entries within the time period indicated on this line
        withinTime = isbetween(t,deployLog.Start(i)-hours(1),deployLog.End(i)+hours(1));
        
        %Make those lines "true" to keep
        keep(withinTime) = true;
        
    elseif strcmp(deployPod,podName) && isField
        %The reference file name is flexible if you're analyzing field data (it can be 'N/A' or anything else)
        
        %Get T/F list of time entries within the time period indicated on this line
        withinTime = isbetween(t,deployLog.Start(i)-hours(1),deployLog.End(i)+hours(1));
        
        %Make those lines "true" to keep
        keep(withinTime) = true;
    end
end

%% Return only the values from when the pod was colocated with the specified reference file
t = t(keep);
Y = Y(keep,:);
X = X(keep,:);

% %Throw an error for data with no matches
% assert(foundEntry > 0, ['No entries in the deployment log found for ' refFileName ' and ' podName])
% 
% %Throw error if there is no overlap between data and entries in deployment log
% assert(~isempty(t),['No overlap between data and entries in deployment log for ' refFileName ' and ' podName]);

end

