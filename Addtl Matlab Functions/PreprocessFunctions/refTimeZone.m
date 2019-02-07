function [X,t] = refTimeZone(X, t, settingsSet)
%{
This function uses the pod inventory file to adjust for differences between
the timezone in which the pod is recording time and the reference is
reported in.  Note that this does not account for DST issues
%}

%% Get on current pod and reference file
%Get the name of the current reference file
refFile = settingsSet.fileList.colocation.reference.files.name;
if ~ischar(refFile)
    refFile = refFile(settingsSet.loops.i);
end
%Current pod being analyzed
currentPod = settingsSet.podList.podName{settingsSet.loops.j};

%% Get the associated timezone (this will just take the first match from the deployment log)
foundMatch = false;
for i = 1:size(settingsSet.deployLog)
    logPodName = [settingsSet.deployLog.Type{i} settingsSet.deployLog.Name{i}];
    logRefFile = settingsSet.deployLog.ReferenceFile{i};
    if strcmpi(logRefFile, refFile) && strcmpi(logPodName, currentPod)
        refTZ = hours(settingsSet.deployLog.TimeZoneDeployed(i));
        foundMatch = true;
        break
    end
end

%Check that nothing weird happened
assert(foundMatch,['Error finding a match for pod ' currentPod ' and file ' refFile ' in the deployment log while correcting timezone.'])

%Get the timezone that this pod was programmed in
podTZ = hours(settingsSet.podList.timezone(settingsSet.loops.j));

%Adjust the timeseries of the pod to match the reference file
t = t + (podTZ - refTZ);

end%function