function [fileList, podList] = getFilesList(analyzeDir)
%{ 
This function is designed to integrate with the unified folder structure to 
load all pod and reference data, and then to return a list of files for
each for further analysis.  It also returns a list of unique pod names

THE SUB FUNCTION "modFileLists" SHOULD BE MODIFIED IF YOU ARE ADDING NEW "POD" TYPES
%}

%% Initialize other variables
%Fix for cross compatibility between OSX and Windows
if ispc == 1;slash = '\';else slash = '/';end

%% Get directory information for colocation and field data
%This is the layout for the unified file structure but could be modified
colocPodDir = [analyzeDir slash 'Colocation' slash 'Pods'];
colocRefDir = [analyzeDir slash 'Colocation' slash 'Reference'];
fieldPodDir = [analyzeDir slash 'Field']; 

%% Get reference data files from subfolder
fileList.colocation.reference.dir = colocRefDir; 
% Get the list of everything in the folder selected
fileList.colocation.reference.files = dir(fullfile(fileList.colocation.reference.dir,'*.csv'));
%Ignore hidden files that are auto generated by OSX (all are exactly 4096 bytes)
fileList.colocation.reference.files = fileList.colocation.reference.files([fileList.colocation.reference.files.bytes]~=4096);
%Convert the list of reference files to a table if it's not empty
if isempty(fileList.colocation.reference.files)
    fileList.colocation.reference.files=table([],[],[],[],[],[],[],[],[],[],'VariableNames',{'name','folder','date','bytes','isdir','datenum','podType','podName','hasHeaders','pollutants'});
else
    fileList.colocation.reference.files = struct2table(fileList.colocation.reference.files);
end

%Assume that all reference files will have headers
fileList.colocation.reference.files.hasHeaders = ones(size(fileList.colocation.reference.files,1),1);
%Make empty cells to hold the list of pollutants found in each reference file
fileList.colocation.reference.files.pollutants = cell(size(fileList.colocation.reference.files,1),1);

%% Get colocated pod data files from subfolder
fileList.colocation.pods.dir = colocPodDir; 
% Get the list of every raw file in the folder selected
fileList.colocation.pods.files = dir(fullfile(fileList.colocation.pods.dir,'*.txt')); 
%Ignore hidden files that are auto generated by OSX
fileList.colocation.pods.files = fileList.colocation.pods.files([fileList.colocation.pods.files.bytes]~=4096);
%Convert the list of colocated pod files to a table if it's not empty
if isempty(fileList.colocation.pods.files)
    fileList.colocation.pods.files=table([],[],[],[],[],[],[],[],[],[],'VariableNames',{'name','folder','date','bytes','isdir','datenum','podType','podName','hasHeaders','delimiter'});
else
    fileList.colocation.pods.files = struct2table(fileList.colocation.pods.files);
end

%% Get uncolocated pod data files from subfolder
fileList.field.pods.dir = fieldPodDir; 
% Get the list of every raw file in the folder selected
fileList.field.pods.files = dir(fullfile(fileList.field.pods.dir,'*.txt')); 
%Ignore hidden files that are auto generated by OSX
fileList.field.pods.files = fileList.field.pods.files([fileList.field.pods.files.bytes]~=4096);
%Convert the list of uncolocated field pod files to a table if it's not empty
if isempty(fileList.field.pods.files)
    fileList.field.pods.files=table([],[],[],[],[],[],[],[],[],[],'VariableNames',{'name','folder','date','bytes','isdir','datenum','podType','podName','hasHeaders','delimiter'});
else
    fileList.field.pods.files = struct2table(fileList.field.pods.files);
end

%% Get the number of files in each folder
ncolocFiles = length(fileList.colocation.pods.files.bytes);
nrefFiles = length(fileList.colocation.reference.files.bytes);
nfieldFiles = length(fileList.field.pods.files.bytes);

%% Initialize additional variables for Pod file information to be entered
[fileList.colocation.pods.files.podType] = cell(ncolocFiles,1);
[fileList.colocation.pods.files.podName] = cell(ncolocFiles,1);
[fileList.colocation.pods.files.hasHeaders] = zeros(ncolocFiles,1);
[fileList.colocation.pods.files.delimiter] = cell(ncolocFiles,1);

[fileList.field.pods.files.podType] = cell(nfieldFiles,1);
[fileList.field.pods.files.podName] = cell(nfieldFiles,1);
[fileList.field.pods.files.hasHeaders] = zeros(nfieldFiles,1);
[fileList.field.pods.files.delimiter] = cell(nfieldFiles,1);

%% Use sub funtion that loops through each list of files and extracts some info about each file (pod name, pod type, headers, etc)
[fileList.colocation.pods.files, colocPodList] = modFileLists(fileList.colocation.pods.files);
[fileList.field.pods.files, fieldPodList] = modFileLists(fileList.field.pods.files);

%Join Pod Lists
tempPodList = [colocPodList fieldPodList];

%Find and return the list of unique pods
podList = unique(tempPodList);

end


function [filestruct, tempPodList] = modFileLists(filestruct)
%% Sub function to loop through the list of files and get some details

%Get the number of files in the folder
nFiles = length(filestruct.bytes);

if (nFiles > 0) %Check that the folder isn't empty
    for i = 1:nFiles
        
        %Get the current filename
        fileName = filestruct.name{i}; 
        
        %If is a U-Pod, file names are different
        if contains(fileName,'YPOD') 
            tempName = char(extractAfter(fileName,'YPOD'));
            filestruct.podType{i} = 'YPOD';
            filestruct.hasHeaders(i) = 0;
            filestruct.delimiter{i} = ',';
        else
            tempName = char(fileName);
            filestruct.podType{i} = 'UPOD';
            filestruct.hasHeaders(i) = 1;
            filestruct.delimiter{i} = ',';
        end
        
        %Write convenient name
        tempName = tempName(1:2);
        filestruct.podName{i} = tempName;

        %Save to a list
        tempPodList{i}=[filestruct.podType{i} filestruct.podName{i}];
        
    end%loop through all files
    
else
    %Need to empty create tempPodList variable to pass back to prevent errors
    tempPodList = cell(0);
end%if statement checking that the folder isn't empty

end%sub function
