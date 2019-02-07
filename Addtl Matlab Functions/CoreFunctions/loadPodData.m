function [podData] = loadPodData(fileStruct, podName)

%% Loop through list of pod files and upload ones that match the pod name
%First, get the number of files in each
nPodFiles = length(fileStruct.files.bytes);
%Get the directory (folder) where colocated pod data are stored
fileDir = fileStruct.dir;

%Loop through each file in the pod file directory and check if it contains data for this pod
for i = 1:nPodFiles
    
    %If there's only one file, indexing is weird
    if nPodFiles == 1
        filePodName = fileStruct.files.podName;
        filePodType = fileStruct.files.podType;
        fileName = fileStruct.files.name;
    else
        filePodName = fileStruct.files.podName{i};
        filePodType = fileStruct.files.podType{i};
        fileName = fileStruct.files.name{i};
    end
    
    %Create full file path for reading file
    filePath = fullfile(fileDir,fileName);
    
    %Load the file if the name matches
    if strcmpi([filePodType,filePodName],podName)
        
        %Get the equivalent mat file name
        matFilePath = strrep(filePath,'.txt','.mat');
        matFilePath = strrep(matFilePath,'.TXT','.mat');
        
        %Load the file into memory
        fprintf('Loading file %s ...\n',matFilePath);
        temp = load(matFilePath);
        temp = temp.rawData;
        
        %If first loaded file, create the podData variable and assign "temp" to it
        if exist('podData','var')==0
            podData = temp;
        else
            podData = [podData;temp];
        end
    end
end

%Throw an error if no pod data was loaded (the variable temp will not exist)
if exist('temp','var')==0
    warning(['No .mat files found for ' podName '.  This may be caused by pods not in both the "field" and "colocation" folder. Analysis for this pod will be skipped.']);
    podData = table(0);
end
end


