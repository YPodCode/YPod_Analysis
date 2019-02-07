function [refData] = loadRefData(settingsSet)

%Get the file path to load
i = settingsSet.loops.i;
if ischar(settingsSet.fileList.colocation.reference.files.name)
    reffileName = settingsSet.fileList.colocation.reference.files.name;
else
    reffileName = settingsSet.fileList.colocation.reference.files.name{i};
end
%Get the full path to the folder
direct = settingsSet.fileList.colocation.reference.dir;
%Append that path to the specific reference file
filePath = char(fullfile(direct,reffileName));

%See if an equivalent mat file exists
matFilePath = strrep(filePath,'.txt','.mat');
matFilePath = strrep(matFilePath,'.TXT','.mat');
matFilePath = strrep(matFilePath,'.csv','.mat');
matFilePath = strrep(matFilePath,'.CSV','.mat');

%Verify that there is a .mat file and throw error if it doesn't exist
assert(exist(matFilePath, 'file') == 2,['Reference .mat data file for [' reffileName '] not found!']);

%Load the data
alldata = load(matFilePath);
loadData = whos('-file',matFilePath);
refData = alldata.(loadData.name);
clear alldata1

try  %For a dataset
    headers = refData.Properties.VariableNames;
catch err  %For a table
    headers = refData(1,:);
end
    
end