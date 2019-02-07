function convertRefDatatoMat(settingsSet)
%{
Loop through all reference files, check if a mat file exists, and create
one if there isn't one already 
%}

if isempty(settingsSet.fileList.colocation.reference.files)
    return;
end

%Extract the structure holding file information (for code readability)
filesStruct = settingsSet.fileList.colocation.reference;

for i = 1:length(filesStruct.files.bytes)
    
    %Indexing is weird if only one file, so must handle separately
    if length(filesStruct.files.bytes)==1
        fileName = {filesStruct.files.name};
    else
        fileName = filesStruct.files.name(i);
    end
    
    %Get the file directory and then assemble with the name to get the path
    fileDir = filesStruct.dir;
    filePath = fullfile(fileDir,fileName);
    
    %See if an equivalent mat file exists
    matFilePath = strrep(filePath,'.txt','.mat');
    matFilePath = strrep(matFilePath,'.TXT','.mat');
    matFilePath = strrep(matFilePath,'.csv','.mat'); 
    matFilePath = strrep(matFilePath,'.CSV','.mat');
    
    %Convert to character array for functionality
    matFilePath = char(matFilePath);
    filePath = char(filePath);
    
    %% Convert the file if it has not already been converted (the .mat file doesn't exist)
    if exist(matFilePath, 'file') ~= 2
        fprintf('No mat file, reading plain text file for %s ...\n',char(fileName));
        
        %Read in the data
        opts = detectImportOptions(filePath,'Delimiter',',');
        opts = setvartype(opts,opts.VariableNames,[{'char'} repmat({'double'},1,size(opts.VariableNames,2)-1)]);
        opts.VariableNames{1} = 'datetime';
        rawData = readtable(filePath, opts);
        
        %Save the file and then clear the variable for next loop
        save(matFilePath, 'rawData');
        clear rawData;
    end% if statement that checks whether to make a new mat file
end% for look that goes through each file

end% function