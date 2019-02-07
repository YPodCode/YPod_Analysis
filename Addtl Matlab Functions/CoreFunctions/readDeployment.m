function deployLog = readDeployment(analyzeDir)
%This function reads the deployment log for this set of data 

%Fix for Windows/Mac slash differences
if ispc == 1; slash = '\'; else; slash = '/'; end

%The deployment log should be stored in a folder titled "Logs"
logDir = [analyzeDir slash 'Logs'];
assert(exist(logDir,'dir')~=0,'Folder "Logs" for deployment log and inventory does not exist!')

%Create the full file path
filePath = fullfile(logDir,'Deployment Log.csv');
assert(exist(filePath,'file')~=0,'Deployment log does not exist!')

%Clean up the options associated with importing the file
opts = detectImportOptions(filePath);
opts.ImportErrorRule = 'omitrow';
opts = setvaropts(opts,{'Start','End'},'InputFormat','MM/dd/yy HH:mm');

%Import the deployment file
deployLog = readtable(filePath, opts, 'ReadVariableNames',true);

end