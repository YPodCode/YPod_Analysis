function [podList] = readInventory(settingsSet)
%{
This function reads the pod inventory log for this set of data, which
contains information about what variables are stored in which column and
what their data type is.  This information is appended to the pod list,
which previously is just a list of unique pods for analysis
%}

%% Get the file path
%Fix for Windows/Mac slash differences
if ispc == 1; slash = '\'; else; slash = '/'; end

analyzeDir = settingsSet.analyzeDir;

%The inventory file should be stored in a folder titled "Logs"
inventoryDir = [analyzeDir slash 'Logs'];
assert(exist(inventoryDir,'dir')~=0,'Folder for deployment log and inventory does not exist!')

%Create the full file path
filePath = fullfile(inventoryDir,'Pod Inventory.csv');
assert(exist(filePath,'file')~=0,'Pod Inventory does not exist or the name is not correct!')

%% Clean up the options associated with importing the file
opts = detectImportOptions(filePath);
opts.ImportErrorRule = 'omitrow';
opts = setvartype(opts,{'Timezone'},'double');
opts = setvartype(opts,{'PodName','VarNames','VarTypes'},'char');
opts = setvartype(opts,{'dateModified'},'datetime');
opts = setvaropts(opts,{'dateModified'},'InputFormat','M/d/yy');

%Import the inventory file
podInventory = readtable(filePath, opts, 'ReadVariableNames',true);

%Get list of pods for analysis
podList = array2table(settingsSet.podList','VariableNames',{'podName'});
podList.VarNames = cell(size(podList,1),1);
podList.VarTypes = podList.VarNames;
podList.timezone = NaN(size(podList,1),1);

%Read variable information into cell arrays to be able to store headers of different length
for i=1:size(podInventory,1)
    if isnan(podInventory.Timezone(i));continue;end
    %Split the comma separated info and check that there is a type for each variable
    VarNames = split(podInventory.VarNames(i),',')';
    VarTypes = split(podInventory.VarTypes(i),',')';
    assert(size(VarNames,2) == (size(VarTypes,2)),['Error reading variable information from Pod Inventory for pod: ',podInventory.PodName{i}]);
    
    %Save the cell array into the pod inventory table
    podInventory.VarNames{i} = VarNames;
    podInventory.VarTypes{i} = VarTypes;
    
    %Assign these variables into the Pod List that is tracking info for each unique pod
    for j=1:size(podList,1)
        if all(podList.podName{j}==podInventory.PodName{i})
            podList.VarNames{j} = VarNames;
            podList.VarTypes{j} = VarTypes;
            podList.timezone(j) = podInventory.Timezone(i);
        end%if
    end%for pods in pod list
end%for inventory lines

end%function