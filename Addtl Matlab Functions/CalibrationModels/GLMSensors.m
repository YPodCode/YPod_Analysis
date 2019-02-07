function func = GLMSensors(a)

switch a
    case 1; func = @glmsensGen;
    case 2; func = @glmsensApply;
    case 3; func = @glmsensReport;

end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function mdlobj = glmsensGen(Y,X,settingsSet)

%Get a list of column names
columnNames = X.Properties.VariableNames;
warning('off','stats:glmfit:BadScaling')

%These lists hold the gases that the sensors are useful for
gas_2600 = {'CH4','NatGas','NMHC'};
gas_2602 = {'CH4','Gasoline','NatGas','NMHC'};
gas_2611 = {'O3','CONO'};
gas_5121 = {'CO1','CONO'};
gas_mocon = {'Gasoline','NMHC','NatGas'};
gas_CO2 = {'CO2','CH4'};
gas_NOB4 = {'NO1','NO2','CONO'};
gas_COB4 = {'CO1','CONO'};

%%Loop through each column of Y
mdl = cell(size(Y,2),1);
keepList = false(length(columnNames),size(Y,2));
for zz = 1:size(Y,2)
    %Assume that the first column of Y is the modeled pollutant
    pollutant = Y.Properties.VariableNames{zz};
    
    %% Loop through each column to ID if it is useful for this pollutant
    for i = 1:length(columnNames)
        %Get the current column name
        currentCol = columnNames{i};
        gaslist = false;
        
        %Check what sensor this is
        if any(regexpi(currentCol,'Fig2600'))
            gaslist = gas_2600;
        elseif any(regexpi(currentCol,'Fig2602'))
            gaslist = gas_2602;
        elseif any(regexpi(currentCol,'2611'))
            gaslist = gas_2611;
        elseif any(regexpi(currentCol,'5121'))
            gaslist = gas_5121;
        elseif any(regexpi(currentCol,'NDIR'))
            gaslist = gas_CO2;
        elseif any(regexpi(currentCol,'mocon'))
            gaslist = gas_mocon;
        elseif any(regexpi(currentCol,'NO_B4'))
            gaslist = gas_NOB4;
        elseif any(regexpi(currentCol,'CO_B4'))
            gaslist = gas_COB4;
        end
        
        %Loop through the list of pollutants for this sensor and keep the column if
        %it matches the current pollutant
        if isa(gaslist,'cell')
            for j = 1:length(gaslist)
                if any(regexpi(pollutant,gaslist{j}))
                    %Keep this column if there's a match
                    keepList(i,zz) = true;
                    break
                end
            end
        else
            %Don't keep this column if it's not useful for this pollutant
            keepList(i,zz) = false;
        end
    end
    
    %% Fit the model
    sensors = columnNames(keepList(:,zz));
    C = [Y(:,pollutant) X(:,keepList(:,zz))];
    C.temperature = X.temperature;
    C.humidity = X.humidity;
    
    %Define the model
    mdlspec = [pollutant '~ humidity '];
    for i = 1:length(sensors)
        mdlspec = [mdlspec '+ temperature*' sensors{i}];
    end
    
    %Fit the linear model
    mdl{zz}  =fitglm(C,mdlspec,'Distribution','gamma');
    
end
%% Make a structure to hold both the model and the list of columns to keep
mdlobj = {mdl, keepList};
warning('on','stats:glmfit:BadScaling')

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = glmsensApply(X,mdlobj,~)

%Extract fitted objects
mdl = mdlobj{1};
keepList = mdlobj{2};
y_hat = zeros(size(X,1),length(mdl));

for zz = 1:length(mdl)
    %Extract data from X
    C = X(:,keepList(:,zz));
    C.temperature = X.temperature;
    C.humidity = X.humidity;
    
    %Predict
    y_hat(:,zz) = predict(mdl{zz},C);
end
end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function glmsensReport(mdlobj,~)
try
    mdl = mdlobj{1};
    for zz = 1:length(mdl)
        mdl{zz}
    end
catch err
    disp('Error reporting this model');
end

end
%--------------------------------------------------------------------------