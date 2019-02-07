function func = smart4T(a)
%{
Uses our knowlege of what sensors are good for various gases to select
appropriate sensors and then fit a model using them and their interactions
with temperature.

This does not seem to work well (so far), but I'm not sure quite why yet.
It may be overfitting the data, but that's just an educated guess
%}
switch a
    case 1; func = @smart4TFit;
    case 2; func = @smart4TPredict;
    case 3; func = @linsensReport;
end

end

%--------------------------------------------------------------------------
function mdlobj = smart4TFit(Y,X,settingsSet)

%Get a list of column names
columnNames = X.Properties.VariableNames;

%These lists hold the gases that the sensors are useful for
gas_2600 = {'CH4','NatGas','NMHC'};
gas_2602 = {'CH4','Gasoline','NatGas','NMHC'};
gas_2611 = {'O3','CONO'};
gas_5121 = {'CO1','CONO'};
gas_2710 = {'NO1','NO2','CONO'};
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
        elseif any(regexpi(currentCol,'2710'))
            gaslist = gas_2710;
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
    C = X(:,keepList(:,zz));
    C.temperature = X.temperature;
    C.humidity = X.humidity;
    C.telapsed = X.telapsed;
    cnames = C.Properties.VariableNames;
    %C = table2array(C);
    %Y_fit = table2array(Y(:,pollutant));
    C = [C Y(:,pollutant)];
    
    
    %Define the model
    %Typical 4T: pollutant = '(v-p(1) - p(3).*temperature - p(4).*humidity - p(5).*telapsed)/(p(2) + p(6)*temperature)'
    tempmdlspec = ['@(b,x) ((b(1)'];
    for i = 1:length(sensors)
        tempmdlspec = [tempmdlspec ' + b(' num2str(i+1) ')*x(:,' num2str(i) ')'];
    end
    
    tempmdlspec = [tempmdlspec ' + b(' num2str(i+2) ')*x(:,' num2str(i+1) ')'...
        ' + b(' num2str(i+3) ')*x(:,' num2str(i+2) ')' ...
        ' + b(' num2str(i+4) ')*x(:,' num2str(i+3) ')' ...
        ')./(b(' num2str(i+5) ') + b(' num2str(i+6) ')*x(:,' num2str(i+1) ')))'];
    tempmdlspec = str2func(tempmdlspec);
    
    %Initialize the coefficients to small random numbers
    b = randn(1,i+6);
    
    %Fit the linear model
    testMdl  =fitnlm(C,tempmdlspec,b);
    
    %Define the model again, omitting useless variables
    %Typical 4T: pollutant = '(v-p(1) - p(3).*temperature - p(4).*humidity - p(5).*telapsed)/(p(2) + p(6)*temperature)'
    mdlspec = ['@(b,x) ((b(1)']; nb = 1;
    for i = 1:length(sensors)
        if testMdl.Coefficients.pValue(i+1)<0.05
            nb = nb+1;
            mdlspec = [mdlspec ' + b(' num2str(nb) ')*x(:,' num2str(i) ')'];
        end
    end
    mdlspec = [mdlspec ' + b(' num2str(nb+1) ')*x(:,' num2str(i+1) ')'...
        ' + b(' num2str(nb+2) ')*x(:,' num2str(i+2) ')' ...
        ' + b(' num2str(nb+3) ')*x(:,' num2str(i+3) ')' ...
        ')./(b(' num2str(nb+4) ') + b(' num2str(nb+5) ')*x(:,' num2str(i+1) ')))'];
    mdlspec = str2func(mdlspec);
    
    %Initialize the coefficients to small random numbers
    b = randn(1,nb+5);
    
    %Fit the linear model
    mdl{zz}  =fitnlm(C,mdlspec,b);
    
    
end

%% Make a structure to hold both the model and the list of columns to keep
mdlobj = {mdl, keepList};

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = smart4TPredict(X,mdlobj,~)

%Extract fitted objects
mdl = mdlobj{1};
keepList = mdlobj{2};

for zz = 1:length(mdl)
    %Extract data from X
    C = X(:,keepList(:,zz));
    C.temperature = X.temperature;
    C.humidity = X.humidity;
    C.telapsed = X.telapsed;
    
    %Predict
    y_hat = predict(mdl{zz},C);
end
end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function linsensReport(mdlobj,~)
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