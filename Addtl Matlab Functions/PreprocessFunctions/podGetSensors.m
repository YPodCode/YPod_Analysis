function [X,t] = podGetSensors(X, t, settingsSet)
%Removes columns that I don't want cluttering the analysis

%Check if any values in each row are exactly -999
keeplist = false(size(X,2),1);

%Column names to keep
electrochem = {'NO2_B1','CO_B4','H2S_BH','O3_B4','NO_B4'};
figaros = {'fi2600','fig2602','fig4161','fig2611'};
e2vs = {'e2v2611','MICS5121','MICS2611','MICS5525'};
othersens = {'co2_NDIR','bl_mocon'};
env = {'temperature','humidity'};

%Join those lists
allsens = [electrochem figaros e2vs othersens env];
for i = 1:size(X,2)
    currentcol = X.Properties.VariableNames{i};
    for j = 1:length(allsens)
        currentsens = allsens{j};
        if any(regexpi(currentcol,currentsens))
            keeplist(i)=true;
        end
    end
end

%% Get just those columns
X = X(:,keeplist);

end

