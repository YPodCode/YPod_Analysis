function [X,t] = rmWarmup(X, t, ~)
% This function removes 30 minutes after gaps longer that 5* the typical
% interval between measurements

%% Length of time to remove after gaps:
trem = minutes(60);

%% Find missing values
%Get two shifted timeseries
time1 = t(1:end-1);
time2 = t(2:end);
%Get delta time between one step to the next
deltat = time2 - time1;
clear time1 time2
%Get typical time step
typdelta = median(deltat,'omitnan');

%% Loop through t backwards to remove gaps
%Make list of rows to remove
remList = false(length(t),1);
if any(deltat>(typdelta*5))
    for i = (size(t,1)-1) : -1 : 1
        dt = t(i+1)-t(i);
        if dt>typdelta
            remList((t>=t(i+1)) & (t<=(t(i+1)+trem)))=true;
        end
    end
end

%% Remove those rows
t(remList,:)=[];
X(remList,:)=[];
        
end

