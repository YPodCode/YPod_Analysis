function [X,t] = addACF(X, t, ~)
%Add a rolling autocorrelation measure spanning the past 5 minutes

%Minutes to lag by
t_lag = minutes(5);

%Start by finding where to begin (at least 5 minutes in)
index = 1:length(t);
shift = min(t)+t_lag;
index(t>shift)=[];
index = max(index);

%Make empty array to input values
acfArray = zeros(length(t),size(X,2));

%Loop beginning at that point
for i=(index+1):length(t)
    %Get two shifted timeseries
    t_current = t(i);
    t_earlier = t_current - t_lag;
    X_roll = table2array(X((t<t_current & t>t_earlier),:));
    for j = 1:size(X,2)
        try
            [normalizedACF, lags] = autocorr(X_roll(:,j));
            acfArray(i,j) = normalizedACF(2);
        catch
            
        end
    end
end

%Assign nice variable names
acfArray = array2table(acfArray);
for i = 1:size(X,2)
    acfArray.Properties.VariableNames{i} = [X.Properties.VariableNames{i} '_ACF'];
end

%Join the two arrays
X = [X acfArray];

end