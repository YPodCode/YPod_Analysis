function [X, t] = makeTElapsed(X, t, settingsSet)
%{
This function calcualtes elapsed time in seconds and adds it as a
predictor variable in X called X.telapsed.  The telapsed added here is
simply the time - the earliest time in t
%}

%Convert to datenum (time in seconds since some reference time
t_temp = datenum(t);
%Find the earliest time in this range
startt = min(t_temp);
%Subtract the start time from t and then add it to X
X.telapsed = t_temp - startt;


end

