function [X, t] = makeDiscontTElaps(X, t, settingsSet)
%{
This function calcualtes elapsed time in seconds and adds it as a
predictor variable in X called X.telapsed.  The telapsed added here does
NOT include gaps in time longer than "t_gap"
%}

%Minimum gap in the timeseries to ignore as a "gap"
t_gap = datenum(minutes(settingsSet.timeAvg))*5;

%Need to convert to datenum to get the time in seconds
tempt = datenum(t);

%First sort the timeseries and get the index used for putting back into X
[tempt,index]=sort(tempt);

%Make an array to totalize time
tsegs = zeros(size(t,1),1);

%Run through the time array and totalize time, ignoring gaps longer than t_gap
for i = 2:size(tempt,1)
    %Calculate the time difference between this timestamp and the next
    delta = tempt(i)-tempt(i-1);
    %Only add to the rolling time if the delta is less than the threshold for a "gap"
    if delta < t_gap
        tsegs(i) = tsegs(i-1)+delta;
    else
        tsegs(i) = tsegs(i-1);
    end
end

%Add to X
tsegs(index) = tsegs;
X.telapsed = tsegs;


end

