function [X,t] = removeDST(X, t, ~)
%{
This function removes an hour from timestamps that were recorded with an
instrument that observes  daylight savings time (DST) in order to allow 
matching with instruments that do not observe DST
%}

%Check that the toolbox with function nweekdate is installed
assert(~isempty(which('nweekdate')),'Function nweekdate does not exist! Make sure that the Financial Toolbox is installed');

%Check that dates are datetime variables
assert(isdatetime(t),'Time vector is not in the datetime format!');

%Get which years the deployment occurred during
yearsanalyzed = unique(t.Year);
%Loop through those years and correct for DST
for i = 1:length(yearsanalyzed)
    %Get start date of DST (2am on second Sunday in March)
    startdate = nweekdate(2, 1, yearsanalyzed(i), 3,[],'datetime')+hours(2);
    %Get end date of DST (2am on first Sunday in November)
    enddate = nweekdate(1, 1, yearsanalyzed(i), 11,[],'datetime')+hours(2);
    %Subtract 1 hour during daylight savings time
    t(t>=startdate & t<enddate) = t(t>=startdate & t<enddate) - hours(1);
end

end