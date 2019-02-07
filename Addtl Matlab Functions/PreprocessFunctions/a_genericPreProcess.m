function [X,t] = a_genericPreProcess(X, t, settingsSet)
%This is the general format for a preprocessing function.  These may filter
%data, add variables, or otherwise modify "X", which can be either the
%matrix of predictor variables OR the vector of reference values

%An example that would remove rows with datetimes in the future
X(t>datetime('now'),:)=NaN;

end