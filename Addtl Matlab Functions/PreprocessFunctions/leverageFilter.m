function [X,t] = leverageFilter(X, t, settingsSet)
%{
DOES NOT WORK CURRENTLY
Remove high leverage points from the reference data stream
Uses 1.5 times the number of predictor variables, rather than the standard 2
%}

hh = leverage([x',v']);
x(hh>1.25*predictors./number_points) = NaN;
leverage_removed = sum(hh>1.25*predictors./number_points);

end