function [X,t] = addTimeofDay(X, t, ~)
%Gets the time of day from t - NOTE: make sure that "t" is in the
%local time of day when this function is called!

%Get the time of day in hours
tod = hour(t)+minute(t)/60+second(t)/3600;

%Make a proxy for time of day by converting hour to cosine wave (max at noon, minimum at midnight
tod = cos((tod-12)*pi/12)+1;

%Append to X
X.ToD = tod;

end