function [X, t] = sortbyTime(X, t, ~)
%{
This function uses t to sort X monotonically
%}

%First sort the timeseries and get the index used for sorting X
[t,index]=sort(t);

%Sort X
X = X(index,:);

end

