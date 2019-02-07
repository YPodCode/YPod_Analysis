function valList = prepostVal(~, ~, t, n)
%Select groups of data from out to in (group 1 gets the first and last x points and group n is the middle)

%First sort the timeseries
[t,index]=sort(t);

%Initialize list
k = size(t,1);
valList = ones(k,1);

%Find the mipoint
midpoint = floor(k/2);

%Calculate groups for each point (using a line from 0:n at the midpoint and then descending)
indices = 1:midpoint;
secindices = midpoint:k;
valList(1:midpoint) = (2*n)/k * indices;
valList(midpoint:end) = 2*n - (2*n)/k * secindices;
valList = ceil(valList);

%Catch zeros and over numbers caused by rounding odd numbers
valList(valList>n)=n;
valList(valList<=0)=1;

%Assign using the original index incase t was not sorted
valList = valList(index);
end