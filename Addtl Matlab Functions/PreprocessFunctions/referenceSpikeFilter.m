function [X,t] = referenceSpikeFilter(X, t, ~)
%Remove values where the point to point deviation is greater than 2 std
%deviations as calculated within a rolling 60 minute window

%Convert to array
xnames = X.Properties.VariableNames;
tempX = table2array(X);

%Window length can be changed here
l_w = datenum(hours(1));

%Convert to datenums for much faster math
tempt = datenum(t);

%Number of std deviations to allow point to point variation within
std_max = 2;

%Loop through each variable
threshMat = zeros(size(tempX));
for j = 1:size(tempX,2)
    %Go through all values
    for i = 1:size(tempX,1)
        %Get a boolean list of points within the window
        tnow = tempt(i);
        window = tempt>=(tnow-l_w) & tempt<=(tnow+l_w);
        
        %Extract just that data from X
        X_window = tempX(window,j);
        
        %Calculate the threshold based on the std deviation within the window
        threshMat(i,j) = std_max * nanstd(X_window);
    end
end
clear i j

%Calcualte forward and backward absolute differences
forw_diffs = zeros(size(tempX));
back_diffs = zeros(size(tempX));
forw_diffs(1:end-1,:) = abs(tempX(1:end-1,:) - tempX(2:end,:));
back_diffs(2:end,:) = abs(tempX(2:end,:) - tempX(1:end-1,:));

%Determine which points vary outside of the acceptable bounds
badPtArray = forw_diffs>threshMat | back_diffs>threshMat;

%Make those values NaN
tempX(badPtArray)=NaN;
X = array2table(tempX,'VariableNames',xnames);

end

