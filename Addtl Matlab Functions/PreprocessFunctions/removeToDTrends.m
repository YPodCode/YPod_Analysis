function [X,t] = removeToDTrends(X, t, ~)
%{
This function fits a smooth curve to each column of X versus time of day 
and then subtracts that fit to get values with diurnal trends removed 
%}

%Get the variable names for cleaner code
varNames = X.Properties.VariableNames;

%Get matrices to do work on
tempX = table2array(X);
tempt = datenum(timeofday(t));

%Loop through each column and each row in X
for i = 1:size(X,2)
    
    %Sort the timeseries
    [B, I] = sort(tempt);
    tempMat = tempX(I,i);
    %Calculate median # of points in each hour block
    medhrs = zeros(24,1);
    for j=0:23;medhrs(j+1)=sum(hour(t)==(j));end
    k = round(mean(medhrs),0);
    
    %Make times loop around so the curve is smooth at midnight
    temp1 = tempMat(1:k,i);
    temp2 = tempMat(end-(k-1):end,i);
    tempMat = [temp2; tempMat; temp1];
    temp1 = tempt(I(1:k),i)+1;
    temp2 = tempt(I((end-k+1):end),i)-1;
    tplot = [temp2; tempt(I); temp1];
%     %% Manual moving median
%     %Initialize M
%     m = tempt;
%     hr=1/24;
%     tlist = tempt>=0&tempt<=hr;
%     m(tlist) = median(tempX(tlist,i),'omitnan');
%     for j = 1:23
%         hr = j/24;
%         nexthr = (j+1)/24;
%         tlist = tempt>hr&tempt<=nexthr;
%         m(tlist) = median(tempX(tlist,i),'omitnan');
%     end
%     %% Moving median automated
%     m = movmedian(tempMat,k,'omitnan');
%     m = m((k+1):end-k);
    %% Fit a smooth model based on time of day
    s = smooth(tplot,tempMat,0.2,'rlowess');

    %Make into a fitted object to make it easier to subtract
    f = fit(tplot,s,'smoothingspline');
    smoothX = f(tempt);
    
    %Subtract the smoothed values
    tempX(:,i) = tempX(:,i)-smoothX;    
    
    
end

end