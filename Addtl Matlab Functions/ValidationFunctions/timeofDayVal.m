function valList = timeofDayVal(~, ~, t, n)
%Select portions of data by time, starting at the earliest timestamp

%Get the time of day in hours
tod = hour(t)+minute(t)/60+second(t)/3600;

%Make a proxy for time of day by converting hour to cosine wave (max at noon, minimum at midnight
%tod = (cos((tod-12)*pi/12)+1)/2;
tod = abs(tod-12)/12;
%tod = tod*n/23+1;

%valList = round(tod*(n-1))+1;
quants=quantile(tod,n-1);
%quants = prctile(tod,[1:n].*10);
valList = ones(size(t,1),1);

valList(tod<quants(1))=1;
for i = 2:n-1
    valList(tod>=quants(i-1) & tod<quants(i))=i;
end
valList(tod>=quants(n-1))=n;

% figure
% subplot(1,3,1)
% plot(t, valList,'ro')
% subplot(1,3,2)
% plot(hour(t),valList,'bo')
% subplot(1,3,3)
% plot(t,tod)
% 
% for j = 1:n; disp([num2str(j) ',' num2str(sum(valList==j))]);end

end