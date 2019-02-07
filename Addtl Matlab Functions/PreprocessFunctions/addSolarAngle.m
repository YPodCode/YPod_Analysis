function [X,t] = addSolarAngle(X, t, settingsSet)
%Add the solar angle as a predictive variable

%CHANGE THIS DEPENDING ON LOCATION
%Longitude and latitude of deployment (want to eventually load this from deployment log)
%LONG = -105.14; LAT = 39.93; %Boulder, CO
LONG = -118.28; LAT = 34.03; %Los Angeles, CA

%Get the day of the year
n = year(t); %First get the year that it is (in case this stradles the new year)
n = datetime(n,1,1); %Then set it to the first day of that year
n = t - n; %Subtract to find the time since new year
n = days(n); %Finally convert that to full days

%Solar Declination
d = 23.45*sind(360*(284+n)/365);

%EOT Factor
B = (360/365)*(d-81);

%Equation of Time
EOT = 9.87*sind(2*B)-7.53*cosd(B)-1.5*sind(B);

%Find the local timezone as entered in the deployment log
podName = settingsSet.podList.podName{settingsSet.loops.j};
deployLog = settingsSet.deployLog;
deltaT_gmt = ones(size(X,1),1);
matchlog = false;
for i = 1:size(deployLog,1)
    %Join the pod name and type from the deploy log for convenience
    deployPod = deployLog.PodName{i};
    
    %If the pod name matches, try to use this entry
    if strcmp(deployPod,podName) 
        matchlog = true;
    end
    
    %If a match was found in the deployment log
    if matchlog
        %Get the timezone from this entry
        refTZ = deployLog.TimeZoneDeployed(i);
        
        %Get T/F list of time entries within the time period indicated on this line
        withinTime = isbetween(t,deployLog.Start(i),deployLog.End(i));
        
        %Assign those values to the list of timezones
        deltaT_gmt(withinTime) = refTZ;
        
        %Reset match flag
        matchlog = false;
    end
end
clear deployLog withinTime refTZ matchlog

%Local solar time meridian
%deltaT_gmt = settingsSet.refTZ;
LSTM = 15*deltaT_gmt;

%Time correction factor in minutes
TC = 4.*(LONG - LSTM) + EOT;

%Local solar time
LST = hours(timeofday(t)) + TC/60;

%Local solar hour
LSH = (12-LST); 

%Solar Hour Angle
h_s = 15*LSH;

%Solar Altitude
alpha = asind(sind(LAT).*sind(d)+cosd(LAT).*cosd(d).*cosd(h_s)); %Altitude in Radians
%alpha = asin(sind(LAT).*sind(d)+cosd(LAT).*cosd(d).*cosd(h_s)); %Altitude in degrees
%alpha(alpha<0) = 0; %If you want to ignore night

%Append to X
X.alpha = alpha;

end