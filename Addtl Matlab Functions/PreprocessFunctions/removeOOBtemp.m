function [X,t] = removeOOBtemp(X, t, settingsSet)
%Remove unrealistic temperature values (over 400K (260F) or below 230K (-45F))

%Get temperature and flag unrealistic values
temp= X.temperature;
uppertemp= temp > 400;
lowertemp= temp < 230;

%Remove high values
X(uppertemp==1,:)=[];
t(uppertemp==1,:)=[];

%Remove low values
X(lowertemp==1,:)=[];
t(lowertemp==1,:)=[];
end
