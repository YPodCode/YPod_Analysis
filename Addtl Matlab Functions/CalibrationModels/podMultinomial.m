function func = podMultinomial(a)

switch a
    case 1; func = @podMultinomialGen;
    case 2; func = @podMultinomialApply;
    case 3; func = @podMultinomialReport;
end

end

function [mdl, y_hat] = podMultinomialGen(Y,X,~)

%Need as an array
X = table2array(X);
Y = table2array(Y);

%Fit multinomial regression
B = mnrfit(X,Y,'Model','nominal','Interactions','on');

%Verify that Y is categorical
assert(iscategorical(Y),'Y is not a categorical variable!')

%Categories list
catlist = categories(Y);

%Make predictions
pihat = mnrval(B,X);

%Get the maximum likelihood state
[~,I]=max(pihat');

%Get the category that was predicted
y_hat = catlist(I);
y_hat = categorical(y_hat,catlist);

%Export the model
mdl = {B,catlist};

end

function y_hat = podMultinomialApply(X,mdl,~)

%Extract info from fit
B = mdl{1};
catlist = mdl{2};

%Need as an array
X = table2array(X);

%Make predictions
pihat = mnrval(B,X);

%Get the maximum likelihood state
[~,I]=max(pihat');

%Get the category that was predicted
y_hat = catlist(I);
y_hat = categorical(y_hat,catlist);

end

function podMultinomialReport(fittedMdl,~)
try
    
catch err
    disp('Error reporting the multinomial regression model');
end

end