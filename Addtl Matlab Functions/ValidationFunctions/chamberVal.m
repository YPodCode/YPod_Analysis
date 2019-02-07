function valList = chamberVal(Y, X, t, n)
%Select groups of data from out to in (group 1 gets the first and last x points and group n is the middle)

% %Convert Y to matrix for maths
% Y = table2array(Y);
% 
% %Initialize the list of points to default points into the last fold
% valList = ones(size(Y,1),1)*n;
% 
% %Only look at points that are not exactly 0
% selectList = 1:size(Y,1);
% medy = median(Y(:,1),'omitnan');
% selectList = selectList(Y>(medy/100) & Y>1e-3);
% 
% %Number of points per group
% L_drop = round(length(selectList)/(n-1),0);
% gr = 1;
% for i = 1:length(selectList)
%     if mod(i,L_drop)==0
%         gr = gr+1;
%     end
%     valList(selectList(i))=gr;
% end


%% Use last set of data for validation
%Initialize the list of points to default points into the last fold
valList = ones(size(Y,1),1)*2;
t = datenum(t);
t_threshold = datenum(datetime(2018,7,1));
valList(t>t_threshold)=1;


end