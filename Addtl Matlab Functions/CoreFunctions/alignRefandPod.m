function [Y,X,t] = alignRefandPod(Y,yt,X,xt,~)
%Align by datetime columns to ensure that each reference measurement is
%matched by a Pod entry.  Returns nx1 table for Y, nxp table for X, and nx1
%timestamps in t

%Join time to Y for alignment
Y.datetime = yt;
X.datetime = xt;

%Align the two matrices using the datetime column, eliminating entries with
%no match in the other
temp = innerjoin(Y,X,'Keys','datetime');
Y.datetime=[];
X.datetime=[];

%Extract and then remove the datetime column (its location is not fixed)
t = temp.datetime;
temp.datetime=[];

%The next columns are from Y
Y = temp(:,1:size(Y,2));

%The rest of the columns are from X
X = temp(:,size(Y,2)+1:end);

end

