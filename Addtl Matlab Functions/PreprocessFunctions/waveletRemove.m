%function waveletRemove(s_seq,nBins)
% Load original one-dimensional signal. 
load leleccum; s = leleccum(1:3920); 

sm = mean(Y_ref.CH4_ppm);
s = Y_ref.CH4_ppm - sm;
%Length of original data
ls = length(s); 

% Perform decomposition of signal at level 3 using db5. 
[c,l] = wavedec(s,5,'db5');

% Reconstruct s from the wavelet decomposition structure [c,l]. 
%a0 = waverec(c,l,'db5');
a0 = appcoef(c,l,'db5',0); length(s)/length(a0)
a1 = appcoef(c,l,'db5',1);length(s)/length(a1)
a2 = appcoef(c,l,'db5',2);length(s)/length(a2)
a3 = appcoef(c,l,'db5',3);length(s)/length(a3)

%Get first three
[cd1,cd2] = detcoef(c,l,[1 2]);

%Calculate reconstruction error
%err = norm(s-a0)

% plot(1:length(s),s,'ko')
% line(1:length(a0),a0,'Color','r')
% title('Original signal')
% 
% figure
% plot(1:length(cd2),cd2,'Color','r')
% line(1:length(cd1),cd1,'Color','g')
% title('Detail Coefficients (cd3)')

figure;
subplot(5,1,1)
line(1:length(s),s,'Color','g');
subplot(5,1,2)
line(1:1:length(a0)*1,a0,'Color','r')
subplot(5,1,3)
line(1:2:length(a1)*2,a1,'Color','r')
subplot(5,1,4)
line(1:4:length(a2)*4,a2,'Color','r')
subplot(5,1,5)
line(1:8:length(a3)*8,a3,'Color','r')




%% Lots more testing
testSig = s;
close all
nlevel = 12;

%% Haar Decomposition
[s,w] = haart(testSig,nlevel);

figure;
plot(t,ch4dat);
ylabel('Reference Methane');
grid on

figure
subplot(4,2,1)
ch4haar = ihaart(s,w,10);
hL = plotyy(t,ch4dat,t,ch4haar);
Ax1 = hL(1);
Ax2 = hL(2);
Ax1.YLim = [1 4]; Ax1.YLabel.String = 'Actual CH4_ppm';
Ax2.YLim = [1 4]; Ax2.YLabel.String = 'Estimate';
grid on
title('Level 10');
subplot(4,2,2)
plot(t,ch4dat-ch4haar)
title('Error (act-est)');
grid on

subplot(4,2,3)
ch4haar = ihaart(s,w,8);
hL = plotyy(t,ch4dat,t,ch4haar);
Ax1 = hL(1);
Ax2 = hL(2);
Ax1.YLim = [1 4]; Ax1.YLabel.String = 'Actual CH4_ppm';
Ax2.YLim = [1 4]; Ax2.YLabel.String = 'Estimate';
title('Level 8');
grid on
subplot(4,2,4)
plot(t,ch4dat-ch4haar)
title('Error (act-est)');
grid on

subplot(4,2,5)
ch4haar = ihaart(s,w,4);
hL = plotyy(t,ch4dat,t,ch4haar);
Ax1 = hL(1);
Ax2 = hL(2);
Ax1.YLim = [1 4]; Ax1.YLabel.String = 'Actual CH4_ppm';
Ax2.YLim = [1 4]; Ax2.YLabel.String = 'Estimate';
title('Level 4');
grid on
subplot(4,2,6)
plot(t,ch4dat-ch4haar)
title('Error (act-est)');
grid on

subplot(4,2,7)
ch4haar = ihaart(s,w,1);
hL = plotyy(t,ch4dat,t,ch4haar);
Ax1 = hL(1);
Ax2 = hL(2);
Ax1.YLim = [1 4]; Ax1.YLabel.String = 'Actual CH4_ppm';
Ax2.YLim = [1 4]; Ax2.YLabel.String = 'Estimate';
title('Level 1');
grid on
subplot(4,2,8)
plot(t,ch4dat-ch4haar)
title('Error (act-est)');
grid on
clear Ax1 Ax2 hL



%% Other wavelet decomposition
%https://www.mathworks.com/videos/understanding-wavelets-part-3-an-example-application-of-the-discrete-wavelet-transform-121284.html
wave1 = 'db2';
wave2 = 'sym8';

%Single level dwt
[cA1,cD1] = dwt(testSig,wave1);
xrec1 = idwt(cA1,zeros(size(cA1)),wave1);

[cA2,cD2] = dwt(testSig,wave2);
xrec2 = idwt(cA2,zeros(size(cA2)),wave2);

%Multilevel dwt
level = 8;
[c1, l1] = wavedec(testSig,level,wave1);

figure;subplot(level+1,1,1); plot(testSig)
for i = 1:level
    
    cA3 = appcoef(c1,l1,wave1,i);
    cD3 = detcoef(c1,l1,i);
    xrec3 = idwt(cA3,[],wave1);
    xrec4 = idwt([],cD3,wave1);
    subplot(level+1,2,2*i);
    plot(xrec3)
    subplot(level+1,2,2*i+1);
    plot(xrec4)
end

[cA4,cD4] = dwt(testSig,wave2);
xrec4 = idwt(cA2,zeros(size(cA2)),wave2,9);

%Plot
f = figure;
plot(t, testSig,'k-',t,xrec1,'r-',t,xrec2,'b-',t,xrec3,'r-',t,xrec4,'b-')
startDate = datetime(2016,9,27,18,0,0);
enddate = datetime(2016,9,28,10,0,0);
%enddate = datetime(2016,10,30);
xlim([startDate enddate]);
xticks(startDate:hours(0.5):enddate);
xtickformat('M/d H:mm')
%datetick('x','mm/dd HH/MM','keeplimits')
xtickangle(90)
grid on
legend('Original',wave1,wave2)
clear f
clear startDate enddate


%% Example from online that is closer
figure;
plot(t,testSig,'k'); hold on;
cols = jet(10);
for i = 4:9
    [C,L] = wavedec(testSig,i,'db2');
    Cnew = C;
    tmp = cumsum(L);
    Cnew(tmp(end-i)+1:tmp(end-1)) = 0;
    Rec_signal=waverec(Cnew,L,'db2');
    plot(t,Rec_signal,'Color',cols(i,:),'linewidth',2); hold on
end
startDate = datetime(2016,9,27,18,0,0);
enddate = datetime(2016,9,28,10,0,0);
xlim([startDate enddate]);
xtickformat('M/d H:mm')
hold off
%legend([{'Ref'} num2cell(num2str(2:10))])
legend('Ref','4','5','6','7','8','9')%'2','3',,'10'
%end