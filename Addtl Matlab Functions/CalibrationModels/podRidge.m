function func = podRidge(a)
%This fits a purely linear model using all columns of X as predictors and
%also includes interaction terms between each variable
switch a
    case 1; func = @ridgeFit;
    case 2; func = @ridgeApply;
    case 3; func = @ridgeReport;
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function mdl = ridgeFit(Y,X,settingsSet)


xnames = X.Properties.VariableNames;
X = table2array(X);
mdl = cell(size(Y,2),1);
tcol=0;hcol=0;
normMat = zeros(size(X,2),2);

for i = 1:size(X,2)
    minx = min(X(:,i),[],'omitnan');
    maxx = max(X(:,i),[],'omitnan');
    normMat(i,1) = minx;
    normMat(i,2) = maxx;
    X(:,i) = 2*(X(:,i)-minx)./(maxx-minx)-1;
    %Find temperature and humidity columns
    currentCol = xnames{i};
    if any(regexpi(currentCol,'temperature'))
        tcol = i;
        continue;
    elseif any(regexpi(currentCol,'humidity'))
        hcol = i;
        continue;
    end
end
%Extract the temperature and humidity columns
tempDat = X(:,tcol);
humDat = X(:,hcol);

%Copy X for manipulation
X_int = X;
for i = 1:size(X,2)
    currentCol = xnames{i};
    if any(regexpi(currentCol,'temperature')) || any(regexpi(currentCol,'humidity'))
        continue;
    end
    %Calculate interaction terms
    t_int = X(:,i).*tempDat; tn = {[currentCol '_t']};
    h_int = X(:,i).*humDat; hn = {[currentCol '_h']};
    th_int =X(:,i).*tempDat.*humDat; thn = {[currentCol '_th']};
    %Append data
    X_int = [X_int t_int h_int th_int];
    %Keep track of names
    xnames = [xnames tn hn thn];
end


currentPod = settingsSet.podList.podName{settingsSet.loops.j};
nref   = length(settingsSet.fileList.colocation.reference.files.bytes);
if nref==1; reffileName = settingsSet.fileList.colocation.reference.files.name;
else; reffileName = settingsSet.fileList.colocation.reference.files.name{settingsSet.loops.i}; end
currentRef = split(reffileName,'.');
currentRef = currentRef{1};



for i = 1:size(Y,2)
    %Get y as table
    y = table2array(Y(:,i));
    
    %Find appropriate k
    %k = 0:1e-4:5e-2;
    k = logspace(-5,2,100);
    rng(1)
    b = ridge(y,X_int,k);
    
    %Make plot
    nplot = min(20,size(b,1));
    [~,aa]=sort(median(abs(b),2),'descend');
    tempb = b(aa(1:nplot),:);
    tempnames = xnames(aa(1:nplot));
    [~,bb] = sort(median(tempb,2),'descend');
    
    figure('Position',get( groot, 'Screensize' ));
    subplot(1,5,1:3)
    cols = jet(nplot);
    for j = 1:nplot
        semilogx(k,tempb(bb(j),:),'Color',cols(j,:),'LineWidth',2)
        hold on
    end
    %plot(k,b(aa(1:nplot),:),'LineWidth',2)
    %ylim([-100 100])
    grid on
    xlabel('Ridge Parameter')
    ylabel('Standardized Coefficient')
    title(['Ridge Trace of Top ' num2str(nplot) ' Most Important Factors for ' currentRef], 'Interpreter', 'none');
    legend(tempnames(bb),'Location','eastoutside', 'Interpreter', 'none')
    
    subplot(1,5,4:5)
    cols = jet(10);
    rng(1)
    b = ridge(y,X_int,k,0);
%     y_hat = zeros(size(X_int,1),1);
    for j = 1:10
        ind=j*10;
        btemp = b(:,ind);
        y_hat = [ones(size(X_int,1),1) X_int] * btemp;
        scatter(y,y_hat,5,cols(j,:),'filled');
        hold on
    end
    refline(1,0);grid on
    l = legend([split(num2str(k(10:10:100),'%10.2e\n'),'  ');'1-1 line'],...
        'Location','best', 'Interpreter', 'none');
    xlabel('Reference Value')
    ylabel('Estimate')
    title(['Estimates Using Select Values of k for ' currentRef]);

    %Ask user for k value to use
    %knew = input('What k to use? ');
    knew = 5; %For now just use 5, which seems to be okay for everything
    
    %Save the graph of ridge traces
    if settingsSet.savePlots && ishandle(1)
        temppath = [currentPod '_' currentRef '_fold' num2str(settingsSet.loops.kk) '_ridgeParams'];
        temppath = fullfile(settingsSet.outpath,temppath);
        saveas(gcf,temppath,'jpeg');
        clear temppath
        close(gcf)
    end
    
    rng(1)
    b_out = ridge(y,X_int,k);
    b_out = array2table(b_out','VariableNames',xnames);
    
    %Re-calculate using the selected value of k
    mdlobj{1} = normMat;
    rng(1)
    mdlobj{2} = ridge(y,X_int,knew,0);
    mdlobj{3} = b_out;
    mdl{i} = mdlobj;
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function y_hat = ridgeApply(X,mdl,~)

xnames = X.Properties.VariableNames;
X = table2array(X);
tcol=0;hcol=0;

%Predict using fitted coefficients
y_hat = zeros(size(X,1),length(mdl));
for j = 1:length(mdl)
    %Extract fitted factors
    mdlobj = mdl{j};
    normMat = mdlobj{1};
    B = mdlobj{2};
    
    %Normalize X using original statistics
    for i = 1:size(X,2)
        minx = normMat(i,1);
        maxx = normMat(i,2);
        X(:,i) = 2*(X(:,i)-minx)./(maxx-minx)-1;
        %Find temperature and humidity columns
        currentCol = xnames{i};
        
        if any(regexpi(currentCol,'temperature'))
            tcol = i;
            continue;
        elseif any(regexpi(currentCol,'humidity'))
            hcol = i;
            continue;
        end
    end
    
    %Extract the temperature and humidity columns
    tempDat = X(:,tcol);
    humDat = X(:,hcol);
    
    %Copy X for manipulation
    X_int = X;
    for i = 1:size(X,2)
        currentCol = xnames{i};
        if any(regexpi(currentCol,'temperature')) || any(regexpi(currentCol,'humidity'))
            continue;
        end
        %Calculate interaction terms
        t_int = X(:,i).*tempDat; tn = {[currentCol '_t']};
        h_int = X(:,i).*humDat; hn = {[currentCol '_h']};
        th_int =X(:,i).*tempDat.*humDat; thn = {[currentCol '_th']};
        %Append data
        X_int = [X_int t_int h_int th_int];
        %Keep track of names
        xnames = [xnames tn hn thn];
    end
    
    %Apply fitted
    y_hat(:,j) = [ones(size(X_int,1),1) X_int] * B;
end
end
%--------------------------------------------------------------------------

%-------------Report relevant stats (coefficients, etc) about the model-------------
function ridgeReport(fittedMdl,mdlStats,settingsSet)
fittedMdl
end