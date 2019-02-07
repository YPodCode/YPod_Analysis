function [X, t] = podSmooth(X, t, settingsSet)
%{
This function uses the averaging method (median, mean, etc) to average the
input timeseries over some interval
%}

%% Select smoothing method
%Check whether this is reference or pod data
if strcmp(settingsSet.filtering, 'ref')
    %If this is smoothign reference data
    alignMethod = settingsSet.refSmooth;
else
    %If this is smoothign pod data
    alignMethod = settingsSet.podSmooth;
end

%Use the value to select different smoothing functions (subfunctions below)
switch alignMethod
    %These are references to subfunctions below.  See "splitapply"
    %documentation for information about how functions should work
    case 0; smoothFunc = @podmedian;
    case 1; smoothFunc = @podmean;
    case 2; smoothFunc = @podinterp;
    case 3; smoothFunc = @podsmooth;
    case 4; smoothFunc = @podmode;
end
disp(['---Using smooth function: ' func2str(smoothFunc)]);

%% Separate discontinuous windows
%Remove rows with bad time data
X(isnat(t),:)=[];
t(isnat(t))=[];

%Time to group data for analysis
tav = minutes(settingsSet.timeAvg);

%Use typical time between measurements to ID gaps
t1 = t(2:end);
t2 = t(1:end-1);
diff = t1-t2;
clear t1 t2
dt = median(diff,'omitnan');

%If the data has already been averaged we can skip this
if dt == tav
    disp('---Data already at smoothing interval, smoothing skipped');
    %Double check that the timestamps start at the minute
    if tav >= minutes(1)
        t = dateshift(t,'start','minute');
    end
    return;
end

%Define "gap" size based on reference or pod data
if strcmp(settingsSet.filtering, 'ref')
    delt = datenum(dt)*5;
else
    delt = datenum(tav);
end

%Sometimes data gets imported weird and the median dt is 0
if delt == 0
    delt = settingsSet.timeAvg*60;
end
diff = datenum(diff);

%% Separate groups of data that are far apart in time to avoid trying to interpolate in between those groups
tsegs = ones(size(t,1),1,'uint16');
seg = 1;
for i = 2:size(t,1)
    %New segment if the gap is more than 5x the typical recording interval
    if diff(i-1)>(delt)
        seg=seg+1;
    end
    tsegs(i)=seg;
end
tsegs(end)=max(tsegs);
clear diff delt

%% Apply smoothing to each segment
for i = 1:max(tsegs)
    disp(['----Smoothing seg ' num2str(i) ' of ' num2str(max(tsegs))])
    %If this segment is just one reading, don't need to "smooth" it
    if size(X(tsegs==i,:),1)==1
        X_seg = X(tsegs==i,:);
        t_seg = t(tsegs==i,:);
    else
        [X_seg,t_seg] = smoothFunc(X(tsegs==i,:),t(tsegs==i,:));
    end

    if i==1
        X_temp=X_seg;
        t_temp=t_seg;
    else
        X_temp=[X_temp;X_seg];
        t_temp=[t_temp;t_seg];
    end
end

%% Fix time vector
%Convert timestamps back into datetime format (from posix time, which was needed for some math operations)
%t_temp = datetime(t_temp,'ConvertFrom','posixtime');

%% Return the smoothed data and timestamps
t = t_temp;
X = X_temp;


%% ------------------------SMOOTHING SUBFUNCTIONS BELOW------------------------
%------------------Median (method 0)-------------------------------
    function [X,t] = podmedian(X,t)
        %Make groups based on the time window selected
        [G,t_steps] = podGroup(t);
                
        %Assign time steps to the start of each window
        t = t_steps(1:max(G))';
        
        %Initialize a temporary matrix for smoothed values
        tempMat = X(1:max(G),:);
        
        %Function for median
        splitfunc = @(x)(median(x,'omitnan'));
        
        %Apply to each column of X
        for j=1:size(X,2)
            %Apply the averaging function selected to each column
            tempArray = splitapply(splitfunc,X(:,j),G);
            tempMat(:,j) = array2table(tempArray);
            clear tempArray
        end
        
        %Return X
        X = tempMat;
    end
%-------------------------------------------------------

%------------------Mean (method 1)---------------------------------
    function [X,t] = podmean(X,t)
        %Make groups based on the time window selected
        [G,t_steps] = podGroup(t);
                
        %Assign time steps to the start of each window
        t = t_steps(1:max(G))';
        
        %Initialize a temporary matrix for values
        tempMat = X(1:max(G),:);
        
        %Function for mean
        splitfunc = @(x)(mean(x,'omitnan'));
        
        for j=1:size(X,2)
            %Apply the averaging function selected to each column
            tempArray = splitapply(splitfunc,X(:,j),G);
            tempMat(:,j) = array2table(tempArray);
            clear tempArray
        end
        
        %Return X
        X = tempMat;
    end
%-------------------------------------------------------

%------------------Mode (method 4)---------------------------------
    function [X,t] = podmode(X,t)
        %Make groups based on the time window selected
        [G,t_steps] = podGroup(t);
                
        %Assign time steps to the start of each window
        t = t_steps(1:max(G))';
        
        %Initialize a temporary matrix for values
        tempMat = X(1:max(G),:);
        
        %Function for mode (ignores NaN by default)
        splitfunc = @(x)(mode(x));
        
        %Find the mode of each variable
        for j=1:size(X,2)
            %Apply the averaging function selected to each column
            tempArray = splitapply(splitfunc,X(:,j),G);
            tempMat(:,j) = array2table(tempArray);
            clear tempArray
        end
        
        %Return X
        X = tempMat;
    end
%-------------------------------------------------------

%------------------Linear Interpolation (method 2)-----------------
    function [X,t] = podinterp(X,t)
        %Make groups based on the time window selected
        [G,t_steps] = podGroup(t);
                
        %Assign time steps to the start of each window
        t_windows = t_steps(1:max(G))';
        
        %Make the smoothing window as "k" fraction of the data
        k = ceil((tav*2)/dt);
        k = k/size(X,1);
        if k>1; k=1; elseif k<=0,k=0.05;end
        
        %Make table to hold smoothed values
        tempMat = X(1:max(G),:);
        
        %% Smooth each column
        for j = 1:size(tempMat,2)
            %Get the current variable
            tempX = table2array(X(:,j));
            %Fit a smoothing spline with parameter "k"
            interpFit = fit(posixtime(t),tempX,'linearinterp');
            %Use that smooth to predict on the new times
            tempX = interpFit(posixtime(t_windows));
            %Overwrite the old column with those values
            tempMat(:,j) = table(tempX,'VariableNames',tempMat.Properties.VariableNames(j));
        end
        
        %% Overwrite X and t
        X = tempMat;
        t = t_windows;
    end
%-------------------------------------------------------

%------------------LOWES (method 3)--------------------------------
    function [X,t] = podsmooth(X,t)
        %Make groups based on the time window selected
        [G,t_steps] = podGroup(t);
                
        %Assign time steps to the start of each window
        t_windows = t_steps(1:max(G))';
        
        %Make the smoothing window as "k" fraction of the data
        k = ceil((tav*2)/dt);
        k = k/size(X,1);
        if k>1; k=1; elseif k<=0,k=0.05;end
        
        %Make table to hold smoothed values
        tempMat = X(1:max(G),:);
        splitfunc = @(x)(median(x,'omitnan'));
        
        %% Smooth each column
        for j = 1:size(tempMat,2)
            %Get the current variable
            tempX = table2array(X(:,j));
            %Fit a robust smoothing spline with parameter "k"
            smoothedX = smooth(tempX,k,'rloess');
            %Get median of smoothed values within each window
            tempX = splitapply(splitfunc,smoothedX,G);
            %Overwrite the old column with those values
            tempMat(:,j) = table(tempX,'VariableNames',tempMat.Properties.VariableNames(j));
        end
        
        %% Overwrite X and t
        X = tempMat;
        t = t_windows;
    end
%-------------------------------------------------------

%------------------SEGMENT GROUPING---------------------
    function [G, t_steps] = podGroup(t)
        
        %Split the time at the start of each averaging window, e.g. 5 minutes        
        [G, t_steps] = discretize(t,tav);
        
        %Reduce the size of G for speed
        G = cast(G,'uint16');
        
        %Make sure that the groupings of G go from 1:nGroups
        tempG = cast(ones(size(G,1),1),'uint16');
        for zz = 2:length(tempG)
            if G(zz)==G(zz-1)
                tempG(zz)=tempG(zz-1);
            else
                tempG(zz)=tempG(zz-1)+1;
            end
        end
        
        %Re-assign into G
        G = tempG;
        
    end
%-------------------------------------------------------

end
