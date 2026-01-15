clear all; close all
addpath(genpath("./Data")) % Add data to path
addpath(genpath("./Dependencies")) % Add depencies to path
myLogReport={};
subID = 6;
%recordingDay = 3;

SubCounter=0;
for subID = [1:20]
    clearvars -except subID recordingDay Sub myLogReport SubCounter SilScore_words SilScore_lang tt1 tt2
    SubCounter = SubCounter+1;
    if subID<10
        textSubID = "./Data/BINGO_S0";
    else
        textSubID = "./Data/BINGO_S";
    end
    myData = load_xdf(strcat(textSubID, num2str(subID), "_GRENG.xdf")); % Load data
    for iStream = 1:length(myData)
        if strcmp(myData{iStream}.info.type,'Markers')
            tmpMarkers= myData{iStream}.time_series;
            dataMarkers = {tmpMarkers{2,:}};
            timeMarkers = myData{iStream}.time_stamps;
        end
        if strcmp(myData{iStream}.info.type,'EEG')
            dataEEG = double(myData{iStream}.time_series);
            timeEEG = myData{iStream}.time_stamps;
            fsEEG = round(myData{iStream}.info.effective_srate);
        end
    end

    if timeEEG(end)-timeEEG(1)<timeMarkers(end)-timeMarkers(1)
        myLogReport{end+1} = (strcat('check EEG duration ',textSubID, num2str(subID), "_DAY", num2str(recordingDay), ".xdf"));
    end

    if fsEEG <299
        myLogReport{end+1} = (strcat('check EEGfs ',textSubID, num2str(subID), "_DAY", num2str(recordingDay), ".xdf"));
    end

    [b,a]=butter(3, [30 100]/(fsEEG/2));
    filteredEEG=filtfilt(b,a,dataEEG')';
    wo = 50/(300/2);
    bw = wo/35;
    [b,a] = iirnotch(wo,bw);
    filteredEEG=filtfilt(b,a,filteredEEG')';
    excludedSensors = [16,17,20,21,24];
    filteredEEG = filteredEEG(setdiff([1:24],excludedSensors),:);
    denoisedEEG=filteredEEG;

    load chanlocs_dsi24.mat
    %% Trial Segmentation
    for i=1:length(timeMarkers)
        idxStartEEG(i) = find(abs((timeEEG-timeMarkers(i)))==min(abs(timeEEG-timeMarkers(i))));
    end
    %trialStartTime = idxStartEEG(9:465);
    trialsEEG = [];
    trialsVisualEEG = [];
    labels = [];
    uniqueMarkers = setdiff(unique(dataMarkers),{'Break','Welcome','EO','EC','EXP_begin','EndOfExperiment'});

    trialsIndexStart = find(strcmp(dataMarkers, 'EXP_begin'));
    for i=trialsIndexStart+1:length(dataMarkers)
        tmp = find(strcmp(uniqueMarkers, dataMarkers{i}));
        if ~isempty(tmp)
            trialsEEG(end+1,:,:) = denoisedEEG(:,idxStartEEG(i)+0*fsEEG:idxStartEEG(i)+4*fsEEG-1);

            %trialsVisualEEG(end+1,:,:) = filteredEEG(:,idxStartEEG(i)+0*fsEEG:idxStartEEG(i)+1.5*fsEEG);
            labels{end+1}=uniqueMarkers{tmp};
        else
            dataMarkers{i};
        end
    end


    if length(labels) ~=301
        myLogReport{end+1} = (strcat('check number of trials ',textSubID, num2str(subID), "_GRENG.xdf"));
    end

    eyesOpenIndex = find(strcmp(dataMarkers, 'EO'));
    eyesOpenRestingEEG = denoisedEEG(:,idxStartEEG(eyesOpenIndex):idxStartEEG(eyesOpenIndex)+60*fsEEG-1);
    eyesClosedIndex = find(strcmp(dataMarkers, 'EC'));
    eyesClosedRestingEEG = denoisedEEG(:,idxStartEEG(eyesClosedIndex):idxStartEEG(eyesClosedIndex)+60*fsEEG-1);

    Sub.(strcat("S", num2str(subID))).eyesClosedRestingEEG = eyesClosedRestingEEG;
    Sub.(strcat("S", num2str(subID))).eyesOpenRestingEEG = eyesOpenRestingEEG;
    Sub.(strcat("S", num2str(subID))).trials = trialsEEG;
    Sub.(strcat("S", num2str(subID))).labels = labels;
    Sub.(strcat("S", num2str(subID))).fs = fsEEG;
    Sub.(strcat("S", num2str(subID))).time = [1:size(trialsEEG,3)]/fsEEG-2.5;
    Sub.(strcat("S", num2str(subID))).Antenna.gr = find(contains(labels,'Κεραία'))
    Sub.(strcat("S", num2str(subID))).Antenna.eng = find(contains(labels,'Antenna'))
    Sub.(strcat("S", num2str(subID))).Apple.gr = find(contains(labels,'Μήλο'))
    Sub.(strcat("S", num2str(subID))).Apple.eng = find(contains(labels,'Apple'))
    Sub.(strcat("S", num2str(subID))).Arrow.gr = find(contains(labels,'Βέλος'))
    Sub.(strcat("S", num2str(subID))).Arrow.eng = find(contains(labels,'Arrow'))
    Sub.(strcat("S", num2str(subID))).Belt.gr = find(contains(labels,'Ζώνη'))
    Sub.(strcat("S", num2str(subID))).Belt.eng = find(contains(labels,'Belt'))
    Sub.(strcat("S", num2str(subID))).Button.gr = find(contains(labels,'Κουμπί'))
    Sub.(strcat("S", num2str(subID))).Button.eng = find(contains(labels,'Button'))
    Sub.(strcat("S", num2str(subID))).Candle.gr = find(contains(labels,'Κερί'))
    Sub.(strcat("S", num2str(subID))).Candle.eng = find(contains(labels,'Candle'))
    Sub.(strcat("S", num2str(subID))).Compass.gr = find(contains(labels,'Πυξίδα'))
    Sub.(strcat("S", num2str(subID))).Compass.eng = find(contains(labels,'Compass'))
    Sub.(strcat("S", num2str(subID))).Dice.gr = find(contains(labels,'Ζάρι'))
    Sub.(strcat("S", num2str(subID))).Dice.eng = find(contains(labels,'Dice'))
    Sub.(strcat("S", num2str(subID))).Feather.gr = find(contains(labels,'Φτερό'))
    Sub.(strcat("S", num2str(subID))).Feather.eng = find(contains(labels,'Feather'))
    Sub.(strcat("S", num2str(subID))).Guitar.gr = find(contains(labels,'Κιθάρα'))
    Sub.(strcat("S", num2str(subID))).Guitar.eng = find(contains(labels,'Guitar'))
    Sub.(strcat("S", num2str(subID))).Pencil.gr = find(contains(labels,'Μολύβι'))
    Sub.(strcat("S", num2str(subID))).Pencil.eng = find(contains(labels,'Pencil'))
    Sub.(strcat("S", num2str(subID))).Plate.gr = find(contains(labels,'Πιάτο'))
    Sub.(strcat("S", num2str(subID))).Plate.eng = find(contains(labels,'Plate'))
    Sub.(strcat("S", num2str(subID))).Saddle.gr = find(contains(labels,'Σέλα'))
    Sub.(strcat("S", num2str(subID))).Saddle.eng = find(contains(labels,'Saddle'))
    Sub.(strcat("S", num2str(subID))).Vehicle.gr = find(contains(labels,'Όχημα'))
    Sub.(strcat("S", num2str(subID))).Vehicle.eng = find(contains(labels,'Vehicle'))
    Sub.(strcat("S", num2str(subID))).Wheel.gr = find(contains(labels,'Ρόδα'))
    Sub.(strcat("S", num2str(subID))).Wheel.eng = find(contains(labels,'Wheel'))

    trialsEEG= Sub.(strcat("S", num2str(subID))).trials(:,:,Sub.(strcat("S", num2str(subID))).time>0);
    %trialsVisualEEG= trialsVisualEEG(1:30,:,:);
    labels=Sub.(strcat("S", num2str(subID))).labels;

    %% Calculate Covariances
    for i_trial=1:length(labels)
        filteredCovariances{i_trial}=cov(squeeze(trialsEEG(i_trial,:,:))');% I keep only the first ten seconds. You can try more options
    end
    % Initial covariance space
    figure,mydist=zeros(size(filteredCovariances,2),size(filteredCovariances,2));
    for i_trial=1:size(filteredCovariances,2)
        for j_trial=1:size(filteredCovariances,2)-1
            mydist(i_trial,j_trial)=(distance_riemann(squeeze(filteredCovariances{i_trial}),squeeze(filteredCovariances{j_trial})));
            mydist(j_trial,i_trial)=mydist(i_trial,j_trial);
        end
    end
    for i_trial=1:size(filteredCovariances,2)
        mydist(i_trial,i_trial)=0;
    end
    Y = cmdscale(mydist);
    Dtriu = mydist(find(tril(ones(length(mydist)),-1)))';
    gscatter(Y(:,1)',Y(:,2)',labels')
    SilScore_words = mean(silhouette(Y(:,[1:2]), labels'));

    unilabels = unique(labels);
    labels_greng={};
    for i_label=1:length(labels)
        if (isempty(find(contains({unilabels{1:15}},labels{i_label}))))
            labels_greng{i_label} = 'Greek';
        else
            labels_greng{i_label} = 'English';
        end
    end
    gscatter(Y(:,1)',Y(:,2)',labels_greng')
    SilScore_lang = mean(silhouette(Y(:,[1:2]), labels_greng'));

    labelsCat = categorical(labels);

    % Compute individual silhouette values
    [s_words, h] = silhouette(Y(:,[1:10]), labelsCat);

    % Identify unique clusters
    clusters = categories(labelsCat);
    numClusters = numel(clusters);

    % Preallocate array for mean silhouette scores
    meanSilhouettePerCluster = zeros(numClusters, 1);

    % Compute mean silhouette score for each cluster
    for i = 1:numClusters
        idx = labelsCat == clusters{i};
        meanSilhouettePerCluster(i) = mean(s_words(idx));
    end

    % Display results in a table
    tt1{SubCounter} = table(clusters, meanSilhouettePerCluster, ...
        'VariableNames', {'ClusterLabel', 'MeanSilhouetteScore'})


    labelsCat = categorical(labels_greng);
    % Compute individual silhouette values
    [s_langs, h] = silhouette(Y(:,[1:10]), labelsCat);

    % Identify unique clusters
    clusters = categories(labelsCat);
    numClusters = numel(clusters);

    % Preallocate array for mean silhouette scores
    meanSilhouettePerCluster = zeros(numClusters, 1);

    % Compute mean silhouette score for each cluster
    for i = 1:numClusters
        idx = labelsCat == clusters{i};
        meanSilhouettePerCluster(i) = mean(s_langs(idx));
    end

    % Display results in a table
    tt2{SubCounter} = table(clusters, meanSilhouettePerCluster, ...
        'VariableNames', {'ClusterLabel', 'MeanSilhouetteScore'})
end
