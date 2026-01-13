clearvars -except myWordScore_ALL myFreqScore_ALL
load('BINGO_GRENG.mat')


Fs=300; t=1:1200;time=t*(1/Fs);
tstart=knnsearch(time',2.5);tend=knnsearch(time',4); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
str_fields = fieldnames(Sub);
for j=1:length(str_fields)
    str_fields{j}
sub_fields = fieldnames(Sub.(str_fields{j}));sub_fields=sub_fields(7:end);

ALL_DiscrMaps=[]; %for i1=1:3,for i2=1:3

mySTs = Sub.(str_fields{j}).trials;

mySTs=permute(mySTs,[2, 3, 1]);

for i=1:length(sub_fields)
i
STs = cat(3,mySTs(:,:,Sub.(str_fields{j}).(sub_fields{i}).gr),mySTs(:,:,Sub.(str_fields{j}).(sub_fields{i}).eng));
class_labels = [Sub.(str_fields{j}).labels(Sub.(str_fields{j}).(sub_fields{i}).gr) Sub.(str_fields{j}).labels(Sub.(str_fields{j}).(sub_fields{i}).eng)]';
[Nsensors,Ntime,Ntrials]=size(STs);
% average re-ref
%re_STs=[];for i_trial=1:Ntrials, ST_DATA=STs(:,:,i_trial); re_STs(:,:,i_trial)=ST_DATA-mean(ST_DATA);end,STs=re_STs;

% class_labels= LABELS{i1,i2};
% class_labels={}

%% Step1 : constructing wavelet filterbanks
fb = cwtfilterbank('SignalLength',Ntime,'SamplingFrequency',Fs,'FrequencyLimits',[1 45],'VoicesPerOctave',10);

%% Step2  : Estimating  single-trial (time-varying) CWT-transform & computing averaged Scalograms profiles
WordScore=[];FreqScore=[]; parfor i_sensor=1:Nsensors, i_sensor/Nsensors

WT=[];for i_trial=1:Ntrials,
     signal=STs(i_sensor,:,i_trial); [wt,Faxis,coi]=cwt(zscore(signal),'FilterBank',fb);  % knnsearch(Faxis,35)   knnsearch(Faxis,5)
      WT(:,:,i_trial)=abs(wt);end,
    %SCAL1=abs(mean(WT(:,:,labels==1),3));SCAL2=abs(mean(WT(:,:,labels==2),3));
     %SCAL1=mean(abs(WT(:,:,:)),3); % SCAL1=std(abs(WT(:,:,:)),[],3);
     %sensorSCAL(:,:,i_sensor)=SCAL1;end   
     signal=STs(1,:,1);[wt,Faxis,coi]=cwt(signal,'FilterBank',fb);

no_classes=numel(unique(class_labels))
llabels=unique(class_labels);
%r_pairs=[];for ii=1:no_classes
%    rest_labels=setdiff([1:no_classes],ii);
%     r_pairs=[r_pairs; ii*ones(3,1),kept_labels']]; end 

PairWiseMaps=[];for i_class=1:no_classes
AAA1=WT(:,:,string(class_labels)==string(llabels(i_class)));    
AA1=reshape(AAA1,[numel(Faxis)*Ntime,size(AAA1,3)])';         
         rest_labels=setdiff([1:no_classes],i_class);
          

AAA2=WT(:,:,find(ne(string(class_labels),string(llabels(i_class))))); 
AA2=reshape(AAA2,[numel(Faxis)*Ntime,size(AAA2,3)])';       
paired_labels= [ones(size(AAA1,3),1);2*ones(size(AAA2,3),1)];
[~, Z] = rankfeatures([abs(AA1);abs(AA2)]',paired_labels,'criterion','wilcoxon');
PairWiseMaps(:,:,i_class)=reshape(Z,numel(Faxis),Ntime); end
%WordScore(i_sensor,:)=squeeze(mean(mean(PairWiseMaps,2),1))';
WordScore(i_sensor,:)=squeeze(max(mean(PairWiseMaps(:,tstart:tend,:),2),[],1))';
%FreqScore(i_sensor,:)=mean(mean(PairWiseMaps(:,tstart:tend,:),2),3)';
FreqScore(i_sensor,:)=max(mean(PairWiseMaps(:,tstart:tend,:),2),[],3)';
end

myWordScore(i,:)=WordScore(:,1);
myFreqScore(i,:,:)=FreqScore;

end
myWordScore_ALL(j,:,:)=myWordScore;
myFreqScore_ALL(j,:,:,:)=myFreqScore;
end

load sensor_nnames_xy.mat

signal=STs(1,:,1);[wt,Faxis,coi]=cwt(signal,'FilterBank',fb);
figure(25),clf,subplot(2,1,1),imagesc(squeeze(mean(myWordScore_ALL,1))'),colormap hot,
colorbar
yticks([1:19]),yticklabels(sensor_names),ylabel('#sensor'),xticks([1:15]),xticklabels(sub_fields)
clim([1.2 2])
FreqScore_avg = squeeze(mean(myFreqScore_ALL,1));FreqScore_avg=squeeze(mean(FreqScore_avg,1));
subplot(2,1,2),imagesc(flip(FreqScore_avg,2)),colorbar,
xticks(1:5:numel(Faxis)),xticklabels((round(Faxis(flip(xticks))))),
yticks([1:19]),yticklabels(sensor_names),xlabel('Hz'),ylabel('#sensor')
xticks(([knnsearch(flip(Faxis),[1:5:45]')'])),xticklabels([1:5:45])

% 
% signal=STs(1,:,1);[wt,Faxis,coi]=cwt(signal,'FilterBank',fb);
% llabels=unique(class_labels);
% figure(22),
% subplot(2,1,1),imagesc(WordScore),colormap hot, colorbar
% yticks([1:19]),yticklabels(sensor_names),ylabel('#sensor'),xticks([1:13]),xticklabels(llabels)
% subplot(2,1,2),imagesc(flip(FreqScore,2)),colorbar,
% %xticks(1:5:numel(Faxis)),xticklabels((round(Faxis(flip(xticks))))),
% yticks([1:19]),yticklabels(sensor_names),xlabel('Hz'),ylabel('#sensor')
% xticks(([knnsearch(flip(Faxis),[1:5:45]')'])),xticklabels([1:5:45])
