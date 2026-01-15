clear
load('BINGO_GRENG.mat')
load sensor_nnames_xy.mat

str_fields = fieldnames(Sub);
for j=1:length(str_fields)
    str_fields{j}
sub_fields = fieldnames(Sub.(str_fields{j}));sub_fields=sub_fields(7:end);

STs = Sub.(str_fields{j}).trials;

STs=permute(STs,[2, 3, 1]);


Fs=300; t=1:1200; time=t*(1/Fs);

[Nsensors,Ntime,Ntrials]=size(STs);
% average re-ref
%re_STs=[];for i_trial=1:Ntrials, ST_DATA=STs(:,:,i_trial); re_STs(:,:,i_trial)=ST_DATA-mean(ST_DATA);end
%STs=re_STs;

% Task
%tstart=knnsearch(time',2.5);tend=knnsearch(time',4); 
tstart=1; tend=1200;
trialPSD=[];for i_trial=1:Ntrials, ST_DATA=STs(:,tstart:tend,i_trial); 
    [STpsd,faxis]=pspectrum(ST_DATA',Fs,'FrequencyLimits',[1 45],'FrequencyResolution',2); trialPSD(:,:,i_trial)=STpsd'; end 

figure(j),subplot(2,1,1), plot(faxis,pow2db(mean(trialPSD,3))),legend(sensor_names),title('trial averaged PSD')
          subplot(2,1,2),plot(time,STs(:,:,1)),legend(sensor_names) %plot(time,mean(abs(STs),3)),legend(sensor_names),title('single trial')
          
end