function main_trackTest

addpath(cd(cd('MatlabFunctions')));
rng(1, "twister");

%close all

mins2sec = 60;
hours2sec = 60*mins2sec;
days2sec = 24*hours2sec;

%[sensorData, colours, truth] = simulateDataHarmonics();
%[sensorData, colours, truth] = getData();
%[sensorData, colours, truth] = simulateData2Targets();
ntargets = 10;
[sensorData, colours, truth] = simulateDenseTargets(ntargets);% sensorData = sensorData([2 3]);
%[sensorData, colours] = simulateMMSITest();
%[sensorData, colours] = simulateBuoyScenario();
%------
sensorData{1}.sensor.priorVisProb = 0.5;
sensorData{1}.sensor.rates.meas = 1/(1*hours2sec);
sensorData{1}.sensor.rates.hide = 1/(2*days2sec);
sensorData{1}.sensor.rates.reveal = 1/(1*days2sec);
sensorData{2}.sensor.priorVisProb = 0.4;
sensorData{2}.sensor.rates.meas = 1/(1.1*hours2sec);
sensorData{2}.sensor.rates.hide = 1/(2.1*days2sec);
sensorData{2}.sensor.rates.reveal = 1/(1.1*days2sec);
sensorData{3}.sensor.priorVisProb = 0.3;
sensorData{3}.sensor.rates.meas = 1/(1.2*hours2sec);
sensorData{3}.sensor.rates.hide = 1/(2.2*days2sec);
sensorData{3}.sensor.rates.reveal = 1/(1.2*days2sec);

for i = 1:numel(colours)
    colours(i).q = 1e-5;
end
%------
for i = 1:numel(sensorData)
    sensorData{i}.sensor.gatesdThresh = 5;
end
%colours = [];

%------
% Round times down to second for better comparison with Python
for i = 1:numel(sensorData)
    sensorData{i}.meas.times = dateshift(sensorData{i}.meas.times, "start", "seconds");
end
%------

prior.posMin = [-84 34];
prior.posMax = [  9 62];
prior.speedMetresSD = 10;

% Set up transition model stuff
stationary_speed = (prior.speedMetresSD);
txmodel.isOrnsteinUhlenbeck = true;
txmodel.q_metres = 0.1;
% Stationary velocity s.d. of Ornstein-Uhlenbeck is sqrt(q/(2*K)) so set K
% to make prior velocity the stationary distribution
txmodel.K = txmodel.q_metres/(2*stationary_speed^2);
txmodel.deathRate = 1/(4*days2sec);
% Birth rate?

params.killProbThresh = 0.1;%0.01;
params.measHistLen = 1;%2;
%params.modelHistLen = 1; (assumed to be 1 currently)
% Delete component if log probability less than this
params.compLogProbThresh = log(1e-3);%-inf;%
params.trackstooutput = 'gated';%'mostlikely';%
params.estNumUnconfirmedTargets = 1;

%------
fid = fopen("colours_q.json", "w");
fprintf(fid, jsonencode(colours, PrettyPrint=true));
fclose(fid);
writeMeasJSON(sensorData, "measdata_10target_q.json");
% writeMeasMongoDB(sensorData);
%------
tic
[trackdata, tracks] = tracker(sensorData, txmodel, prior, colours, params);
toc

sd = sensorData;
for i = 1:numel(sd)
    sd{i}.meas.times = seconds(sd{i}.meas.times - trackdata.initDateTime);
end
%td = [trackdata.time trackdata.id trackdata.mean trackdata.cov];
%drawresults(td, [1 3], 4);

save trackdata_matlab_10targets_q_ou

% figure
% drawtrackdata(trackdata, [1 3], true);
% %plotMeasCoords(sd, [1 2]);
% [~,~,idx] = uniquewithcounts(truth.mmsi);
% for i = 1:numel(idx)
%     plot(truth.coords(idx{i},1), truth.coords(idx{i},2), 'k-');
% end
% axis equal
% 
% figure; hold on
% %drawresults(td, 1, 4)
% drawtrackdata(trackdata, 1, true);
% plotMeasCoords(sd, 1);
% 
% plotMMSIProbs(trackdata, 1);
