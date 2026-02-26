function main_denseTargetsTest2

addpath(cd(cd('MatlabFunctions')));

rng(1, 'twister')

estNumUnconfirmedTargets = 10;%10.^2;
ntargets = 10; % Number of targets to simulate

[sensorData, colours, truth] = simulateDenseTargets(ntargets);
testExpNumTargets(sensorData, colours, truth, estNumUnconfirmedTargets);

%--------------------------------------------------------------------------

function [ntracksTot, ntracksEnd, score] = testExpNumTargets(...
    sensorData, colours, truth, estNumUnconfirmedTargets)

ntargets = size(truth.lonLat, 1);
verbose = true;

mins2sec = 60;
hours2sec = 60*mins2sec;
days2sec = 24*hours2sec;

for i = 1:numel(sensorData)
    sensorData{i}.sensor.gatesdThresh = 5;
end
sensorData = sensorData(1:2);

%------
% Strip out targets of interest
targetIds = [4 10];%1:ntargets;%[4 10];%[2 4 8 10];%[7 8];%[2 4 8];%1:ntargets;%[7 8];%
truth = selectrows(truth, targetIds);
for i = 1:numel(sensorData)
    sensorData{i}.meas = selectrows(sensorData{i}.meas,...
        ismember(sensorData{i}.meas.targetNum, targetIds));
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

params.killProbThresh = 0.01;
params.measHistLen = 2;
%params.modelHistLen = 1; (assumed to be 1 currently)
% Delete component if log probability less than this
params.compLogProbThresh = log(1e-3);
params.trackstooutput = 'mostlikely';
params.outputMostLikelyComp = true;
params.estNumUnconfirmedTargets = estNumUnconfirmedTargets;

params.truth = truth;
[trackdata, tracks] = tracker(sensorData, txmodel, prior, colours, params, verbose);
trackdata.targetNum = getTrueTargetNum(trackdata, sensorData);

ntracksTot = max(trackdata.id);
ntracksEnd = numel(tracks);
score = getTracksScore(trackdata, sensorData);

if verbose
    figure
    drawtrackdata(trackdata, [1 3], true);
    plot(truth.lonLat(:,1), truth.lonLat(:,2), 'k+')
    for i = 1:numel(targetIds)
        text(truth.lonLat(i,1), truth.lonLat(i,2), ['  ' num2str(targetIds(i))]);
    end
    plotMeasCoords(sensorData, [1 2], {'r.','k.'})
    xlabel('Lon'); ylabel('Lat');
    
    figure
    drawtrackdata(trackdata, 1, true);
    plotMeasCoords(sensorData, 1, {'r.','k.'})
    xlabel('Time'); ylabel('Lon');
    
    figure
    drawtrackdata(trackdata, 3, true);
    plotMeasCoords(sensorData, 2, {'r.','k.'})
    xlabel('Time'); ylabel('Lat');
end
