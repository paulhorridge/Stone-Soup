function plot_python_tracks(trackfile)

addpath(cd(cd('MatlabFunctions')));
rng(1, "twister");

if ~exist("trackfile", "var")
    trackfile = "C:\Users\prh\PyCharmProjects\Stone-Soup\stonesoup\" +...
        "PRHCode\Data\trackout_10target_q.json";
end

matlabdata = load("trackdata_matlab_10targets_q_ou.mat");

track_data = jsondecode(fileread(trackfile));
colour_data = jsondecode(fileread("C:\Users\prh\PyCharmProjects\" +...
    "Stone-Soup\stonesoup\PRHCode\Data\colours.json"));
sensor_types = cellfun(@string, fieldnames(...
    track_data(1).tracks(1).components(1).log_vis_given_existence));
nscans = numel(track_data);
statedim = 4;

% Get unique track ids in track data
uq_track_ids = [];
for k = 1:nscans
    if ~isempty(track_data(k).tracks)
        uq_track_ids = union(uq_track_ids, [track_data(k).tracks.id]);
    end
end
ntracks = numel(uq_track_ids);

% Get which track ids are in each scan and which track number it is
track_in_scan = zeros(numel(uq_track_ids), nscans);
for k = 1:nscans
    if ~isempty(track_data(k).tracks)
        [ism, idx] = ismember(uq_track_ids, [track_data(k).tracks.id]);
        track_in_scan(:,k) = idx;
    end
end

tracks = cell(1, ntracks);
for ti = 1:ntracks
    scanidx = find(track_in_scan(ti,:));
    this_tracklen = numel(scanidx);
    tracks{ti}.timestamp = NaT(1, this_tracklen);
    tracks{ti}.means = zeros(statedim, this_tracklen);
    tracks{ti}.covs = zeros(statedim, statedim, this_tracklen);
    tracks{ti}.log_exist_prob = zeros(1, this_tracklen);

    c_mn = cell(1, this_tracklen);
    c_cv = cell(1, this_tracklen);
    vis_logp = cell(1, this_tracklen);
    for ki = 1:this_tracklen
        k = scanidx(ki);
        this_data = track_data(k).tracks(track_in_scan(ti,k));
        [mns, cvs, logws] = get_state_distribution_mixture(this_data.components);
        [c_mn{ki}, c_cv{ki}] = get_colour_distributions(this_data.components);
        vis_logp{ki} = get_log_visibility_probs(this_data.components);
        tracks{ti}.timestamp(ki) = datetime(this_data.timestamp);
        [tracks{ti}.means(:,ki), tracks{ti}.covs(:,:,ki)] =...
            mergegaussians(mns, cvs, logws);
        tracks{ti}.log_exist_prob(ki) = sumvectorinlogs(...
            [this_data.components.log_weight] +...
            [this_data.components.log_exist_prob]);
    end
    fn = fieldnames(c_mn{1});
    for fi = 1:numel(fn)
        tracks{ti}.colours.(fn{fi}).mean = cellfun(@(x)x.(fn{fi}), c_mn);
        tracks{ti}.colours.(fn{fi}).var = cellfun(@(x)x.(fn{fi}), c_cv);
    end
    fn = fieldnames(vis_logp{1});
    for fi = 1:numel(fn)
        tracks{ti}.log_visibility_given_exist.(fn{fi}) =...
            cellfun(@(x)x.(fn{fi}), vis_logp);
    end

end

% Colours
% =======
hfigs.colours = figure;
if exist("matlabdata", "var")
    % Plot matlab colours
    for i = 1:numel(colour_data)
        % matlab index of current colour name
        colour_i = find(arrayfun(@(x)string(x.name), matlabdata.colours)...
            == colour_data(i).name);
        subplot(2,4,i); hold on
        mat_uq_idx = unique(matlabdata.trackdata.id);
        for ii = 1:numel(mat_uq_idx)
            idx = matlabdata.trackdata.id==mat_uq_idx(ii);
            t = matlabdata.trackdata.time(idx);
            mn = matlabdata.trackdata.colourMeans(idx,colour_i);
            std = sqrt(matlabdata.trackdata.colourVars(idx,colour_i));
            plot(t, mn, 'b-')
            plot(t, mn + std, 'b-')
            plot(t, mn - std, 'b-')
        end
        title(colour_data(i).name)
    end
end
% Plot Python colours
for i = 1:numel(colour_data)
    subplot(2,4,i); hold on
    for ii = 1:ntracks
        t = tracks{ii}.timestamp;
        mn = tracks{ii}.colours.(colour_data(i).name).mean;
        std = sqrt(tracks{ii}.colours.(colour_data(i).name).var);
        plot(t, mn, 'm--')
        plot(t, mn + std, 'm--')
        plot(t, mn - std, 'm--')
    end
end
sgtitle("Colours")

% Coords
% ======
hfig.coords = figure; hold on
idx = [1 3];
if exist("matlabdata", "var")
    for i = 1:numel(idx)
        subplot(numel(idx),1,i)
        drawtrackdata(matlabdata.trackdata, idx(i), true, {'b'});
    end
end
for i = 1:numel(idx)
    for ti = 1:numel(tracks)
        subplot(numel(idx),1,i); hold on
        mn = tracks{ti}.means(idx(i),:);
        std = sqrt(permute(tracks{ti}.covs(idx(i),idx(i),:), [2 3 1]));
        plot(tracks{ti}.timestamp, mn, 'm+--')
        plot(tracks{ti}.timestamp, mn + std, 'm--')
        plot(tracks{ti}.timestamp, mn - std, 'm--')
    end
end
sgtitle("Coords")

% Existence probability
% =====================
hfig.existprob = figure; hold on
if exist("matlabdata", "var")
    uq_track_ids = unique(matlabdata.trackdata.id);
    for i = 1:numel(uq_track_ids)
        idx = matlabdata.trackdata.id==uq_track_ids(i);
        t = matlabdata.trackdata.time(idx);
        pe = matlabdata.trackdata.existProb(idx);
        plot(t, pe, 'b.-')
    end
end
for i = 1:numel(tracks)
    plot(tracks{i}.timestamp, exp(tracks{i}.log_exist_prob), 'm--')
end
sgtitle("exist probability")

% Visibility probability
% ======================
hfig.visprob = figure; hold on
if exist("matlabdata", "var")
    for j = 1:numel(sensor_types)
        subplot(numel(sensor_types), 1, j); hold on
        sensor_i = find(sensor_types(j)==cellfun(@(x)string(x.name),...
            matlabdata.sensorData));
        for i = 1:numel(uq_track_ids)
            idx = matlabdata.trackdata.id==uq_track_ids(i);
            t = matlabdata.trackdata.time(idx);
            pv = matlabdata.trackdata.visProbs(idx,sensor_i);
            plot(t, pv, 'b-')
        end
    end
end
for j = 1:numel(sensor_types)
    subplot(numel(sensor_types), 1, j); hold on
    for i = 1:numel(tracks)
        t = tracks{i}.timestamp;
        pv = exp(tracks{i}.log_visibility_given_exist.(sensor_types(j)) +...
            tracks{i}.log_exist_prob);
        plot(t, pv, 'm--')
    end
    title(sensor_types{j})
end
sgtitle("visibility probability")

%keyboard

end

%--------------------------------------------------------------------------

function [mns, cvs, logws] = get_state_distribution_mixture(components)

ncomponents = numel(components);
logws = [components.log_weight];
statedim = size(components(1).state_distribution.mean, 1);
mns = zeros(statedim, ncomponents);
cvs = zeros(statedim, statedim, ncomponents);
for c = 1:ncomponents
    mns(:,c) = components(c).state_distribution.mean;
    cvs(:,:,c) = reshape(components(c).state_distribution.covar, ...
        statedim, statedim);
end
logws = normaliseinlogs(logws);

end

function [means, vars] = get_colour_distributions(components)

colour_names = fieldnames(components(1).colour_distribution);

means = [];
vars = [];
component_logws = normaliseinlogs([components.log_weight]);
ncomponents = numel(components);

for colouri = 1:numel(colour_names)

    this_colour_name = colour_names{colouri};
    component_means = zeros(1, ncomponents);
    component_vars = zeros(1, 1, ncomponents);

    for compi = 1:numel(components)
        this_comp_data = components(compi).colour_distribution.(this_colour_name);
        these_logws = normaliseinlogs(...
            arrayfun(@(x)(x.component.log_weight), this_comp_data));
        
        these_means = arrayfun(@(x)(x.component.mean),...
            this_comp_data, "UniformOutput", false);
        these_means = cat(2, these_means{:});

        these_vars = arrayfun(@(x)(x.component.var),...
            this_comp_data, "UniformOutput", false);
        these_vars = cat(3, these_vars{:});

        [component_means(:,compi), component_vars(:,:,compi)] =...
            mergegaussians(these_means, these_vars, these_logws(:)');
    end

    [means.(this_colour_name), vars.(this_colour_name)] =...
        mergegaussians(component_means, component_vars, component_logws);
end

end


function logp = get_log_visibility_probs(components)

fn = fieldnames(components(1).log_vis_given_existence);
logp = [];
component_logws = normaliseinlogs([components.log_weight]);
for fi = 1:numel(fn)
    vis_logp = arrayfun(@(x)x.log_vis_given_existence.(fn{fi}), components);
    logp.(fn{fi}) = sumvectorinlogs(component_logws(:) + vis_logp(:));
end

end


function [mu, C, logw] = mergegaussians(means, covs, logweights)

% [mu, C, logw] = mergegaussians(means, covs, logweights)

logw = sumvectorinlogs(logweights);
weights = exp(logweights - logw);
mu = sum(weights.*means, 2);
Ci = mulXXtrans(means - mu) + covs;
C = sum(reshape(weights, [1 1 numel(weights)]).*Ci, 3);

end

function xx = mulXXtrans(x)

xx = permute(x, [1 3 2]).*permute(x, [3 1 2]);

end


function gaussellipse(mean, cov, c, m, ln)

% gaussellipse(mean, cov, c, m, ln)

if ~exist('c','var') || isempty(c)
    c = 'b';
end
if ~exist('ln','var') || isempty(ln)
    ln = '-';
end

hold on

npoints = 100;%20;
theta = linspace(0, 2*pi, npoints+1);
circlepoints = [sin(theta); cos(theta)];
nellipses = size(mean,2);
if size(cov,3) == 1 && nellipses > 1
    cov = repmat(cov, 1, 1, nellipses);
end

for i=1:nellipses
    [v,d] = eig(cov(:,:,i));
    vd = v*sqrt(d);
    ellipsepoints = vd*circlepoints;
    plot(ellipsepoints(1,:)+mean(1,i), ellipsepoints(2,:)+mean(2,i), ln, 'Color', c)
end
if exist('m', 'var') && ~isempty(m)
    plot(mean(1,:), mean(2,:), [m '-'], 'Color', c)
end

end
