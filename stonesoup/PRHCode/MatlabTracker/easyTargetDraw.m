function easyTargetDraw

d = load('Data/testData100targets.mat');

figure
drawtrackdata(d.trackdata, [1 3], false);
[~,~,idx] = uniquewithcounts(d.truth.mmsi);
for i = 1:numel(idx)
    plot(d.truth.coords(idx{i},1), d.truth.coords(idx{i},2), 'k-');
    plot(d.truth.coords(idx{i}(1),1), d.truth.coords(idx{i}(1),2), 'k+');
end
axis equal
xlabel('Longitude (degrees)'); ylabel('Latitude (degrees)')

figure
drawtrackdata(d.trackdata, 1, false);
% [~,~,idx] = uniquewithcounts(d.truth.mmsi);
% for i = 1:numel(idx)
%     plot(d.truth.coords(idx{i},1), d.truth.coords(idx{i},2), 'k-');
%     plot(d.truth.coords(idx{i}(1),1), d.truth.coords(idx{i}(1),2), 'k+');
% end
% axis equal
% xlabel('Longitude (degrees)'); ylabel('Latitude (degrees)')