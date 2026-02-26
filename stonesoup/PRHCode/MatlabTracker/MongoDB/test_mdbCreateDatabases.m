function test_mdbCreateDatabases

rng(1, 'twister');

% Shuffle measurements out of time order
load mdb_testData sensorData
% for i = 1:numel(sensorData)
%     thismeas = sensorData{i}.meas;
%     idx = randperm(numel(thismeas.times));
%     sensorData{i}.meas = selectrows(thismeas, idx);
% end
% save mdb_testData sensorData

% Open MongoDB connection
server = "localhost";
port = 27017;
dbname = "test_measurement_db";
conn = mongoc(server, port, dbname);
assert(isopen(conn));

% Initialise track collection
collection_name = "measdata";
if any(conn.CollectionNames == collection_name)
    remove(conn, collection_name, "{}")
else
    createCollection(conn, collection_name);
end

for k = 1:numel(sensorData)
    thistype = sensorData{k}.name;
    thesemeas = sensorData{k}.meas;
    nmeas = numel(thesemeas.times);
    for m = 1:nmeas
        newdata = struct('type', thistype, 'datenum', datenum(thesemeas.times(m)));
        fn = fieldnames(thesemeas);
        for f = 1:numel(fn)
            thisdata = thesemeas.(fn{f})(m,:);
            for fi = 1:size(thisdata, 2)
                thisfn = fn{f};
                if size(thisdata, 2) > 1
                    thisfn = [thisfn '_' num2str(fi)];
                end
                if iscell(thisdata(fi))
                    newdata.(thisfn) = thisdata{fi};
                else
                    newdata.(thisfn) = thisdata(fi);
                end
            end
        end
        n = insert(conn, collection_name, newdata);
    end
end

end
