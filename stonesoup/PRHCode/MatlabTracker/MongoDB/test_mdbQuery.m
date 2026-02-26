function test_mdbQuery

% Open MongoDB connection
server = "localhost";
port = 27017;
dbname = "test_measurement_db";
conn = mongoc(server, port, dbname);
assert(isopen(conn));
collection_name = "measdata";

inittime = datenum('10-Aug-2017 00:00:00');
finaltime = inittime + 1;
batchtimedelta = 10 / (24*3600);

prevtime = inittime;
numdocs = 0;

while prevtime <= finaltime

    batchmintime = prevtime;
    batchmaxtime = batchmintime + batchtimedelta;
    mongoquery = "{""datenum"": {""$gte"": " + sprintf("%f", batchmintime) +...
        ", ""$lt"": " + sprintf("%f", batchmaxtime) + "}}";
    d = mystruct2cell(find(conn, collection_name, "Query", mongoquery,...
        "Sort", "{""datenum"":1.0}"));
    numdocs = numdocs + numel(d);
    for i = 1:numel(d)
        disp(d{i}.times)
        disp(d{i}.type);
    end

    prevtime = batchmaxtime;

end

numdocs

end


function d2 = mystruct2cell(d)

if iscell(d)
    d2 = d;
else
    d2 = cell(1, numel(d));
    for i = 1:numel(d)
        d2{i} = d(i);
    end
end

end
