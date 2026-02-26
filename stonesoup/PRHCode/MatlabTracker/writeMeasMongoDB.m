function writeMeasMongoDB(sensorData)

all_data = measDataFromSensorData(sensorData);

% Open MongoDB connection
server = "localhost";
port = 27017;
dbname = "test_database";
conn = mongoc(server, port, dbname);
assert(isopen(conn));

fn = fieldnames(all_data);
for fi = 1:numel(fn)

    this_fieldname = fn{fi};
    this_collection_name = "pythontest_" + string(this_fieldname);
    if any(conn.CollectionNames == this_collection_name)
        remove(conn, this_collection_name, "{}")
    else
        createCollection(conn, this_collection_name);
    end
    n = insert(conn, this_collection_name, all_data.(this_fieldname));
end

end
