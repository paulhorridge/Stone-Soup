function all_data = measDataFromSensorData(sensorData)

% Get measurement data from sensor data so we can put it in a JSON file or
% MongoDB database

nsensors = numel(sensorData);

all_data = [];

for s = 1:nsensors

    meas = sensorData{s}.meas;
    nmeas = numel(meas.times);
    fn = fieldnames(meas);

    data = cell(1, nmeas);
    for i = 1:nmeas
        for fi = 1:numel(fn)
            thisfn = fn{fi};
            if string(thisfn)=="mmsi"
                data{i}.(thisfn) = string(meas.(thisfn){i});
            else
                data{i}.(thisfn) = meas.(thisfn)(i,:);
            end
        end
        if ~isfield(meas, "ptime")
            data{i}.ptime = posixtime(meas.times(i));
        end
    end

    all_data.(sensorData{s}.name) = data;

end