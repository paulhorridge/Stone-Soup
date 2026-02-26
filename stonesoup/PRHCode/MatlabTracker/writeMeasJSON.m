function writeMeasJSON(sensorData, outfilename)

% nsensors = numel(sensorData);
% 
% all_data = [];
% 
% for s = 1:nsensors
% 
%     meas = sensorData{s}.meas;
%     nmeas = numel(meas.times);
%     fn = fieldnames(meas);
% 
%     data = cell(1, nmeas);
%     for i = 1:nmeas
%         for fi = 1:numel(fn)
%             thisfn = fn{fi};
%             if string(thisfn)=="mmsi"
%                 data{i}.(thisfn) = string(meas.(thisfn){i});
%             else
%                 data{i}.(thisfn) = meas.(thisfn)(i,:);
%             end
%         end
%     end
% 
%     all_data.(sensorData{s}.name) = data;
% 
% end
all_data = measDataFromSensorData(sensorData);
fid = fopen(outfilename, "w");
fprintf(fid, jsonencode(all_data, PrettyPrint=true));
fclose(fid);

end
