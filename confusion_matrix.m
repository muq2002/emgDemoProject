function out_data = confusion_matrix(actual_data, predicted_data, my_classes)
    out_data = zeros(length(my_classes));
    
    for row = 1:length(actual_data)
        % For First Class
        if actual_data(row,1) == str2num(predicted_data{row,1}) ...
            && (str2num(predicted_data{row,1}) == 1)
            out_data(1,1) = out_data(1,1) + 1;
        end
        if (actual_data(row,1) ~= str2num(predicted_data{row,1})) ...
         && (actual_data(row,1) == 1) && (str2num(predicted_data{row,1}) == 2)
            out_data(1,2) = out_data(1,2) + 1;
        end
        if (actual_data(row,1) ~= str2num(predicted_data{row,1})) ...
         && (actual_data(row,1) == 1) && (str2num(predicted_data{row,1}) == 3)
            out_data(1,3) = out_data(1,3) + 1;
        end
        
        % For Second Class
        if (actual_data(row,1) == str2num(predicted_data{row,1})) ...
            && (str2num(predicted_data{row,1}) == 2)
            out_data(2,2) = out_data(2,2) + 1;
        end
        if (actual_data(row,1) ~= str2num(predicted_data{row,1})) ...
            && (actual_data(row,1) == 2) && (str2num(predicted_data{row,1}) == 1)
            out_data(2,1) = out_data(2,1) + 1;
        end
        if (actual_data(row,1) ~= str2num(predicted_data{row,1})) ...
            && (actual_data(row,1) == 2) && (str2num(predicted_data{row,1}) == 3)
            out_data(2,3) = out_data(2,3) + 1;
        end
        
        % For Thrid Class
        if actual_data(row,1) == str2num(predicted_data{row,1}) ...
            && (str2num(predicted_data{row,1}) == 3)
            out_data(3,3) = out_data(3,3) + 1;
        end
        if (actual_data(row,1) ~= str2num(predicted_data{row,1})) ...
            && (actual_data(row,1) == 3) && (str2num(predicted_data{row,1}) == 1)
            out_data(3,1) = out_data(3,1) + 1;
        end
        if (actual_data(row,1) ~= str2num(predicted_data{row,1})) ...
            && (actual_data(row,1) == 3) && (str2num(predicted_data{row,1}) == 2)
            out_data(3,2) = out_data(3,2) + 1;
        end
    end
end
