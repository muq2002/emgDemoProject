function segments = segmentationSignal(signal,samples_signal,samples_window, number_of_windows)
    segments = mat2cell(signal, [ samples_window * ones(1, number_of_windows) samples_signal - number_of_windows * samples_window], 2);
end 
