function m = sigma_delta_UD_Counter_col_not_selected_skipped(x,ref,Zero_ind)
        % x is the image
        %ref is the reference signal for comparator
        % Zero_ind is the vector contaiting the indices of pixels made to
        % zero which translates to not selected pixels and hence sigma
        % delta must not function in this case.
        
        % this function will be in the loop for all columns of image
        if ndims(x)>2
            x = rgb2gray(x);
        end
        x = double(x);
        nc = size(x,2);
        nr = size(x,1);
        col = zeros(nc,1);
        f0=255;
        aref = ref;%127; % comparator reference voltage
        for j=1:nc
            x1 = x(:,j);
           
            a0 = 0;
            f=0;
            count = zeros(nr,1);
            %acc = zeros(nr,1);
            %comp = zeros(nr,1);
            Dig_out = zeros(nr,1);
            k=0; 
            for i=1:nr
                if ~ismember(i,Zero_ind) % 
                    a = x1(i) - f;
                    a1 = a+a0; % ----> accumulator's present output
                    a0 = a1;
                    % 1-bit quantizer which gives +1 (255 for images) and -1 (-255) as the output

                    %acc(i) = a1;
                    c = a1-aref; % output of the comparator
                    %comp(i) = c;
                    if c>=0
                        f = f0; %=255 if images
                        count(i) = 1;
                        Dig_out(i) = k +1;
                        k = Dig_out(i);
                    else
                        f=-f0; %=-255 if images
                        count(i) = 0;
                        Dig_out(i) = k-1;
                        k = Dig_out(i);
                    end
                end
            end
            col(j) = max(Dig_out);
        end
        %m = sum(col);
        m = col;
end