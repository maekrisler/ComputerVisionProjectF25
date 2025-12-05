% Project 1
% Computer Vision


function main()

    % get the folder name with all image files
    folder_name = 'test_images';

    classify_images(folder_name);

% SINGLE FILE TEST
% ____________________________________________________________
    
    % image_name = "ADK_Images_Batch_B/IMG_20251004_154703800.jpg";
    % im_read = imread(image_name);
    % 
    % im_kmeans = kmeans3(im_read); % comment out if using kmeans_test
    % 
    % figure;
    % imshow(im_kmeans);
    % title("kmeans before morphology");
    % 
    % yellow_edges = isolate_yellow(im_kmeans);
    % 
    % 
    % [centers, radii] = imfindcircles(yellow_edges, ...
    %         [100 300], 'ObjectPolarity', 'bright', 'Sensitivity', 0.96);
    % 
    % if ~isempty(centers)
    %     % convert center+radii into [x y r]
    %     circles = [centers radii];
    % 
    %     % draw circles over original image
    %     out_with_circles = insertShape(im_read, ...
    %         'Circle', circles, ...
    %         'Color', 'red', 'LineWidth', 5);
    % 
    %     % output to found trail markers directory
    %     imwrite(out_with_circles, "fulloutput_test2.jpg");
    % else
    %     disp("no markers found :(");
    % 
    % end
    % 
    % figure;
    % imshow(yellow_edges);
    % colorbar;
    % title("kmeans with morphology");

% ____________________________________________________________



end


%
% Goes through all jpgs in an input image, attempts to draw circles over
% trail markers if present, then saves to 2 separate folders depending on
% if a marker was found or not.
%
function classify_images(input_dir)

    output_trail_dir = "Found Trail Markers";
    output_no_trail_dir = "No Trail Markers";

    % Create output folders if they don't exist
    if exist(output_trail_dir, 'dir')
        delete(fullfile(output_trail_dir, '*'));
    else
        mkdir(output_trail_dir);
    end

    if exist(output_no_trail_dir, 'dir')
        delete(fullfile(output_no_trail_dir, '*'));
    else
        mkdir(output_no_trail_dir);
    end


    % get all jpg images in the input folder
    image_files = dir(fullfile(input_dir, '*.jpg'));

    for i = 1:length(image_files)

        % get full path to image
        img_path = fullfile(input_dir, image_files(i).name);
        fprintf("Processing %s...\n", image_files(i).name);
        img = imread(img_path);

        % do some preprocessing on image
        im_clustered = kmeans3(img);

        out_image = isolate_yellow(im_clustered);

        % look for circles in the processed image
        [centers, radii] = imfindcircles(out_image, ...
            [100 300], 'ObjectPolarity', 'bright', 'Sensitivity', 0.96);

        % if circles found...
        if ~isempty(centers)

            fprintf("Found a trail marker in %s\n", image_files(i).name);

            % convert center+radii into [x y r]
            circles = [centers radii];

            % draw circles over original image
            out_with_circles = insertShape(img, ...
                'Circle', circles, ...
                'Color', 'red', 'LineWidth', 5);

            % output to found trail markers directory
            output_filename = fullfile(output_trail_dir, image_files(i).name);
            imwrite(out_with_circles, output_filename);
        else
            % no circles found, output original image to no trail markers
            % directory
            fprintf("No trail markers found in %s\n", image_files(i).name);
            
            output_filename = fullfile(output_no_trail_dir, image_files(i).name);
            imwrite(img, output_filename);
        end
    end
    
    fprintf("Done processing all photos.");
end



function out_image = isolate_yellow(image)

    
    im_gray = rgb2gray(image);

    im_gray = im2single(im_gray);

    im_b_star = adapthisteq(im_gray);

    % figure;
    % imshow(im_b_star);
    % colorbar;
    % title("b* before morphology");

    b_thresh = 0.45;

    % only display where b* is above certain threshold
    im_yellow = im_b_star > b_thresh;

    % figure;
    % imshow(im_yellow);
    % colorbar;
    % title("yellow after thr");

    % remove little isolated blips
    im_yellow = imopen(im_yellow, strel('disk', 5));
    
    % remove little isolated blips
    im_yellow = imclose(im_yellow, strel('disk', 8));

    % fill in areas completely surrounded by whtie
    im_yellow = imfill(im_yellow, 'holes');

    out_image = im_yellow;

end


% Uses histogram back propogration with b* channel to find most
% concentrated yellow in the given image
function yellow_map = get_hist(image)

    % read in image and convert to double
    % im_read = imread(image);
    % 
    % im_double = im2double(image);
    % im_lab = rgb2lab(im_double);
    % 
    % % get a and b star channels from LAB
    % im_b_star = im_lab(:, :, 3);
    % 
    % % normalize b* from [-128, 127] to [0, 1]
    % im_b_star = (im_b_star + 128) / 255;
    % 
    % im_b_star = adapthisteq(im_b_star);
    % 
    % b_thresh = 0.65;
    % 
    % % only display where b* is above certain threshold
    % im_yellow = im_b_star > b_thresh;

    % get rows and cols and pre allocate yellow array
    [rows, cols] = size(image);
    row_yellow_count = zeros(rows, 1);

    % for every row in the image
    for row = 1:rows
        % get the yellow row val
        row_yel = image(row, :);
        % increment hist array
        row_yellow_count(row) = sum(row_yel);
    end

    col_yellow_count = zeros(cols, 1);

    % for every col in the image
    for col = 1:cols
        % get the yellow row val
        col_yel = image(:, col);
        % increment hist array
        col_yellow_count(col) = sum(col_yel);
    end

    % normalize both hist to [0, 1]
    % multiply together to get the area where feature is most likley to be

    normalized_rows = row_yellow_count / max(row_yellow_count);
    normalized_cols = col_yellow_count / max(col_yellow_count);

    % multiply for back projection
    yellow_map = normalized_rows .* normalized_cols';

    % display for testing
    % figure;
    % subplot(1, 2, 1);
    % imagesc(yellow_map);
    % colorbar;
    % title("Histogram Back Projection of Yellow");


end


% tests to fit a perfect circle
function perfect_circle_test(image_name)

    % perfect circle test
    perfect_circle = imread(image_name);
    perfect_circle = rgb2gray(perfect_circle);
    circle_edges = edge(perfect_circle, "canny");
    out_image = isolate_yellow(perfect_circle);
    [centers, radii, metric] = imfindcircles(perfect_circle, [100 200]);

end



% test using the matlab object detection package
function object_detection(image_files)

    % init array for all read images
    final_images = cell(length(image_files), 1);

    for image = 1:length(image_files)
        % get path to image
        path = fullfile(folder_name, image_files(image).name);
        % get the file name
        im_folder_image = imread(path);

        % add image to final image list
        final_images{image} = im_folder_image;

    end

    testing foreground detector
    xy = object_detection_test(final_images);

end


function im_palletized = kmeans2(image)
    % smooth image for noise removal
    % im_smooth = noise_removal(image);
    
    % take b* channel from LAB image
    im_lab = rgb2lab(image);
    im_b_star = im_lab(:, :, 3);
    im_a_star = im_lab(:, :, 2);  % Keep for visualization later
    
    % normalize b* to [0, 1] range for distance calculations
    im_single_b = im2single(im_b_star);
    
    % get row and col dims to shape matrix
    im_rows = size(im_single_b, 1);
    im_cols = size(im_single_b, 2);
    
    % convert to 1D column vector for kmeans (only b* channel)
    im_b_shaped = reshape(im_single_b, im_rows*im_cols, 1);
    
    % set k clusters and run kmeans
    k = 5;
    [cluster_idx, cluster_center] = kmeans(im_b_shaped, k, ...
        "Distance", "sqeuclidean", 'Replicate', 3, 'MaxIter', 500);
    
    % reshape cluster indices back to 2D image
    im_clustered = reshape(cluster_idx, im_rows, im_cols);

    % reshape original image to match the flattened data used in kmeans
    im_reshaped = reshape(image, im_rows * im_cols, 3);

    % define pallet array
    rgb_pallet = zeros(k, 3);

    % create rgb pallet
    for cluster = 1:k
        % create logical mask to isolate current cluster
        cluster_mask = (cluster_idx == cluster);
        % get rgb values from pixels in the cur cluster
        cluster_pix = im_reshaped(cluster_mask, :);
        % use mean rgb val as cluster color
        rgb_pallet(cluster, :) = mean(cluster_pix, 1);
    end

    % create palletized image using rgb pallet
    im_palletized = zeros(im_rows, im_cols, 3);
    for r = 1:im_rows
        for c = 1:im_cols
            cur_cluster = im_clustered(r, c);
            im_palletized(r, c, :) = rgb_pallet(cur_cluster, :);
        end
    end

    % figure;
    % imshow(im_palletized);
    % title("kmeans clustered and palletized");

end


function im_clustered = kmeans3(image)

    im_smooth = imgaussfilt(image, 4);
    lab_he = rgb2lab(im_smooth);
    ab = lab_he(:,:,2:3);
    ab = im2single(ab);
    
    
    k = 10;
    pixel_labels = imsegkmeans(ab,k,NumAttempts=3);
    
    b_star_means = zeros(k, 1);
    for cluster = 1:k
        cluster_mask = pixel_labels == cluster;
        b_star_means(cluster) = mean(ab(cluster_mask));

        % Display each cluster in its own figure
        cur_cluster = im_smooth .* uint8(cluster_mask);
        figure;
        imshow(cur_cluster);
        title("Cluster " + string(cluster) + " - b* mean: " + string(b_star_means(cluster)));
    end

    yellow_cluster = max(b_star_means);

    best_mask = pixel_labels == 6; % HARD CODED TEST
    im_clustered = im_smooth.*uint8(best_mask);

end



function Ransac_Points_to_Fit(xy, height, width) 

    figure();
    plot( xy(1,:), xy(2,:), 'wo', 'MarkerSize', 3, 'MarkerFaceColor', 'c', 'LineWidth', 1 );

    % set axis equal to the img dimensions
    axis equal;
    % flip axis to reflect the transformed vector passed in
    set(gca, 'YDir', 'reverse'); 
    grid on;
    set(gca,'GridAlpha', 1);

    best_num_points = -Inf;
    best_circle = [];
    max_iters = 5000; % make super large for sparse circle edges

    for round = 1:max_iters
        points = 3;
        % get the indexes of x and y values from the xy array
        % points is how many rand points we want to grab
        % size gets number of cols (i.e [x,y] pairs)
        random_idxs = randperm(size(xy, 2), points);
        % extract actual xy vals using idxs from ^ 
        % rand points is now a 2*3 array of xy vals
        random_points = xy(:, random_idxs);

        % Extract x and y values from the random points
        x = random_points(1, :);
        y = random_points(2, :);  

        % --- Fit a circle through 3 points ---
        % Build linear system: [x y 1] * [A; B; C] = -(x.^2 + y.^2)
        A_mat = [x, y, ones(size(x))];
        b_vec = -(x.^2 + y.^2);
        
        params = A_mat \ b_vec;  % solve least-squares for A, B, C
        
        % Extract circle center and radius
        a = -params(1) / 2;
        b = -params(2) / 2;
        r = sqrt((params(1)^2 + params(2)^2)/4 - params(3));

        % check to make sure the best circle found is within img bounds
        % if (a - r) < 1 || (a + r) > width || ...
        %     (b - r) < 1 || (b + r) > height
        %     continue;
        % 
        % end
         
        % compute distances of all points to the circle
        x_all = xy(1, :);
        y_all = xy(2, :);
        distances = abs(sqrt((x_all - a).^2 + (y_all - b).^2) - r);
        
        % define inliers: points whose distance to circle boundary < threshold
        threshold = r * 0.02; % make the thr adaptive for different sized markers
        inliers = distances < threshold;
        num_inliers = sum(inliers);

        % avoid fitting a circle to random noise by setting a 
        % min number of points to lie within fit
        min_inliers = 40;
        
        % Update best fit if current model has more inliers
        if num_inliers > best_num_points && num_inliers > min_inliers
            best_num_points = num_inliers;
            best_circle = [a, b, r];
        end

    end

    % ERROR CHECK 
    if isempty(best_circle)
        warning('No valid circle found with current constraints');
        return;
    end


    % plot the parabola on the gaph of points
    hold on;
    % a is the center for x
    a = best_circle(1);
    % b is the center for y
    b = best_circle(2);
    % r is the radius 
    r = best_circle(3);


    % use parametric equ to plot the circle
    % 1000 = the number of iterations to fit a circle
    theta = linspace(0, 2*pi, 1000);
    x_circle = a + r * cos(theta);
    y_circle = b + r * sin(theta);

    % plot
    plot(x_circle, y_circle, 'w-', 'LineWidth', 3);
    % mark center for testing
    plot(a, b, 'w+', 'MarkerSize', 20, 'LineWidth', 3);



end

function xy = object_detection_test(folder_name)
    % get all .jpg files from folder (for object_detection_test)
    image_files = dir(fullfile(folder_name, '*.jpg'));

    % grab foreground image
    bim_name = "IMPORTANT_TEST.jpg";
    foregroundImage = imread(bim_name);
    fore_double = im2double(foregroundImage);
    imbstar_f = rgb2lab(fore_double);
    imbstar_f = imbstar_f(:, :, 3);

    % Create a vision.ForegroundDetector object
    % The NumTrainingFrames simulates background learning
    foregroundDetector = vision.ForegroundDetector('NumTrainingFrames', ...
        length(final_images), 'InitialVariance', 30*30);

    targetSize = size(final_images{1}, [1 2]);

    % for every image in the dir of training background images:
    for image = 1:length(final_images)
        cur_image = imresize(final_images{image}, targetSize);
        im_double = im2double(cur_image);
        % clean that image!
        im_lab = rgb2lab(im_double);
        % b* is the best channel to isolate yellow
        im_bstar = im_lab(:, :, 3);

        % pass the detection algorithm created to matlab system object
        % to be run, give the current image as argument for training
        step(foregroundDetector, im_bstar);

    end

    % Now detect the foreground in the new image
    foregroundMask = step(foregroundDetector, imbstar_f);

    % use morphology to close the circles
    foregroundMask = imclose(foregroundMask, strel('disk', 20));
    

    % Display results
    figure;
    subplot(1,2,1); imshow(foregroundImage); title('Input Image');
    subplot(1,2,2); imshow(foregroundMask); title('Detected Foreground');

    [y, x] = find(foregroundMask);

    x = double(x);
    y = double(y);

    xy = [x';y'];


end

