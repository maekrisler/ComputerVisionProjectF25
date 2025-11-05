% Project 1
% Computer Vision


function main()

    % image = "IMPORTANT_TEST.jpg";
    % test_im(image);

    % get the folder name with all image files
    folder_name = 'test_images';

    % get all .jpg files from folder
    image_files = dir(fullfile(folder_name, '*.jpg'));

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

    xy = object_detection_test(final_images);

    Ransac_Points_to_Fit(xy);

end


% this does a good job of getting the edges in the background and on 
% the trail marker
% I'm thinking we can use these edges as input for ransac 
% and fit it to a circle?
function xy = object_detection_test(final_images)
    % TODO: test vision.ForegroundDetector for finding object

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

    edges = edge(foregroundMask, 'Canny');

    [y, x] = find(edges);

    x = double(x);
    y = double(y);

    xy = [x';y'];

end



function Ransac_Points_to_Fit(xy) 

    figure();
    plot( xy(1,:), xy(2,:), 'ko', 'MarkerSize', 15, 'MarkerFaceColor', 'c', 'LineWidth', 3 );

    % set axis equal to the img dimensions
    axis equal;
    % flip axis to reflect the transformed vector passed in
    set(gca, 'YDir', 'reverse'); 
    grid on;
    set(gca,'GridAlpha', 1);

    best_num_points = -Inf;
    max_iters = 1000;

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
       

        % get three points to fit a circle
        
        % TODO --> I think the math for this is wrong
        % I based my solution off the following example
        % https://computervision-projects.firebaseapp.com/assignment2 

        % Build linear system: [x y 1] * [A; B; C] = -(x.^2 + y.^2)
        A_mat = [x, y, ones(size(x))];
        b_vec = -(x.^2 + y.^2);
        
        % solve least-squares for A, B, C
        params = A_mat \ b_vec;
        
        % Extract circle center and radius
        a = -params(1) / 2;
        b = -params(2) / 2;
        r = sqrt((params(1)^2 + params(2)^2)/4 - params(3));
        
        % calculate dist of all points to the circle 
        x_all = xy(1, :);
        y_all = xy(2, :);
        % [x y 1] * [A; B; C] = -(x.^2 + y.^2)
        distances = abs(sqrt((x_all - a).^2 + (y_all - b).^2) - r);
        
        % get all inliers (points whose distance to circle boundary <
        % threshold)
        threshold = 0.1;
        inliers = distances < threshold;
        num_inliers = sum(inliers);
        
        % Update best fit if current model has more inliers
        if num_inliers > best_num_points
            best_num_points = num_inliers;
            best_circle = [a, b, r];
        end

    end

    % plot value
    figure; hold on;
    % reverse Y for correct plotting ? 
    set(gca, 'YDir', 'reverse');
    % plot xy vals over image
    plot(xy(1,:), xy(2,:), 'k.');
    viscircles([best_circle(1), best_circle(2)], best_circle(3), 'Color', 'r');
    title('RANSAC Circle Fit');
    axis equal;

end



function test_im(image)

    testing = imread(image);
    im_blue = testing(:, :, 3);

    im_smooth = imgaussfilt(im_blue,8);

    im_contrast = im_smooth.^2;

    im_sobel = edge(im_contrast, 'log');
   
    figure();
    imshow(im_contrast);
    figure();
    imshow(im_sobel);

    imfindcircles(im_contrast, [100 700], ...
        ObjectPolarity="dark", ...
        Sensitivity=0.92, ...
        EdgeThreshold=0.1)



end