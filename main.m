% Project 1
% Computer Vision
%
% Authors: Mae Krisler & Hannah Clapp
%


function main()

    % get the folder name with all image files
    folder_name = 'test_images';

    classify_images(folder_name);

end


%
% Goes through all jpgs in an input directory, attempts to draw circles over
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
        % im_clustered = kmeans3(img);
        % kmeans didnt work well so sad :(
        % out_image = isolate_yellow(im_clustered);

        % preprocess image w thresholding and morphology
        out_image = isolate_yellow(img);


        % look for circles in the processed image
        [centers, radii] = imfindcircles(out_image, ...
            [70 300], 'ObjectPolarity', 'bright', 'Sensitivity', 0.94);

        % if circles found...
        if ~isempty(centers)

            fprintf("Found a trail marker in %s\n", image_files(i).name);

            % convert center + radii into [x y r]
            circles = [centers radii];

            % draw circles in red over original image
            out_with_circles = insertShape(img, ...
                'Circle', circles, ...
                'Color', 'red', 'LineWidth', 10);

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

    % smooth image
    im_smooth = imgaussfilt(image, 3);

    % convert to LAB color space
    lab = rgb2lab(im_smooth);
    b_star = lab(:,:,3);

    % min/max normalization to [0,1]
    b_star_norm = (b_star - min(b_star(:))) / (max(b_star(:)) - min(b_star(:)));

    % threshold from trial and error
    b_thresh = 0.53;

    % only display where b* is above certain threshold
    im_yellow = b_star_norm > b_thresh;

    % remove little isolated blips
    im_yellow = imopen(im_yellow, strel('disk', 5));
    
    % % remove little isolated black blips
    % im_yellow = imclose(im_yellow, strel('disk', 8));

    % fill in areas completely surrounded by whtie
    im_yellow = imfill(im_yellow, 'holes');

    out_image = im_yellow;

end

%
% We tried implementing kmeans to get yellow cluster.
% Unfortunately, simple thresholding just worked better than this, so it 
% is not being used.
%
function im_clustered = kmeans3(image)

    im_smooth = imgaussfilt(image, 4);
    lab = rgb2lab(im_smooth);

    L = lab(:,:,1);
    a_star = lab(:,:,2);
    b_star = lab(:,:,3);

    % coordinates
    dst_wt = 0.1;
    dims = size(image);
    [xs, ys] = meshgrid(1:dims(2), 1:dims(1));

    % build attributes image for imsegkmeans: x, y, L, a*, and 2x b*
    attributes = cat(3, ...
        xs * dst_wt, ...
        ys * dst_wt, ...
        L, ...
        a_star, ...
        2 * b_star ...
    );

    attributes = im2single(attributes);

    k = 10;
    pixel_labels = imsegkmeans(attributes, k, NumAttempts=3);

    % compute b* mean per cluster
    b_star_means = zeros(k,1);
    for cluster = 1:k
        mask = pixel_labels == cluster;
        b_channel = attributes(:,:,5);        % get 5th feature (2x b*)
        b_star_means(cluster) = mean(b_channel(mask));  % then index with mask
    end

    % pick the most yellow cluster (highest b*)
    [~, yellow_cluster] = max(b_star_means);

    % build output mask
    best_mask = pixel_labels == yellow_cluster;

    % return isolated yellow cluster
    im_clustered = im_smooth .* uint8(best_mask);

end
