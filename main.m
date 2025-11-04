% Project 1
% Computer Vision


function main()

    % image = "IMPORTANT_TEST.jpg";
    % test_im(image);

    % get the folder name with all image files
    folder_name = 'ADK_Images_Batch_D';

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

    object_detection_test(final_images);

end


% this does a good job of getting the edges in the background and on 
% the trail marker
% I'm thinking we can use these edges as input for ransac 
% and fit it to a circle?
function object_detection_test(final_images)
    % TODO: test vision.ForegroundDetector for finding object

    % grab foreground image
    bim_name = "IMPORTANT_TEST.jpg";
    foregroundImage = imread(bim_name);

    % Create a vision.ForegroundDetector object
    % The NumTrainingFrames simulates background learning
    foregroundDetector = vision.ForegroundDetector('NumTrainingFrames', ...
        length(final_images), 'InitialVariance', 30*30);

    % for every image in the dir of training background images:
    for image = 1:length(final_images)
        % pass the detection algorithm created to matlab system object
        % to be run, give the current image as argument for training
        step(foregroundDetector, final_images{image});

    end

    % Now detect the foreground in the new image
    foregroundMask = step(foregroundDetector, foregroundImage);
    

    % Display results
    figure;
    subplot(1,2,1); imshow(foregroundImage); title('Input Image');
    subplot(1,2,2); imshow(foregroundMask); title('Detected Foreground');


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