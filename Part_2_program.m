%Rounak Sen, PID: A59020344
%ECE 271A Quiz 2 Solution

%load data
data = load('TrainingSamplesDCT_8_new.mat');
bg = TrainsampleDCT_BG;
fg = TrainsampleDCT_FG;

%prior probabilities using Maximum Likelihood Estimation
bg_prior = size(TrainsampleDCT_BG, 1) / (size(TrainsampleDCT_BG, 1) + size(TrainsampleDCT_FG, 1));
fg_prior = 1 - bg_prior;

disp(['Prior probability for foreground is: ', num2str(fg_prior)]);
disp(['Prior probability for background is: ', num2str(bg_prior)]);

%getting mean and covariance for each feature

mean_fg = zeros(1, size(TrainsampleDCT_FG, 2));
mean_bg = zeros(1, size(TrainsampleDCT_BG, 2));

%Applying MLE to get mean vector for FG
for col = 1 : size(TrainsampleDCT_FG, 2)
    sum_col = 0;
    for row = 1 : size(TrainsampleDCT_FG, 1)
        sum_col = sum_col + fg(row,col);
    end
    mean_fg(col) = sum_col / size(TrainsampleDCT_FG, 1);
end
%Getting covariance matrix for fg by using Matrix in-built method
cov_fg = cov(fg);

%Applying MLE to mean vector for BG
for col = 1 : size(TrainsampleDCT_BG, 2)
    sum_col = 0;
    for row = 1 : size(TrainsampleDCT_BG, 1)
        sum_col = sum_col + bg(row,col);
    end
    mean_bg(col) = sum_col / size(TrainsampleDCT_BG, 1);
end
%Getting covariance matrix for bg by using Matrix in-built method
cov_bg = cov(bg);

close all; %Close all open figure before plotting new ones
%%
%Getting class conditional probability densities for FG, BG using
%Gaussian approximation
for col = 1 : size(TrainsampleDCT_FG, 2)
    %FG
    x_fg_sorted = sort(fg(:, col));
    y_fg = zeros(1, size(TrainsampleDCT_FG, 1));
    for row = 1 : size(TrainsampleDCT_FG, 1)
        y_fg(row) = exp((x_fg_sorted(row) - mean_fg(col)) * (x_fg_sorted(row) - mean_fg(col)) / (-2 * cov_fg(col, col))) / sqrt(2 * pi * cov_fg(col, col));
    end
    %BG
    x_bg_sorted = sort(bg(:, col));
    y_bg = zeros(1, size(TrainsampleDCT_BG, 1));
    for row = 1 : size(TrainsampleDCT_BG, 1)
        y_bg(row) = exp((x_bg_sorted(row) - mean_bg(col)) * (x_bg_sorted(row) - mean_bg(col)) / (-2 * cov_bg(col, col))) / sqrt(2 * pi * cov_bg(col, col));
    end

    %Plotting fg and bg probability densities
    subplot(11, 6, col);
    %figure(col);
    plot(x_fg_sorted, y_fg, x_bg_sorted, y_bg);
    title(col);
    pos = get(gcf, 'Position');
    set(gcf, 'Position',pos+[-100 -250 100 250])
end

snapnow

%8 best features by examing the sub-plots
best_features = [1 25 27 32 33 40 41 45];

%8 worst features by examing the sub-plots
worst_features = [2 3 4 5 59 62 63 64];

%Creating classification mask for grass and cheetah

%read cheetah test image
cheetah_img = im2double(imread('cheetah.bmp'));

%padded image matrix to account for right most and bottom most boundaries
cheetah_img_padded = zeros(size(cheetah_img, 1) + 7, size(cheetah_img, 2) + 7);

for row = 1 : size(cheetah_img, 1)
    for col = 1 : size(cheetah_img, 2)
        cheetah_img_padded(row, col) = cheetah_img(row, col);
    end
end

cheetah_pred = zeros(size(cheetah_img, 1), size(cheetah_img, 2));
zig_zag_mat = readmatrix('Zig-Zag Pattern.txt');

for row = 1 : size(cheetah_img, 1)

    for col = 1 : size(cheetah_img, 2)

        cheetah_img_block = cheetah_img_padded(row : row  + 7, col : col + 7);% divide image into blocks
        freq_coeff = dct2(cheetah_img_block);
        X = zeros(1, 64); % 1x64 vector
        
        for r = 1 : size(zig_zag_mat, 1)

            for c = 1 : size(zig_zag_mat, 2)

                X(1 + zig_zag_mat(r,c)) = freq_coeff(r,c); 
        
            end
  
        end

        %Getting class conditional probabilities for multivariate Gaussian
        %case for both cheetah and grass
        temp_C = X - mean_fg;
        temp_G = X - mean_bg;
        P_X_C = exp(-0.5 * temp_C * inv(cov_fg) * transpose(temp_C)) / sqrt(power(2 * pi, 64) * abs(det(cov_fg)));
        P_X_G = exp(-0.5 * temp_G * inv(cov_bg) * transpose(temp_G)) / sqrt(power(2 * pi, 64) * abs(det(cov_bg)));

        %Bayes Decision Rule to predict pixel value
        if P_X_C * fg_prior > P_X_G * bg_prior
            cheetah_pred(row, col) = 1;
        else
            cheetah_pred(row, col) = 0;
        end

    end

end

%%
%display image
figure;
cheetah_pred_image = imagesc(cheetah_pred);
colormap(gray(255))
title('All 64 features')

snapnow

%Getting cheetah prediction image for top 8 features

cheetah_pred_best = zeros(size(cheetah_img, 1), size(cheetah_img, 2));

for row = 1 : size(cheetah_img, 1)

    for col = 1 : size(cheetah_img, 2)

        cheetah_img_block = cheetah_img_padded(row : row  + 7, col : col + 7);% divide image into blocks
        freq_coeff = dct2(cheetah_img_block);
        X = zeros(1, 64); % 1x64 vector
        
        for r = 1 : size(zig_zag_mat, 1)

            for c = 1 : size(zig_zag_mat, 2)

                X(1 + zig_zag_mat(r,c)) = freq_coeff(r,c); 
        
            end
  
        end

        %Getting X, mean and covariance matrixes for cheetah and grass by
        %considering only the 8 best features

        X_best = zeros(1, 8);%1x8 vector for 8 best features
        mean_fg_best = zeros(1, 8);
        mean_bg_best = zeros(1, 8);
        cov_fg_best = zeros(8, 8);
        cov_bg_best = zeros(8, 8);

        for r = 1 : size(best_features, 2)
  
            X_best(r) = X(best_features(r));
            mean_fg_best(r) = mean_fg(best_features(r));
            mean_bg_best(r) = mean_bg(best_features(r));

            for c = 1 : size(best_features, 2)

                cov_fg_best(r, c) = cov_fg(best_features(r), best_features(c));
                cov_bg_best(r, c) = cov_bg(best_features(r), best_features(c));

            end

        end

        %Getting class conditional probabilities for multivariate Gaussian
        %case for both cheetah and grass
        temp_C = X_best - mean_fg_best; 
        temp_G = X_best - mean_bg_best;
        P_X_C = exp(-0.5 * temp_C * inv(cov_fg_best) * transpose(temp_C)) / sqrt(power(2 * pi, 8) * abs(det(cov_fg_best)));
        P_X_G = exp(-0.5 * temp_G * inv(cov_bg_best) * transpose(temp_G)) / sqrt(power(2 * pi, 8) * abs(det(cov_bg_best)));

        %Bayes Decision Rule to predict pixel value
        if P_X_C * fg_prior > P_X_G * bg_prior
            cheetah_pred_best(row, col) = 1;
        else
            cheetah_pred_best(row, col) = 0;
        end

    end

end

%%
%display image
figure;
cheetah_pred_image_best = imagesc(cheetah_pred_best);
colormap(gray(255))
title('8 best features')

snapnow

%error probability calculation for all 64 features taken into account
cheetah_img_real = im2double(imread('cheetah_mask.bmp'));
error_prob = 0;

for r = 1 : size(cheetah_img_real, 1)
    for c = 1 : size(cheetah_img_real, 2)
        error_prob = error_prob + (abs(cheetah_img_real(r,c) - cheetah_pred(r,c))) / (size(cheetah_img_real, 1) * size(cheetah_img_real, 2));
    end
end

disp(['The error probability with all 64 features is: ', num2str(error_prob)]);

%error probability calculation for best 8 features taken into account
cheetah_img_real = im2double(imread('cheetah_mask.bmp'));
error_prob_best = 0;

for r = 1 : size(cheetah_img_real, 1)
    for c = 1 : size(cheetah_img_real, 2)
        error_prob_best = error_prob_best + (abs(cheetah_img_real(r,c) - cheetah_pred_best(r,c))) / (size(cheetah_img_real, 1) * size(cheetah_img_real, 2));
    end
end

disp(['The error probability with top 8 features is: ', num2str(error_prob_best)]);

snapnow



