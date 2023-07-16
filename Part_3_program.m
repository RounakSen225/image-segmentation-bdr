%Rounak Sen, PID: A59020344
%ECE 271A Quiz 3 Solution

%load data
data = load('TrainingSamplesDCT_subsets_8.mat');

%First we compute Covariance matrix for this Dataset

%Here D1_BG, D1_FG correspond to the different datasets loaded from sample i
%where i = 1,2,3,4
D1_BG = data.D4_BG; 
D1_FG = data.D4_FG;
bg1 = D1_BG;
fg1 = D1_FG;
cov_bg1 = cov(bg1);
cov_fg1 = cov(fg1);

%prior probabilities using Maximum Likelihood Estimation
bg_prior1 = size(D1_BG, 1) / (size(D1_BG, 1) + size(D1_FG, 1));
fg_prior1 = 1 - bg_prior1;

%Reading data for each strategy
prior_data = load('Prior_2.mat');
mu0_FG = prior_data.mu0_FG;
mu0_BG = prior_data.mu0_BG;
w0 = prior_data.W0;

%read cheetah test image
cheetah_img = im2double(imread('cheetah.bmp'));

%read masked cheetah image
cheetah_img_real = im2double(imread('cheetah_mask.bmp'));

%padded image matrix to account for right most and bottom most boundaries
cheetah_img_padded = zeros(size(cheetah_img, 1) + 7, size(cheetah_img, 2) + 7);

for row = 1 : size(cheetah_img, 1)
    for col = 1 : size(cheetah_img, 2)
        cheetah_img_padded(row, col) = cheetah_img(row, col);
    end
end


zig_zag_mat = readmatrix('Zig-Zag Pattern.txt');

error_prob = zeros(1, size(alpha, 2));
error_prob_map = zeros(1, size(alpha, 2));

cheetah_pred_ml1 = zeros(size(cheetah_img, 1), size(cheetah_img, 2));

close all;
%iterate over alpha
for a = 1 : size(alpha, 2)
    cheetah_pred1 = zeros(size(cheetah_img, 1), size(cheetah_img, 2));
    cheetah_pred_map = zeros(size(cheetah_img, 1), size(cheetah_img, 2));
    %get prior covariance matrix for fg
    cov_fg0_1 = zeros(size(W0, 2), size(W0, 2));
    for r = 1 : size(W0, 2)
        cov_fg0_1(r, r) = alpha(1, a) * w0(1, r);
    end

    %get prior covariance matrix for bg
    cov_bg0_1 = zeros(size(W0, 2), size(W0, 2));
    for r = 1 : size(W0, 2)
        cov_bg0_1(r, r) = alpha(1, a) * w0(1, r);
    end
    
    %get ML estimate for means for fg and bg
    mean_fg1_ml = zeros(1, size(D1_FG, 2));
    mean_bg1_ml = zeros(1, size(D1_BG, 2));
    for col = 1 : size(D1_FG, 2)
        sum_col = 0;
        for row = 1 : size(D1_FG, 1)
            sum_col = sum_col + fg1(row,col);
        end
        mean_fg1_ml(col) = sum_col / size(D1_FG, 1);
    end

    for col = 1 : size(D1_BG, 2)
        sum_col = 0;
        for row = 1 : size(D1_BG, 1)
            sum_col = sum_col + bg1(row,col);
        end
        mean_bg1_ml(col) = sum_col / size(D1_BG, 1);
    end

    %we get optimal mean and covariance for dataset 1 using the formulas
    %derived for Bayesian case (predictive distribution) in DHS
    mu_fg_n1 = cov_fg0_1 * inv(cov_fg0_1 + (1 / size(D1_FG, 1)) * cov_fg1) * transpose(mean_fg1_ml) + (1 / size(D1_FG, 1)) * cov_fg1 * inv(cov_fg0_1 + (1 / size(D1_FG, 1)) * cov_fg1) * transpose(mu0_FG);
    mu_bg_n1 = cov_bg0_1 * inv(cov_bg0_1 + cov_bg1 / size(D1_BG, 1)) * transpose(mean_bg1_ml) + 1 / size(D1_BG, 1) * cov_bg1 * inv(cov_bg0_1 + 1 / size(D1_BG, 1) * cov_bg1) * transpose(mu0_BG);
    cov_fg_n1 = cov_fg0_1 * inv(cov_fg0_1 + 1 / size(D1_FG, 1) * cov_fg1) * 1 / size(D1_FG, 1) * cov_fg1;
    cov_bg_n1 = cov_bg0_1 * inv(cov_bg0_1 + 1 / size(D1_BG, 1) * cov_bg1) * 1 / size(D1_BG, 1) * cov_bg1;
    
    %Dividind cheetah image into 8x8 and traversing through it
    for row = 1 : size(cheetah_img, 1)

        for col = 1 : size(cheetah_img, 2)

            cheetah_img_block = cheetah_img_padded(row : row  + 7, col : col + 7);% divide image into blocks
            freq_coeff = dct2(cheetah_img_block);
            X = zeros(64, 1); %1x64 vector
        
            for r = 1 : size(zig_zag_mat, 1)

                for c = 1 : size(zig_zag_mat, 2)

                    X(1 + zig_zag_mat(r,c), 1) = freq_coeff(r,c); 
        
                end
  
            end

            %Getting class conditional probabilities for multivariate Gaussian
            %case for both cheetah and grass for predictive distribution
            temp_C = X - mu_fg_n1;
            temp_G = X - mu_bg_n1;
            cov_pred_fg1 = cov_fg1 + cov_fg_n1;
            cov_pred_bg1 = cov_bg1 + cov_bg_n1;
            P_X_C = exp(-0.5 * transpose(temp_C) * inv(cov_pred_fg1) * temp_C) / sqrt(power(2 * pi, 64) * abs(det(cov_pred_fg1)));
            P_X_G = exp(-0.5 * transpose(temp_G) * inv(cov_pred_bg1) * temp_G) / sqrt(power(2 * pi, 64) * abs(det(cov_pred_bg1)));

            %Bayes Decision Rule to predict pixel value
            if P_X_C * fg_prior1 > P_X_G * bg_prior1
                cheetah_pred1(row, col) = 1;
            else
                cheetah_pred1(row, col) = 0;
            end

            %Getting class conditional probabilities for multivariate Gaussian
            %case for both cheetah and grass for MAP estimate
            temp_C_map = X - mu_fg_n1;
            temp_G_map = X - mu_bg_n1;
            P_X_C_map = exp(-0.5 * transpose(temp_C_map) * inv(cov_fg1) * temp_C_map) / sqrt(power(2 * pi, 64) * abs(det(cov_fg1)));
            P_X_G_map = exp(-0.5 * transpose(temp_G_map) * inv(cov_bg1) * temp_G_map) / sqrt(power(2 * pi, 64) * abs(det(cov_bg1)));

            %Bayes Decision Rule to predict pixel value
            if P_X_C_map * fg_prior1 > P_X_G_map * bg_prior1
                cheetah_pred_map(row, col) = 1;
            else
                cheetah_pred_map(row, col) = 0;
            end

            %Getting cheetah prediction image for ML case (we need to
            %compute this only once)
            if a == 1
                temp_C_ml = X - transpose(mean_fg1_ml);
                temp_G_ml = X - transpose(mean_bg1_ml);
                P_X_C_ml = exp(-0.5 * transpose(temp_C_ml) * inv(cov_fg1) * temp_C_ml) / sqrt(power(2 * pi, 64) * abs(det(cov_fg1)));
                P_X_G_ml = exp(-0.5 * transpose(temp_G_ml) * inv(cov_bg1) * temp_G_ml) / sqrt(power(2 * pi, 64) * abs(det(cov_bg1)));
           
                %Bayes Decision Rule to predict pixel value
                if P_X_C_ml * fg_prior1 > P_X_G_ml * bg_prior1
                    cheetah_pred_ml1(row, col) = 1;
                else
                    cheetah_pred_ml1(row, col) = 0;
                end
            end

       end

    end

    %%
    %display image
    figure;
    cheetah_pred_image1 = imagesc(cheetah_pred1);
    colormap(gray(255));
    title('Cheetah image from PD for \alpha = ', num2str(alpha(1, a)));

    snapnow

    %Error calculation for predictive equation
    for r = 1 : size(cheetah_img_real, 1)
        for c = 1 : size(cheetah_img_real, 2)
            error_prob(1,a) = error_prob(1,a) + (abs(cheetah_img_real(r,c) - cheetah_pred1(r,c))) / (size(cheetah_img_real, 1) * size(cheetah_img_real, 2));
        end
    end

    %Error calculation for MAP estimate case
    for r = 1 : size(cheetah_img_real, 1)
        for c = 1 : size(cheetah_img_real, 2)
            error_prob_map(1,a) = error_prob_map(1,a) + (abs(cheetah_img_real(r,c) - cheetah_pred_map(r,c))) / (size(cheetah_img_real, 1) * size(cheetah_img_real, 2));
        end
    end

end

error_prob_ml = 0;
%ML error calculation
for r = 1 : size(cheetah_img_real, 1)
    for c = 1 : size(cheetah_img_real, 2)
        error_prob_ml = error_prob_ml + (abs(cheetah_img_real(r,c) - cheetah_pred_ml1(r,c))) / (size(cheetah_img_real, 1) * size(cheetah_img_real, 2));
    end
end

%converting error_prob_ml to list with all same values (to plot with alpha)
error_prob_ml_list = error_prob_ml + zeros(1, size(alpha, 2));

%%
%Plotting error prob wrt alpha
figure;
semilogx(alpha, error_prob);
title('Plotting error prob wrt \alpha');

snapnow

%%
%Plotting error prob ml wrt alpha
figure;
semilogx(alpha, error_prob_ml_list);
title('Plotting error prob ml wrt \alpha');

snapnow

%%
%Plotting error prob map wrt alpha
figure;
semilogx(alpha, error_prob_map);
title('Plotting error prob map wrt \alpha');

snapnow

%%
%Plotting error prob, error prob ml and error prob map with alpha
figure;
semilogx(alpha, error_prob, alpha, error_prob_ml_list, alpha, error_prob_map);
title('STRATEGY 2, DATASET 4'); %Changes based on the strategy and dataset
xlabel ('log \alpha');
ylabel('Probability of error');
legend('PD','ML','MAP');

snapnow


