
%----------------------------Training Data Evaluation-----------------------

%Loading the Training Data:

A = load("TrainingSamplesDCT_8.mat");

FG_samples = (A.TrainsampleDCT_FG);

BG_samples = (A.TrainsampleDCT_BG);



%Calculating the priors from training data:

Cheetah_Prior = size(FG_samples,1)/(size(BG_samples,1) + size(FG_samples,1));

Grass_Prior = 1 - Cheetah_Prior;



%Displaying the prior probabilities obtained above:

disp(['The prior probability for cheetah class is ', num2str(Cheetah_Prior)])

disp(['The prior probability for grass(background) class is ', num2str(Grass_Prior)])



%Calculating the value of X for each 64D FG block:  

X_FG_samples = zeros(size(FG_samples,1), 1);

for row = 1:size(FG_samples,1)
    min_coeff = min(FG_samples(row,:));
    
    [largest_coeff, largest_coeff_index] = max(FG_samples(row,:));
    
    FG_samples(row,largest_coeff_index) = min_coeff;

    [value, X_FG_samples(row,1)]  = max(FG_samples(row,:)); 

end


%Calculating the value of X for each 64D BG block:

X_BG_samples = zeros(size(BG_samples,1),1);

for row = 1:size(BG_samples,1)
    min_coeff = min(BG_samples(row,:));
    
    [largest_coeff, largest_coeff_index] = max(BG_samples(row,:));
    
    BG_samples(row,largest_coeff_index) = min_coeff;

    [value, X_BG_samples(row,1)]  = max(BG_samples(row,:)); 
end



%Evaluating the histograms for P(x|cheetah) and P(x|grass) respectively:

Cheetah_hist = histogram(X_FG_samples, 64, 'Normalization','probability','BinWidth',1,'BinLimits',[0.5,64.5]);

Cheetah_conditional_probabilities = Cheetah_hist.Values;


Grass_hist = histogram(X_BG_samples, 64, 'Normalization','probability','BinWidth',1,'BinLimits',[0.5,64.5]);

Grass_conditional_probabilities = Grass_hist.Values;



%--------------Test Data Evaluation----------------------------------------

%Loading the test image and converting it into doubles in the range [0,1]:

original_test_image = im2double(imread("cheetah.bmp"));



% Creating the test image post-padded with zeros:

padded_test_image = zeros((size(original_test_image,1) + 7), (size(original_test_image,2) + 7));

padded_test_image(1:size(original_test_image,1),1:size(original_test_image,2)) = original_test_image(:,:);


% Creating the matrix test_results which stores predicted Y values:

test_results = zeros(size(original_test_image,1), size(original_test_image,2));


% Creating the blueprint for mapping the 8*8 frequency coefficient blocks
% into 64D vectors obtained by following the zig-zag pattern:

zig_zag_blueprint = readmatrix("Zig-Zag Pattern.txt");



% Main block for sliding the 8*8 image block across the image, determining the value of X and 
% predicting the corresponding Y values.

for row = 1:(size(test_results,1))
    
    %Creating the 8*8 image block:

    im_block = zeros(8,8);
    
    
    for column = 1:(size(test_results,2))
        
        %Assigning the 8*8 image block image values:
        
        im_block = padded_test_image(row:row+7, column:column+7);
        
        

        %Calculating the Discrete Cosine Transform for the 8*8 block:
        
        coeff_block = abs(dct2(im_block));
        
        

        %Creating the 64D row vector for the block:
        
        zig_zag_vector = zeros(1,64);
        

        %Loop for rearranging the 8*8 block into a 64D row vector by following the zig-zag pattern:
        
        for small_row = 1:8
            for small_column = 1:8
                a = 1 + zig_zag_blueprint(small_row, small_column);
                zig_zag_vector(1,a) = coeff_block(small_row, small_column);
            end
        end
        

        %Selecting the index of the 2nd largest coefficient from the
        %zig-zag vector and assigning it to X.

        min_coeff = min(zig_zag_vector(1,:));
    
        [largest_coeff, index] = max(zig_zag_vector(1,:));
    
        zig_zag_vector(1,index) = min_coeff;

        [value,X]  = max(zig_zag_vector(1,:));


        %Calculating the class conditional probabilities of X for Cheetah
        %and Grass classes respectively.

        P_X_given_Cheetah = Cheetah_conditional_probabilities(1,X);
        
        P_X_given_Grass = Grass_conditional_probabilities(1,X);        
        
        % Using the Bayesian Decision Rule, 
        %   
        %       i*(x) = arg_max_{i} P(x|i)P(i) 
        % 
        % we calculate the value of Y, the obtained value of X,

        %if Y is the cheetah class, we assign it a value of 1, otherwise we
        %assign it to be 0.

        if (Cheetah_Prior * P_X_given_Cheetah) > (Grass_Prior * P_X_given_Grass)
            test_results(row,column) = 1;
        else
            test_results(row,column) = 0;
        end
        
    end
end




%Reading the groundtruth segmentation mask of the test image

ground_truth = im2double(imread('cheetah_mask.bmp'));



%Displaying the test_results

img = imagesc(test_results);

colormap(gray(255))



%Calculating the error probability from our predicted mask matrix
%test_results and the provided ground_truth matrix.

error = 0;

for i=1:size(test_results,1)
    for j = 1:size(test_results,2)
        error = error + abs(test_results(i,j) - ground_truth(i,j));
    end
end

error = error/(size(test_results,1)*size(test_results,2));


%Displaying the error probability,

disp(['The error probability is ', num2str(error)])
