function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.1;
sigma = 0.01;

min_error=1;
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
predictions= svmPredict(model, Xval);
while(C<=100)
    for sigma=logspace(-2,1,7)
     C=C*2;
     model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
     predictions= svmPredict(model, Xval);
     error=mean(double(predictions ~= yval));
     if(error<min_error)
        min_error=error;
        C_opt=C;
        sigma_opt=sigma;
     end

    end
end

C=C_opt;
sigma=sigma_opt;

end
