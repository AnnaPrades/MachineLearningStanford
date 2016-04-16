function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % 1 - hypothesis vector
    h = X * theta;
    % 2- errors vector
    error = h - y;
    % 3- theta_change vector
    %he change in theta (the "gradient") is the sum of the product of X 
    %and the "errors vector", scaled by alpha and 1/m
    % Since X is (m x n), and the error vector is (m x 1),and you want
    % theta(2*1 or n *1) -> you need to transpose X
    %vector multiplication authomaticaly sums the products
    theta_change = alpha * (1/m) * (X'*error);
    %4- Subtract this "change in theta" from the original value of theta.
    theta = theta - theta_change;
    

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
