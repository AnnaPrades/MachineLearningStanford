function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%1- Compute a vector h -> h(x) = theta T * x = theta0 + theta*x1
h = X * theta;

%2 -Compute the difference between the hypothesis and y
error = h - y;

%3- Compute the square of each of those error terms (using element-wise exponentiation)
error_sq = error.^2;

% sum of the error_sqr vector, and scale the result (multiply) by 1/(2*m)
J = 1/(2*m) * sum(error_sq);


% =========================================================================

end
