function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X*theta;
theta_ = theta;
theta_(1) = 0;
cost_reg = lambda/(2*m)*sum(theta_ .* theta_);
J = 1/(2*m) * transpose(h-y)*(h-y) + cost_reg;

grad = 1/m * transpose(X) * (h-y);
grad_reg = lambda/m * theta;
grad_reg(1) = 0;
grad = grad + grad_reg;

% =========================================================================

grad = grad(:);

end
