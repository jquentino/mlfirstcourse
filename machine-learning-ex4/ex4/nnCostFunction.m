function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Delta1=0;
Delta2=0;

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m,1) X];

for i=1:m
    %Feedforward propagation
    x_i = X(i,:);
    z2 = Theta1 * x_i';
    a2 = [1 ; sigmoid(z2)];
    z3 = Theta2 * a2;
    h = sigmoid(z3);
    y_vec = zeros(size(Theta2,1),1); %The outpot size is defined by Theta2 matrix
    y_vec(y(i)) = 1 ;
    
    J_i = (-y_vec'*log(h)+(y_vec-1)'*log(1-h));
    J = J + J_i;
    
    %Backforward propagation
    del3 = h - y_vec;
    del2 = Theta2'*del3.*sigmoidGradient([1;z2]);
    del2 = del2(2:end);
    Delta1 = Delta1 + del2*x_i;
    Delta2 = Delta2 + del3*a2';
end

J = J/m + (0.5*lambda/m)*(sum(Theta1(:,2:end).^2,'all') +sum(Theta2(:,2:end).^2,'all'));
Theta1_grad(:,1) = Delta1(:,1)/m;
Theta1_grad(:,2:end) = Delta1(:,2:end)/m + (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,1) = Delta2(:,1)/m;
Theta2_grad(:,2:end) = Delta2(:,2:end)/m + (lambda/m)*Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
