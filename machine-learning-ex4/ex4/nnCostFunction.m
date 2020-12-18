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

% ----------------------------------------
% Forward propagation
Thetas = {Theta1, Theta2};
num_thetas = size(Thetas, 2);
a_i = X;
for i = 1:num_thetas
    Theta = Thetas{i};

    % add bias
    a_i = [ones(m, 1) a_i];

    % Calculate z_i
    z_i = a_i * Theta';

    % Set the next activation
    a_i = sigmoid(z_i);
end

h_theta = a_i;

% y comes in as a list of numbers 0-9 e.g. [1]
% we need to turn this into matrix form
% [ 1
%   0
%   0
%   0
%   0
%   0
%   0
%   0
%   0
%   0 ]
y_relabeled = zeros(m, num_labels);
for index = 1:m
  y_relabeled(index, y(index)) = 1;
end

J = (1 / m) * sum ( sum ( -y_relabeled .* log(h_theta) - (1 - y_relabeled) .* log(1 - h_theta) ));

% Regularization
total = 0;
for index = 1:num_thetas
    Theta = Thetas{index};

    % ignore the bias by starting at index 2
    Theta = Theta(:, 2:size(Theta, 2));
    total = total + sum(sum(Theta .^ 2));
end

J = J + ((lambda / (2 * m)) * total);
% % ---------------------------------------------------

% Backpropagation
Theta_grads = {Theta1_grad, Theta2_grad};
for x = 1:m
    A = {[X(x, :)']};
    Z = {};
    for i = 1:num_thetas
        Theta = Thetas{i};

        % add bias
        A{i} = [1; A{i}];

        % Calculate z of i + 1
        Z{i + 1} = Theta * A{i};

        % Set the next activation
        A{i + 1} = sigmoid(Z{i + 1});
    end

    y_target = zeros(num_labels, 1);
    y_target(y(x), 1) = 1;

    % keep track of the previous delta, as we'll need to use it to calculate the next delta
    d_prev = [];
    for i = 1:num_thetas
        index = (num_thetas + 1) - (i - 1);
        if i == 1
            d_current = A{index} - y_target;
        else
            d_current = (Thetas{index}' * d_prev)(2:end) .* sigmoidGradient(Z{index});
        end

        Theta_grads{index - 1} = ((Theta_grads{index - 1}) + ((1 / m) .* (d_current * A{index - 1}')));
        d_prev = d_current;
    end
end

for i = 1:num_thetas
    Theta_grad = Theta_grads{i};
    Theta = Thetas{i};
    Theta_grad(:, 2:end) = Theta_grad(:, 2:end) + ((lambda / m) * Theta(:, 2:end));
    Theta_grads{i} = Theta_grad;
end

Theta1_grad = Theta_grads{1};
Theta2_grad = Theta_grads{2};
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
