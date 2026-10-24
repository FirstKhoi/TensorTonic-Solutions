<!-- tensortonic:start -->
# Lương Nhất Khôi's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/nhatkhoi.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Implement Adam Optimizer Step | Implement one vectorized Adam optimizer step in NumPy with first and second moments, bias correction, and elementwise parameter updates. | https://www.tensortonic.com/problems/adam-optimizer |
| Batch Normalization (Forward) | Implement the batch-normalization forward pass in NumPy using feature-wise statistics, scale, shift, and numerical stability. | https://www.tensortonic.com/problems/batch-normalization |
| Bernoulli Probability Mass Function & Moments | Compute the Bernoulli probability mass function, expected value, and variance for a valid success probability. | https://www.tensortonic.com/problems/bernoulli-pmf |
| Binomial Probability Mass Function | Compute binomial probability mass and cumulative probabilities from trial count, success probability, and outcome. | https://www.tensortonic.com/problems/binomial-pmf-cdf |
| 2D Convolution (Image Filtering) | Apply a 2D convolution kernel to a single-channel image with configurable zero-padding and stride. | https://www.tensortonic.com/problems/conv2d-image-filtering |
| Implement Cosine Similarity | Implement cosine similarity from scratch with NumPy. Covers dot products, Euclidean norms and zero-vector handling, with tests in the browser. | https://www.tensortonic.com/problems/cosine-similarity |
| Compute Covariance Matrix | Compute a sample covariance matrix from centered observations, preserving feature-to-feature relationships. | https://www.tensortonic.com/problems/covariance-matrix |
| Implement Cross-Entropy Loss | Compute multiclass cross-entropy loss from class probabilities and integer labels with stable logarithms. | https://www.tensortonic.com/problems/cross-entropy-loss |
| Implement Dot Product | Implement the dot product of equal-length numeric vectors by summing element-wise products without library shortcuts. | https://www.tensortonic.com/problems/dot-product |
| Implement Dropout (Training Mode) | Implement training-mode dropout in NumPy with random masking and inverted scaling of retained activations. | https://www.tensortonic.com/problems/dropout-training |
| Expected Value (Discrete Distribution) | Compute the expected value of a discrete distribution from matched outcomes and normalized probabilities. | https://www.tensortonic.com/problems/expected-value-discrete |
| Implement Gradient Descent for a 1D Quadratic | Optimize a one-dimensional quadratic with iterative gradient descent and return the parameter trajectory. | https://www.tensortonic.com/problems/gradient-descent-quadratic |
| He Initialization | Implement He weight initialization from scratch in NumPy. See the formula, why fan-in scaling matters, and test your code in the browser. | https://www.tensortonic.com/problems/he-initialization |
| Implement Leaky ReLU (with α) | Apply Leaky ReLU element-wise with a configurable negative slope while retaining positive inputs. | https://www.tensortonic.com/problems/leaky-relu |
| Linear Layer Forward | Implement a dense linear layer forward pass by multiplying inputs by weights and adding a bias vector. | https://www.tensortonic.com/problems/linear-layer-forward |
| Logistic Regression Training Loop | Train binary logistic regression in NumPy using sigmoid probabilities, gradient descent, and learned weight and bias parameters. | https://www.tensortonic.com/problems/logistic-regression-training |
| Matrix Inverse | Compute a square matrix inverse in NumPy while returning no result for invalid, non-square, or singular inputs. | https://www.tensortonic.com/problems/matrix-inverse |
| Implement Matrix Normalization | Normalize a NumPy matrix using the specified axis and norm while safely handling zero-magnitude slices. | https://www.tensortonic.com/problems/matrix-normalization |
| Matrix Transpose | Implement matrix transpose in NumPy without built-in transpose helpers, preserving rectangular shapes and the original input. | https://www.tensortonic.com/problems/matrix-transpose |
| Mean, Median, Mode | Calculate the mean, median, and deterministic mode of a numeric collection, including tied frequencies. | https://www.tensortonic.com/problems/mean-median-mode |
| Mean Squared Error (MSE) | Compute mean squared error between predictions and targets by averaging their squared element-wise differences. | https://www.tensortonic.com/problems/mean-squared-error |
| Implement Positional Encoding (sin/cos) | Generate sinusoidal Transformer positional encodings across sequence positions and embedding dimensions, with the formula and tests in the browser. | https://www.tensortonic.com/problems/positional-encoding |
| Implement ReLU Activation | Apply the ReLU activation element-wise by replacing negative values with zero and preserving nonnegative inputs. | https://www.tensortonic.com/problems/relu-activation |
| Sample Variance & Standard Deviation | Compute sample variance and standard deviation with Bessel's correction from a numeric collection. | https://www.tensortonic.com/problems/sample-var-std |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Implement a Simple CNN Layer (NumPy) | Implement a NumPy CNN layer forward pass with batched valid convolution across channels and bias addition. | https://www.tensortonic.com/problems/simple-cnn-layer |
| Implement Softmax Function | Implement numerically stable softmax by shifting logits before exponentiation and normalizing probabilities. | https://www.tensortonic.com/problems/softmax-function |
| Implement Tanh Activation | Implement the hyperbolic tangent activation element-wise with outputs bounded between minus one and one. | https://www.tensortonic.com/problems/tanh-activation |
| Xavier Initialization | Scale raw weights into the Xavier uniform range using a bound derived from fan-in and fan-out. | https://www.tensortonic.com/problems/xavier-initialization |
| Data Augmentation | Implement AlexNet image augmentation operations for deterministic crops, horizontal flips, and intensity changes. | https://www.tensortonic.com/research/alexnet/alexnet-augmentation |
| AlexNet Convolution Layer | Implement an AlexNet convolutional layer with learned filters, bias, stride, padding, and multi-channel outputs. | https://www.tensortonic.com/research/alexnet/alexnet-conv-layers |
| Dropout Regularization | Implement inverted dropout for AlexNet with seeded masks and training-versus-inference behavior. | https://www.tensortonic.com/research/alexnet/alexnet-dropout |
| Local Response Normalization | Implement AlexNet local response normalization across neighboring channels using the paper's scaling equation. | https://www.tensortonic.com/research/alexnet/alexnet-lrn |
| Overlapping Max Pooling | Implement AlexNet overlapping max pooling with a 3x3 window and stride two across spatial dimensions. | https://www.tensortonic.com/research/alexnet/alexnet-pooling |
| ReLU Activation Function | Implement AlexNet's elementwise ReLU activation, preserving positive values while setting negative values to zero. | https://www.tensortonic.com/research/alexnet/alexnet-relu |
| BatchNorm in ResNet | Implement ResNet batch normalization with channel statistics, learned scale and bias, and training or inference behavior. | https://www.tensortonic.com/research/resnet/resnet-batch-norm |
| Bottleneck Block | Build a ResNet bottleneck block using 1x1 channel reduction, 3x3 convolution, and 1x1 channel expansion. | https://www.tensortonic.com/research/resnet/resnet-bottleneck |
| Convolutional Block | Implement a ResNet convolutional block with a projected shortcut that matches changed spatial and channel dimensions. | https://www.tensortonic.com/research/resnet/resnet-conv-block |
| Full ResNet Assembly | Assemble a ResNet forward pass from the stem, residual stages, global average pooling, and the classification head. | https://www.tensortonic.com/research/resnet/resnet-full-network |
| Identity Block | Implement a ResNet identity block with a three-layer bottleneck branch, batch normalization, ReLU, and an unchanged skip path. | https://www.tensortonic.com/research/resnet/resnet-identity-block |
| Skip Connection Analysis | Analyze ResNet skip connections by combining residual and identity tensors and tracking gradient flow through the addition. | https://www.tensortonic.com/research/resnet/resnet-skip-connection |
| Scaled Dot-Product Attention | Implement scaled dot-product attention in PyTorch using query-key scores, softmax weights, and value aggregation. | https://www.tensortonic.com/research/transformer/transformers-attention |
| Embedding Layer | Create PyTorch token embeddings and scale each lookup by the square root of the Transformer model dimension. | https://www.tensortonic.com/research/transformer/transformers-embedding |
| Encoder Block | Assemble a Transformer encoder block with multi-head attention, residual paths, layer normalization, and a feed-forward network. | https://www.tensortonic.com/research/transformer/transformers-encoder-block |
| Feed-Forward Network | Implement the Transformer's position-wise feed-forward network with two linear projections and a ReLU activation. | https://www.tensortonic.com/research/transformer/transformers-feed-forward |
| Layer Normalization | Implement Transformer layer normalization in NumPy using per-token mean, variance, scale, and bias. | https://www.tensortonic.com/research/transformer/transformers-layer-normalization |
| Multi-Head Attention | Build NumPy multi-head attention with learned projections, per-head scaled attention, concatenation, and output projection. | https://www.tensortonic.com/research/transformer/transformers-multi-head-attention |
| Positional Encoding | Implement sinusoidal Transformer positional encodings in NumPy with alternating sine and cosine dimensions. | https://www.tensortonic.com/research/transformer/transformers-positional-encoding |
| Tokenization | Build a word-level Transformer tokenizer with fixed special-token IDs, sorted vocabulary entries, encoding, and decoding. | https://www.tensortonic.com/research/transformer/transformers-tokenization |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/nhatkhoi)
<!-- tensortonic:end -->
