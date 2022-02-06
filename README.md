# Image-angle-Regression



Generally we will consider MSE loss for regression task, but angle is periodic, and MSE loss can't handle this problem.

So I change it to vector regression, and the objective function is maximizing the cosine between two vectors.