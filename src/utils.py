import numpy as np
import matplotlib.pyplot as plt

def FID_dataset(samples, testing):
    """Given two different data distributions computes their FID"""
    data1 = samples.detach().cpu().numpy().reshape(-1, 784)
    data2 = testing.detach().cpu().numpy().reshape(-1, 784)
    mu_1 = np.mean(data1, axis=0).reshape(-1, 784)
    mu_2 = np.mean(data2, axis=0).reshape(-1, 784)
    sigma_1 = np.cov(data1, rowvar=False)
    sigma_2 = np.cov(data2, rowvar=False)
    return FID(mu_1, mu_2, sigma_1, sigma_2)
    
def FID(mean_g, mean_d, std_g, std_d):
    """Given the mean and std of two different distributions compute FID"""
    means = np.sum((mean_g - mean_d)**2)
    tr = np.trace(std_g) + np.trace(std_d)
    Q = mat_root(std_g).dot(mat_root(std_d))
    f_term = -2 * np.trace(mat_root(np.matmul(Q, Q.T)))
    return means + tr + f_term

# This function finds the square root of a matrix
# assuming it is symmetric
def mat_root(x):
    """Assuming x is symmetric 2D square matrix, computes its square root"""
    evalues, evectors = np.linalg.eigh(x)
    evalues = np.clip(evalues, a_min=0, a_max=None)
    evalues = np.sqrt(evalues)
    return evectors @ np.diag(evalues) @ evectors.T


def display_images(data, nrows, ncols, titles=None):
    """Displays images"""
    plt.figure(figsize=(ncols*2, nrows*2))
    for i in range(min(nrows * ncols, data.shape[0])):
        plt.subplot(nrows, ncols, i+1)
        plt.tight_layout()
        if titles:
            plt.title(titles[i])
        plt.imshow(data[i].squeeze(), cmap='gray')