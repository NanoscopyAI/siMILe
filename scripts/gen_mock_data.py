import numpy as np
import pandas as pd


def generate_3d_gaussian_data(n, mean=1.0, std=2.0, filename="normal_data_3d.csv"):
    """Generate 3D data points from a normal distribution and save to CSV."""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(mean, (int, float)):
        raise ValueError("mean must be a number")
    if not isinstance(std, (int, float)) or std <= 0:
        raise ValueError("std must be a positive number")

    # Generate 3D data points from normal distribution
    data = np.random.normal(loc=mean, scale=std, size=(n, 3))

    # Create a pandas DataFrame
    df = pd.DataFrame(data, columns=["x", "y", "z"])  # type: ignore pandas typing is bad

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f'Sampled Gaussian 3D data points saved to: "{filename}".')

    return filename


if __name__ == "__main__":
    # Generate 5 data points as an example
    filename = generate_3d_gaussian_data(n=30, mean=1.0, std=2.0, filename="rand_data_3d_a.csv")
    filename = generate_3d_gaussian_data(n=30, mean=1.5, std=2.0, filename="rand_data_3d_b.csv")

    # # Display the first few rows
    # print("First few rows:")
    # print(pd.read_csv(filename))
