import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def visualize(matrix: np.ndarray, title: str = 'scHi-C',
                xlabel: str = 'Genome position 1', ylabel: str = 'Genome position 2') -> None:
    """
    Visualizes a given contact matrix using a heatmap.

    Parameters:
    - matrix: np.ndarray
        The 2D contact matrix to be visualized. Must be a square numpy array.
    - title: str
        The title of the plot. Default is 'scHi-C'.
    - xlabel: str
        The label for the x-axis. Default is 'Genome position 1'.
    - ylabel: str
        The label for the y-axis. Default is 'Genome position 2'.

    Returns:
    - None
        The function generates a heatmap and displays it. Does not return any value.

    Raises:
    - ValueError: If the input 'matrix' is not a square 2D numpy array.
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input 'matrix' must be a square 2D numpy array.")

    # Define a custom color palette
    custom_cmap = LinearSegmentedColormap.from_list('custom_blue', ['#f7f7f7', '#016959'], N=256)

    # Create the plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, cmap=custom_cmap, square=True, cbar_kws={'shrink': 0.5})
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()