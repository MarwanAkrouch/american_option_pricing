# American Option Pricing

In this project, American options are replaced with Bermudan options with a small step size for computational convenience.

The project is based on the Longstaff-Schwartz algorithm, but instead of using polynomial regression to approximate a conditional expectation, a neural network is used.

## Project Structure

The project contains the following files:

- `Cox-Ros-Rub.py`: Implements the Cox-Ross-Rubinstein binomial model for pricing European and American options.
- `DNN.py`: Uses a deep neural network to price Bermudan options using Monte Carlo simulation.
- `DNN_MultiDim.py`: Extends the DNN approach to handle multi-dimensional underlying assets.
- `Least_square_monte_carlo.py`: Implements the Least Squares Monte Carlo (LSMC) method for pricing American options.
- `Least-square-monte-carlo_MultiDim.py`: Extends the LSMC approach to handle multi-dimensional underlying assets.
- `README.md`: This file.

## Files Description

### `Cox-Ros-Rub.py`

This file implements the Cox-Ross-Rubinstein binomial model for pricing European and American options. It includes functions for calculating the option value and generating a binomial tree. The tree is visualized using HTML and JavaScript.
    ```sh
    python Cox-Ros-Rub.py
    ```

### `DNN.py`

This file uses a deep neural network to price Bermudan options using Monte Carlo simulation. It includes a class for generating log-normal paths and functions for training the neural network to approximate the continuation value of the option.
    ```sh
    python DNN.py
    ```

### `DNN_MultiDim.py`

This file extends the DNN approach to handle multi-dimensional underlying assets. It includes functions for generating multi-dimensional log-normal paths and training the neural network for multi-dimensional options.
    ```sh
    python DNN_MultiDim.py
    ```

### `Least_square_monte_carlo.py`

This file implements the Least Squares Monte Carlo (LSMC) method for pricing American options. It includes a class for generating log-normal paths and functions for fitting a polynomial regression to approximate the continuation value of the option.
    ```sh
    python Least_square_monte_carlo.py
    ```

### `Least-square-monte-carlo_MultiDim.py`

This file extends the LSMC approach to handle multi-dimensional underlying assets. It includes functions for generating multi-dimensional log-normal paths and fitting a multivariate polynomial regression to approximate the continuation value of the option.
    ```sh
    python Least-square-monte-carlo_MultiDim.py
    ```

## Usage

1. **intall dependencies**:
    ```sh
    pip install numpy matplotlib keras
    ```

To run the project, you can execute any of the Python files. For example, to run the Cox-Ross-Rubinstein binomial model, use the following command:

## References

* Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). Option pricing: A simplified approach. Journal of Financial Economics, 7(3), 229-263.

* Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation: A simple least-squares approach. The Review of Financial Studies, 14(1), 113-147.