# NUMPY, SCIPY, AND PANDAS: CORRELATION WITH PYTHON
import numpy as np
# Correlation coefficients quantify the association between variables and
# features of a dataset. These stats are of high importance for science and
# technology, and Python has tools to calculate them: SciPy, NumPy, and Pandas

# CORRELATION
# Statistics and data science are concerned about the relationships between two
# or more variables (features) of a dataset. Each data point is an observation
# and the features are the properties or attributes of those observations.
#
# Every dataset uses variables and observations. For example you might be
# interested in understanding:
# - How the height of basketball players is correlated to their shooting
#   accuracy
# - Whether there's a relationship between employee work experience and salary
# - What mathematical dependence exists between the population density and the
#   GDP of different countries
#
# Height, shooting accuracy, years of experience, salary, population density,
# and GDP are features (variables). The data related to each player, employee,
# and each country are the observations.
#
# When the data is represented in the form of a table, the rows are the
# observations, the columns are features.
#
# If you analyze any two features of a dataset, then you'll find some type of
# correlation between those two features.
#
# There are 3 different forms of correlation:
# 1. Negative correlation: y values decrease as x increases.
#    - Strong negative correlation occurs when large values of one feature
#      correspond to small values of the other, and vice versa.
# 2. Weak or no correlation: no obvious trend.
#    - Weak correlation occurs when an association between two features is not
#      obvious or is hardly observable
# 3. Positive correlation: y values increase as x increases.
#    - Strong positive correlation occurs when large values of one feature
#      correspond to large values of the other, and vice versa.
#
# NOTE: When analyzing correlation, always keep in mind that correlation DOES
#       NOT imply causation. It quantifies the strength of the relationship
#       between the features of a dataset. Sometimes, the association is
#       caused by a factor common to several features of interest.
#
# Correlation is tightly connected to other statistical quantities like mean,
# standard deviation, variance, and covariance.
#
# There are several statistics you can use to quantify correlation. Three of
# them are:
# - Pearson's r
# - Spearman's rho
# - Kendall's tau
#
# Pearson's coefficient measures linear correlation, while Spearman and Kendall
# coefficients compare the ranks of data.

# Example: NumPy Correlation Calculation
# NumPy has many statistics routines. np.corrcoef() returns a matrix of Pearson
# correlation coefficients.

# arange() creates an array of integers between 10 (inclusive) and 20
# (exclusive)
x = np.arange(10, 20)  # array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
print(f'First array:\n{x}')

y = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
# array([ 2,  1,  4,  5,  8, 12, 18, 25, 96, 48])

print(f'Second array:\n{y}')

# Once you have two arrays of the same length, you can call np.corrcoef() with
# both arrays as arguments
r = np.corrcoef(x, y)
# array([[1.        , 0.75864029],
#        [0.75864029, 1.        ]])

print(f'Correlation coefficient for both arrays:\n{r}')

# corrcoef() returns the correlation matrix, which is a 2D array with the
# correlation coefficients

# The values on the main diagonal of the correlation matrix (upper left and
# lower right) are equal to 1. The upper left value corresponds to the
# correlation coefficient for x and x, while the lower right value is the
# correlation coefficient for y and y. They are always equal to 1.

# What you usually need are lower left and upper right values of the
# correlation matrix. These values are equal and both represent the Pearson
# correlation coefficient for x and y.
