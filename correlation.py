# NUMPY, SCIPY, AND PANDAS: CORRELATION WITH PYTHON
import numpy as np
import pandas as pd
import scipy.stats
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


# EXAMPLE: NumPy CORRELATION CALCULATION
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


# EXAMPLE: SciPy CORRELATION CALCULATION
# SciPy also has many statistics routines contained in scipy.stats:
# - pearsonr()
# - spearmanr()
# - kendalltau()

pearson = scipy.stats.pearsonr(x, y)  # Pearson's r
# (0.7586402890911869, 0.010964341301680832)
print(f"\nPearson's r for both arrays:\n{pearson}")

spearman = scipy.stats.spearmanr(x, y)  # Spearman's rho
print(f"\nSpearman's Rho:\n{spearman}")
# SpearmanrResult(correlation=0.9757575757575757,
#                 pvalue=1.4675461874042197e-06)

kendall = scipy.stats.kendalltau(x, y)  # Kendall's tau
print(f"\nKendall's Tau:\n{kendall}")
# KendalltauResult(correlation=0.911111111111111,
#                  pvalue=2.9761904761904762e-05)

# The p-value in statistical methods is for testing hypotheses. It is an
# important measure that requires in-depth knowledge of probability and stats
# to interpret.

# We can extract p-values and correlation coefficients using indices (since
# we get back tuples):
# pearson[0], spearman[0], or kendall[0]

# We can also use dot notation for the Spearman and Kendall coefficients:
print(f"\nSpearman correlation coefficient:\n{spearman.correlation}")
print(f"\nKendall correlation coefficient:\n{kendall.correlation}")

# Dot notation may be longer but is more readable and self-explanatory
# We can also use Python unpacking since these functions return tuples
r, p = pearson
print(f"\nPearson's r and p-value, respectively:\n{r}\n{p}")


# EXAMPLE: PANDAS CORRELATION CALCULATION
# Pandas is, in some cases, more convenient than NumPy and SciPy for
# calculating statistics. It offers statistical methods for Series and
# DataFrame instances. For example, given two Series objects with the same
# number of itesm, you can call .corr() on one of them with the other as the
# first argument.
x = pd.Series(range(10, 20))
y = pd.Series([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])

# Perason's r
print(f"\nPearson's r (Pandas) for x and y:\n{x.corr(y)}")
print(y.corr(x))

# Spearman's rho
print(f"\nSpearman's rho (Pandas): {x.corr(y, method='spearman')}")

# Kendall's tau
print(f"\nKendall's tau (Pandas): {x.corr(y, method='kendall')}")

# Notice with Pandas we use .corr() to calculate all 3 correlation coefficients

# LINEAR CORRELATION
# Linear correlation measures the proximity of the mathematical relationship
# between variables or dataset features to a linear function. If the
# relationship between the two features is closer to some linear function, then
# their linear correlation is stronger and the absolute value of the
# correlation coefficient is higher.

# Perason Correlation Coefficient
# Consider a dataset with two features: x and y. Each feature has n values, so
# x and y are n-tuples. If all values from x correspond to all values from y
# (x1 to y1, x2 to y2, etc.) then there are n pairs of corresponding values:
# (x1, y1), (x2, y2), etc. Each of these x-y pairs represents a single
# observation.
#
# The Pearson (product-moment) correlation coefficient is a measure of the
# linear relationship between two features. It's the ratio of the covariance of
# x and y to the product of their standard deviations. It's often denoted with
# the letter r and called Pearson's r and can be expressed methematically.
#
# Some important facts about the Pearson correlation coefficient:
# - The Pearson correlation coefficient can take on any real value in the range
#   -1 <= r <= 1
# - The maximum value r = 1 corresponds to the case when there's a perfect
#   positive linear relationship between x and y (i.e. larger x values
#   correspond to larger y values and vice versa)
# - The value r > 0 indicates positive correlation between x and y
# - The value r = 0 corresponds to the case when x and y are independent
# - The value r < 0 indicates negative correlation between x and y
# - The minimal value r = -1 corresponds to the case when there's a perfect
#   negative linear relationship between x and y (i.e. larger x values
#   correspond to smaller y values and vice versa)
#
# In table form, this info can be summed up:
#     Pearson's r | x and y Correlation
#     ------------+-------------------------------------
#         r = 1   | perfect positive linear relationship
#     ------------+-------------------------------------
#         r > 0   | positive correlation
#     ------------+-------------------------------------
#         r = 0   | independent
#     ------------+-------------------------------------
#         r < 0   | negative correlation
#     ------------+-------------------------------------
#         r = -1  | perfect negative linear relationship
#
# In short, a larger r indicates stronger correlation, closer to a linear
# function. A smaller r indicates weaker correlation.
