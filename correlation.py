# NUMPY, SCIPY, AND PANDAS: CORRELATION WITH PYTHON
import matplotlib.pyplot as plt
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

# Pearson Correlation Coefficient
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

# Linear Regression: SciPy Implementation
# Linear regression is the process of finding the linear function that best
# describes the association between features. This linear function is also
# called the regression line.
#
# We can implement linear regression with SciPy. We'll get the linear function
# that best approximates the relationship between two arrays, as well as the
# Pearson correlation coefficient.
#
# We use scipy.stats.linregress() to perform linear regression for two arrays
# of the same length. The arrays are passed as arguments, and the outputs are
# retrieved using dot notation.
x = np.arange(10, 20)
y = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])

result = scipy.stats.linregress(x, y)

print(
    "\nLinear Regression with SciPy:"
    f"\nSlope: {result.slope}",  # 7.4363636363636365
    f"\nIntercept: {result.intercept}",  # -85.92727272727274
    f"\nr-value: {result.rvalue}",  # 0.7586402890911869
    f"\np-value: {result.pvalue}",  # 0.010964341301680825
    f"\nStandard error: {result.stderr}"  # 2.257878767543913
)

# With scipy we've completed linear regression with the following results:
# - slope: the slope of the regression line
# - intercept: the intercept of the regression line
# - pvalue: the p-value
# - stderr: the standard error of the estimated gradient
#
# We could've also provided our data as a single argument to linregress but it
# must be passed as a 2D array with one dimension of length two.

# NOTE: scipy.stats.linregress() considers rows as features and columns as
#       observations. That's because there are two rows.
#
#       In machine learning, the practice is opposite: rows are observations,
#       columns are features. Many machine learning libraries follow this
#       convention.
#
#       When analyzing correlation in a dataset, you should note how
#       observations and features are indicated.

# linregress() returns the same result if you provide the transpose.
xy = np.array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
               [2, 1, 4, 5, 8, 12, 18, 25, 96, 48]])
print("\nLinear Regression with Transpose (SciPy):",
      f"\n{scipy.stats.linregress(xy.T)}")

# You should also be careful to note whether or not your dataset contains
# missing values. In data science and machine learning, you'll often find
# missing or corrupted data. The usual representation is by using NaN (No a
# Number). If your data contains nan values, you won't get a useful result.
#
# You can check whether a variable corresponds to nan with math.isnan() or
# numpy.isnan().

# Pearson Correlation: NumPy and SciPy Implementation
# We can get the Pearson correlation coefficient with corrcoef() and pearsonr()
#
# NOTE: if we provide an array with nan values to pearsonr(), we get ValueError
#
# Instead of passing two arrays as arguments, we can pass corrcoef() one 2D
# array and get the same results. Again, the first row represents one feature,
# the second row represents the other.
#
# To get correlation coefficients for three features, just provide a 2D array
# with three rows:
xyz = np.array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [2, 1, 4, 5, 8, 12, 18, 25, 96, 48],
                [5, 3, 2, 1, 0, -2, -8, -11, -15, -16]])

print(f"\nCorrelation matrix with a 2D array:\n{np.corrcoef(xyz)}")
# [[ 1.          0.75864029 -0.96807242]
#  [ 0.75864029  1.         -0.83407922]
#  [-0.96807242 -0.83407922  1.        ]]

# By default, numpy.corrcoef() considers rows as features, and columns as
# observations. If you want the opposite behavior, which is widely used in
# machine learning, then use rowvar=False
print("\nMatrix with different convention:",
      f"\n{np.corrcoef(xyz.T, rowvar=False)}")
# [[ 1.          0.75864029 -0.96807242]
#  [ 0.75864029  1.         -0.83407922]
#  [-0.96807242 -0.83407922  1.        ]]

# Pearson Correlation: Pandas Implementation
x = pd.Series(range(10, 20))
y = pd.Series([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
z = pd.Series([5, 3, 2, 1, 0, -2, -8, -11, -15, -16])
xy = pd.DataFrame({'x-values': x, 'y-values': y})
xyz = pd.DataFrame({'x-values': x, 'y-values': y, 'z-values': z})

# NOTE: When working with DataFrame instances, rows are observations and
#       columns are features. This is consistent with the usual practice in
#       machine learning

# If you provide a nan value, then .corr() will still work, but will exclude
# observations that contain nan values
u, u_with_nan = pd.Series([1, 2, 3]), pd.Series([1, 2, np.nan, 3])
v, w = pd.Series([1, 4, 8]), pd.Series([1, 4, 154, 8])
print(f'\nCorrelation without NaN: {u.corr(v)}')  # 0.9966158955401239
print(f'Correlation with NaN: {u_with_nan.corr(w)}')  # 0.9966158955401239
# We get the same value because corr() ignores the pair of values (np.nan, 154)
# that has a missing value

# We can also use corr() with DataFrame objects to get the correlation matrix
# for their columns
corr_matrix = xy.corr()
print(f'\nDataFrame correlation matrix:\n{corr_matrix}')

# The resulting correlation matrix is a new instance of DataFrame and holds the
# correlation coefficients for the columns xy['x-values'] and xy['y-values'].
# These labeled results are usually convenient to work with because you can
# access them with either their labels or their integer position indices:
print(f"\nAccess by label: {corr_matrix.at['x-values', 'y-values']}")
print(f"Access by index: {corr_matrix.iat[0, 1]}")

# You can apply corr() the same way with DataFrame objects that contain three
# or more columns:
print(f'\nDataFrame correlation (3+ columns):\n{xyz.corr()}')
#           x-values  y-values  z-values
# x-values  1.000000  0.758640 -0.968072
# y-values  0.758640  1.000000 -0.834079
# z-values -0.968072 -0.834079  1.000000

# From the above matrix, we have the following correlation coefficients:
# - 0.758640 for x-values and y-values
# - -0.968072 for x-values and z-values
# - -0.834079 for y-values and z-values

# Another useful method is corrwith(), which allows you to calculate the
# correlation coefficients between the rows or columns of one DataFrame and
# another Series or DataFrame passed as the first argument:
print(f'\nCorrelation using corrwith():\n{xy.corrwith(z)}')
# In this case, the result is a new Series with the correlation coefficient for
# the column xy['x-values'] and the values of z, as well as the coefficient for
# xy['y-values'] and z

# corrwith() has the optional parameter 'axis' that specifies whether columns
# or rows represent the features. The default value is axis = 0, and also
# defaults to columns representing features. There's also a 'drop' parameter,
# which indicates what to do with missing values
#
# Both corr() and corrwith() have the optional parameter 'method' to specify
# the correlation coefficient that you want to calculate. The Pearson
# correlation coefficient is the default


# RANK CORRELATION
# Rank correlation compares the ranks (orderings) of the data related to two
# variables or dataset features. If the orderings are similar, then the
# correlation is strong, positive, and high. If the orderings are close to
# reversed, then the correlation is strong, negative, and low. In other words,
# rank correlation is concerned only with the order of values, not with the
# particular values from the dataset.
#
# When looking only at rank, all linear correlation relationships are positive.
# Larger x values corresponding to larger y values are perfect positive rank
# correlations. Smaller x values corresponding to smaller y values are perfect
# negative rank correlations.

# Spearman Correlation Coefficient
# The Spearman correlation coefficient between two features is the Pearson
# correlation coefficient between their rank values. It's calculated the same
# way as the Pearson correlation coefficient but takes into account their ranks
# instead of their values. It's often denoted with the Greek letter rho and
# called Spearman's rho.
#
# Say you have two n-tuples, x and y, where (x1, y1), (x2, y2), ... are the
# observations as pairs of corresponding values. You can calculate the Spearman
# correlation coefficient rho the same way as the Pearson coefficient. You'll
# use ranks instead of the actual values from x and y.
#
# Some important facts about the Pearson correlation coefficient:
# - It can take a real value in the range -1 <= rho <= 1
# - Its maximum value rho = 1 corresponds to the case when there's a
#   monotonically increasing function between x and y. In other words, larger x
#   values correspond to larger y values and vice versa.
# - Its minimum value rho = -1 corresponds to the case when there's a
#   monotonically decreasing function between x and y. In other words, larger x
#   values correspond to smaller y values and vice versa.
#
# You calculate Spearman's rho in Python similar to Pearson's r.

# Kendall Correlation Coefficient
# Considering two n-tuples, x and y. Each x-y pair (x1, y1), (x2, y2), ... is a
# single observation. A pair of observations (xi, yi) and (xj, yj), where i < j
# will be one of three things:
# - concordant if either (xi > xj and yi > yj) or (xi < xj and yi < yj)
# - discordant if either (xi < xj and yi > yj) or (xi > xj and yi < yj)
# - neither if there's a tie in x (xi = xj) or a tie in y (yi = yj)
#
# The Kendall correlation coefficient compares the number of concordant and
# discordant pairs of data. This coefficient is based on the difference in the
# counts of concordant and discordant pairs relative to the number of x-y
# pairs. It's often denoted by the Greek letter tau and called Kendall's tau.
#
# Some important facts about the Kendall correlation coefficient are as
# follows:
# - It can take a real value in the range -1 <= tau <= 1
# - Its max value tau = 1 corresponds to the case when the ranks of the
#   corresponding values in x and y are the same. In other words, all pairs are
#   concordant.
# - Its min value tau = -1 corresponds to the case when the rankings in x are
#   the reverse of the rankings in y. In other words, all pairs are discordant.
#
# You can calculate Kendall's tau in Python similarly to Pearson's r.

# Rank: SciPy Implementation
# We can use scipy.stats to determine rank for each value in an array using
# rankdata()
x = np.arange(10, 20)
y = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
z = np.array([5, 3, 2, 1, 0, -2, -8, -11, -15, -16])
print(
    "\nRank data (SciPy):",
    f'\nx: {scipy.stats.rankdata(x)}',
    f'\ny: {scipy.stats.rankdata(y)}',
    f'\nz: {scipy.stats.rankdata(z)}'
)
# The arrays x and z are monotonic, so their ranks are monotonic as well. The
# smallest value in y is 1 and it corresponds to the rank 1. The second
# smallest is 2, which corresponds to rank 2. The largest value is 96, which
# corresponds to the largest rank 10 since there are 10 items in the array.
#
# rankdata() has the optional parameter 'method'. This tells Python what to do
# if there are ties in the array (if two or more values are equal). By default,
# it assigns them the average of the ranks:
scipy.stats.rankdata([8, 2, 0, 2])  # array([4. , 2.5, 1. , 2.5])
# There are two elements with value 2 and they have the ranks 2.0 and 3.0. The
# value 0 has rank 1.0 and the value 8 has rank 4.0. Then, both elements with
# the value 2 will get the same rank 2.5.

# rankdata() treats nan values as if they were large:
scipy.stats.rankdata([8, np.nan, 0, 2])  # array([3., 4., 1., 2.])
# In this case, the value np.nan corresponds to the largest rank 4.0.

# You can also get ranks with np.argsort():
np.argsort(y) + 1  # array([ 2,  1,  3,  4,  5,  6,  7,  8, 10,  9])
# argsort() returns the indices that the array items would have in the sorted
# array. These indices are zero-based, so you'll need to add 1 to all of them.

# Rank Correlation: NumPy and SciPy Implementation
# You can calculate the Spearman correlation coefficient with
# scipy.stats.spearmanr():
result = scipy.stats.spearmanr(x, y)
# SpearmanrResult(correlation=0.9757575757575757,
#                 pvalue=1.4675461874042197e-06)
print(f'\nSpearman result: {result}')
rho, p = result
print(f'Spearman rho and p-value, respective: {rho}, {p}')

# We get the same result if we provide a 2D array that contains the same data
xy = np.array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
               [2, 1, 4, 5, 8, 12, 18, 25, 96, 48]])
rho, p = scipy.stats.spearmanr(xy, axis=1)
print(f'\nSpearman rho and p (2D array), respective:\n{rho},\n{p}')
# The first row of xy is one feature, while the second row is another feature.
# The optional parameter 'axis' determines whether columns (axis=0) or rows
# (axis=1) represent the features. The default behavior is that rows are
# observations and columns are features.

# Another optional parameter 'nan_policy' defines how to handle nan values. It
# can take one of three values:
# - 'propogate' returns nan if there's a nan value among the inputs. This is
#   the default behavior
# - 'raise' raises a ValueError if there's a nan value among the inputs
# - 'omit' ignores the observations with nan values
#
# If you provide a 2D array with more than two features, you get the
# correlation matrix and the matrix of the p-values
xyz = np.array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [2, 1, 4, 5, 8, 12, 18, 25, 96, 48],
                [5, 3, 2, 1, 0, -2, -8, -11, -15, -16]])
corr_matrix, p_matrix = scipy.stats.spearmanr(xyz, axis=1)
print(f'\nCorrelation Matrix:\n{corr_matrix}')
# [[ 1.          0.97575758 -1.        ]
#  [ 0.97575758  1.         -0.97575758]
#  [-1.         -0.97575758  1.        ]]
print(f'\nP-value Matrix:\n{p_matrix}')
# [[6.64689742e-64 1.46754619e-06 6.64689742e-64]
#  [1.46754619e-06 6.64689742e-64 1.46754619e-06]
#  [6.64689742e-64 1.46754619e-06 6.64689742e-64]]

# The value -1 in the correlation matrix shows that the first and third
# features have a perfect negative rank correlation (i.e. larger values in the
# first row always correspond to smaller values in the third)

# You can obtain the Kendall correlation coefficient with kendalltau()
result = scipy.stats.kendalltau(x, y)
# KendalltauResult(correlation=0.911111111111111,
#                  pvalue=2.9761904761904762e-05)
tau, p = result
print(f'\nKendall tau and p, respective:\n{tau}\n{p}')
# kendalltau() works much like spearmanr(): It takes two 1D arrays, has the
# optional parameter nan_policy, and returns an object with the values of the
# correlation coefficient and p-value

# If you provide only one 2D array as an argument, then kendalltau() will raise
# a TypeError. If you pass two multi-dimensional arrays of the same shape, then
# they'll be flattened before the calculation

# Rank Correlation: Pandas Implementation
# You can calculate the Spearman rho and Kendall tau correlation coefficients
# with Pandas
x, y, z = pd.Series(x), pd.Series(y), pd.Series(z)
xy = pd.DataFrame({'x-values': x, 'y-values': y})
xyz = pd.DataFrame({'x-values': x, 'y-values': y, 'z-values': z})

# We can use corr() and corrwith() just like when calculating the Pearson
# correlation coefficient. We just need to specify the desired correlation
# coefficient 'method', which defaults to 'pearson'
spearman_x = x.corr(y, method='spearman')  # 0.9757575757575757
spearman_xy = xy.corr(method='spearman')
#           x-values  y-values
# x-values  1.000000  0.975758
# y-values  0.975758  1.000000
spearman_xyz = xyz.corr(method='spearman')
#           x-values  y-values  z-values
# x-values  1.000000  0.975758 -1.000000
# y-values  0.975758  1.000000 -0.975758
# z-values -1.000000 -0.975758  1.000000
spearman_with = xy.corrwith(z, method='spearman')

print(f"\nSpearman's rho:\n{spearman_x}\n{spearman_xy}\n{spearman_xyz}")
print(f"\nSpearman's with corrwith():\n{spearman_with}")

# If you want Kendall's tau, use method=kendall:
kendall_x = x.corr(y, method='kendall')
kendall_xy = xy.corr(method='kendall')
kendall_xyz = xyz.corr(method='kendall')
kendall_with = xy.corrwith(z, method='kendall')

print(f"\nKendall's tau:\n{kendall_x}\n{kendall_xy}\n{kendall_xyz}")
print(f"\nKendall's with corriwth:\n{kendall_with}")

# Unlike with SciPy, you can use a single 2D data structure (DataFrame)


# VISUALIZATION OF CORRELATION
# Set the style of the plots
plt.style.use('ggplot')

x = np.arange(10, 20)
y = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
z = np.array([5, 3, 2, 1, 0, -2, -8, -11, -15, -16])
xyz = np.array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [2, 1, 4, 5, 8, 12, 18, 25, 96, 48],
                [5, 3, 2, 1, 0, -2, -8, -11, -15, -16]])

# X-Y Plots with a Regression Line
# Get the slope and intercept of the regression line, as well as the
# correlation coefficient (use linregress())
slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)

# Get the string representation of the equation of the regression line and the
# value of the correlation coefficient
line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
print(f'\n{line}')

# Now cfeate the x-y plot with plot()
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=0, marker='s', label='Data poins')
ax.plot(x, intercept+slope*x, label=line)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(facecolor='white')
plt.show()
# The red squares in the plot represent observations, the blue line is the
# regression line. Its equation is listed in the legend, together with the
# correlation coefficient

# Heatmaps of Correlation Matrices
# The correlation matrix can become big and confusing when you have a lot of
# features. You can present it visually as a heatmap where each field has the
# color that corresponds to its value.
#
# You need the correlation matrix
corr_matrix = np.corrcoef(xyz).round(decimals=2)
# array([[ 1.  ,  0.76, -0.97],
#        [ 0.76,  1.  , -0.83],
#        [-0.97, -0.83,  1.  ]])
#
# You can round numbers with round() as they're going to be on the heatmap
#
# Create heatmap with imshow() and the correlation matrix as its argument
fig, ax = plt.subplots()
im = ax.imshow(corr_matrix)
im.set_clim(-1, 1)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1, 2), ticklabels=('x', 'y', 'z'))
ax.yaxis.set(ticks=(0, 1, 2), ticklabels=('x', 'y', 'z'))
ax.set_ylim(2.5, -0.5)

for i in range(3):
    for j in range(3):
        ax.text(j, i, corr_matrix[i,j], ha='center', va='center', color='r')

cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
plt.show()

# The result is a table with the coefficients. Yellow represents the number 1,
# green corresponds to 0.76, purple is used for negative numbers.


# CONCLUSION
# Correlation coefficients are stats that measure association between variables
# or features of datasets.
#
# You can now use Python to calculate:
# - Pearson's product-moment correlation coefficient
# - Spearman's rank correlation coefficient
# - Kendall's rank correlation coefficient
#
# Now you can use NumPy, SciPy, and Pandas correlation functions and methods
# to effectively calculate these (and other) stats, even when you work with
# large datasets.
#
# You can also visualize data, regression lines, and correlation matrices with
# Matplotlib plots and heatmaps
