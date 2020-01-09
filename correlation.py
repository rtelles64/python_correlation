# NUMPY, SCIPY, AND PANDAS: CORRELATION WITH PYTHON

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
