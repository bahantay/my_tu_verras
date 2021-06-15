Introduction
Today we are going to work on a Boston housing prices dataset. Basically what a data scientist would want to do with such a dataset is to build a model to predict the value (in this case, the MEDV) based on the different other attributes like the number of rooms, the distance to schools etc.

Here, we are not going to make any predictions from models, but we are going to try and understand what the data has to tell us.

First Glance
Let's get down to brask task.

From the scikit-learn website, we know what is inside our dataset, like the different attributes:

• CRIM per capita crime rate by town
• ZN proportion of residential land zoned for lots over 25,000 sq.ft.
• INDUS proportion of non-retail business acres per town
• CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
• NOX nitric oxides concentration (parts per 10 million)
• RM average number of rooms per dwelling
• AGE proportion of owner-occupied units built prior to 1940
• DIS weighted distances to five Boston employment centres
• RAD index of accessibility to radial highways
• TAX full-value property-tax rate per $10,000
• PTRATIO pupil-teacher ratio by town
• B 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
• LSTAT % lower status of the population
• MEDV Median value of owner-occupied homes in $1000’s

Let's explore it deeper and import the data from sklearn:

from sklearn.datasets import load_boston

dataset = load_boston()
Now that we loaded the data, let's peek at it.

→ Load the data in a dataframe and print the first rows.

def load_dataframe(dataset):
	...

boston_dataframe = load_dataframe(dataset)

print(boston_dataframe[:20])

hint: Panda is cool.

We should see something similar to:


Now that we have loaded the dataframe, it is easier to see and investigate into the data. Play around a bit and print some more values.
The dataframe table is cool but is hard to read. Can you tell from it what attributes have the most influence on the MEDV target?

From a quick sneak peek, we feel like the MEDV goes down as the AGE goes up. This is obviously just a supposition and we will try to corroborate it.
Do you see any other relationship between attributes?

Cleaning and Pre-processing
Just like we did with Mr Clean, cleaning is one of the first steps after receiving a dataset to explore. Here the dataset was made to be very clean but just to be sure, let's look for any missing values.

→ Make a function count_missing_values which return the number of missing values for each attribute.

def count_missing_values(boston_dataframe):
	...

print(count_missing_values(boston_dataframe))
If everything works fine, we should have zero missing values in the dataset. If we had any missing values, we would have had another step to remove the rows containing them, as these data woud be relevant.

Data Analysis
Now we are ready to cut to the chase. We will see if our supposition was right by visualizing the data.

Visualization is very important and useful to understand the relationship between the target and the other features.

A great way to get a feel of the data we are dealing with is to plot each atrtibute in a histogram.

→ Plot each attribute in a histogram.

Hint: You can use matplotlib, panda, seaborn or any other libraries.

Here is the result for some of the attributes:


We can see a few things from these histograms:

Looking for correlations
So far we have only taken a quick peek to get a general understanding of the kind of data we are manipulating. Now the goal will be to investigate deeper again.

The size of our dataset is not too large so we can try to analyze linear correlations between every pair of attribute.

→ Write a function compute_correlations_matrix to compute Pearson's correlations between every pair of attributes.

The output of compute_correlation should be a dictionary. For example, print(correlations['MEDV']) should show the different correlation coefficient between the median value and other attributes.

def compute_correlations_matrix(boston_dataframe):
	...

correlations = compute_correlations(boston_dataframe)

print(correlations['MEDV'])

When the coefficient is close to 1 (in absolute value), it means there is a strong correlation between the two variables. If it is positive, it means the linear correction is, well, positive. If it is negative... you get it.
Coeffients close to zero means there is no linear correlation between the attributes.

→ What is the correlation coefficient between the median value and the number of rooms?

This coefficient is positive and is the biggest. It means that the median value increases when the number of rooms increases. Well that makes sense, we expect a house with a lot of rooms to be more expensive than a single-room appartment.

→ Analyse the correlations between the median value and the other attributes. Which attribute is the most negatively correlated with the median value? Does it make sense to you?

Mathematically, all this information can be seen in an object, the covariance matrix. The covariance matrix tells us a lot about the relationship and the distribution of our data.
For example, it can be used in Principal Component Analysis to try and understand the most useful and relevant dimensions. But that's for another day.

Numbers are cool but we, as human being, love colors, pictures, and visual representations. We can easily spot correlations by visualizing the relationship between attributes on graphs.

→ Plot every attribute against each other.

hint: Visualizing every component against each other is usually done with a scatter matrix.

Here is an example:


The main diagonal, instead of representing straight lines, displays a histogram of the attributes.

Since the most linearly correlated feature is the average number of rooms, let's focus on the plot of MEDV as a function of RM.

→ Plot MEDV in function of RM

We should see what the numbers told us before. There is a strong positive linear correlation between the number of room and the median value. We clearly the upward trend.
Ok now let's explore the influence of other attributes.

→ Plot the correlation scatter plot of the median value against LSTAT, AGE, and CRIME.


→ What can you observe? What can you say?.

→ Does the age seems to have any kind of influence on the MEDV?

Well, not so fast! the MEDV / AGE scatterplot does not show any obvious correlation between these attributes. But we can see that the LSTAT feature DOES have some influence on MEDV.
The more LSTAT increases, the lower is the MEDV value. This is valuable information itself. Can we not go further?

Let's step back a minute. If LSTAT influences MEDV, it means that if another attribute influences LSTAT itself, it then actually circles back to influence MEDV!

→ Plot the scatter matrix or print the correlation coefficients for LSTAT. What are the attributes which are the most linearly correlated with LSTAT?

Here is an example, LSTAT against AGE:


Again, we clearly see a trend here. Points are not too dispersed and has an upward trend. LSTAT seems to be positively linearly correlated with AGE.
So all in all, AGE actually seems to influence (negatively) the median value. We could already observe this result on the graph of MEDV / AGE. Varying AGE cascades down (or back-propagates) to make MEDV varies.

You can go on experimenting to try and find more relationship between attributes to understand more deeply how they influence de MEDV target values.

What's next?
After exploring the data and gaining more insights, the next step would be to do another round of data cleaning and find a model to predict the MEDV. A linear regression would be a good start.
