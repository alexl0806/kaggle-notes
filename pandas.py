import pandas as pd

# sample DataFrame
pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})

# sample Dataframe with index
pd.DataFrame({'Haha': [132, 13], 'Baha': [121, 14]}, index=['Yes', 'No'])

# sample Series
pd.Series([1, 2, 3, 4, 5])

# sample Series with index
pd.Series([1, 2, 3], index=['2014', '2015', '2016'], name='Heheheha')

# reading csv file
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0) # index is first column
wine_reviews.shape # checks how large the dataframe is

reviews.country # gets the country columns
reviews['country'] # also works the same way

reviews.iloc[0] # retrieve the first row of data
reviews.iloc[:, 0] # gets the first columns of data
reviews.iloc[[0, 1, 2], 0] # passing in a list as the index
reviews.iloc[-5:] # counts backwards to get the last 5 entries

reviews.iloc[0, 0] # both get the first entry
reviews.loc[0, 'country']

# iloc excludes the last element of the range, whereas loc includes it

reviews.set_index("title") # setting the index to a column

reviews.country == 'Italy' # checks to see if each wine is Italian or not (produces a series with Booleans)
reviews.loc[reviews.country == 'Italy'] # produces a df with only wines from Italy
reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)] # adding conditions to filter
reviews.loc[reviews.country.isin(['Italy', 'France'])] # gets the data whose value "is in" the list of values
reviews.loc[reviews.price.notnull()]

reviews['critic'] = 'everyone' # assigning data (constant value)
reviews['index_backwards'] = range(len(reviews), 0, -1) # assigning data with iterable values

# getting the wines that are from either Australia or New Zealand that have a point score of at least 95
top_oceania_wines = reviews.loc[(reviews.country.isin(['Australia', 'New Zealand'])) & (reviews.points >= 95)]

# getting data from summary functions
reviews.points.mean()
reviews.taster_name.unique() # gets a list of unique values
reviews.taster_name.value_counts() # see how often a value occurs in the data set

review_points_mean = review.points.mean()
reviews.points.map(lambda p: p - review_points_mean) # subtracting the mean from the points for each value, example of map

def remean_points(row):
    row.points = row.points - review_points_mean
    return row

# another method that transforms a whole DataFrame by calling a custom method on each row
reviews.apply(remean_points, axis='columns')

# alternatively, there are some built-in pandas operations
reviews.points - review_points_mean

reviews.country + " - " + reviews.region_1 # combining couuntry and region info in the dataset

# built-in functions are faster than map() and apply(), but are not as versatile

# looking for string in description
n_trop = reviews.description.map(lambda desc: "tropical" in desc).sum()
n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()
descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])
descriptor_counts
