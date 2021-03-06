
# Mod3 Project: PoliticalAffiliation

## Presentation Link
[Google Slides Link](https://docs.google.com/presentation/d/1mTeWWz0e7ZeS89ZXlbpMjVIzXqBZDaZQr57pHrJ2dBg/edit?usp=sharing)

## File Index
Data Collection.ipynb
This workbook scrapes US Census Data as well as bestplaces.net for political affiliation by city

Data Cleaning.ipynb
This workbook cleans our data and deals with outliers so we're ready for EDA

EDA and Feature Engineering.ipynb
This workbook contains our EDA as well as our feature engineering

Modeling.ipynb
This workbook contains our feature selection and modeling.

## Business Case
Our goal with this project is to use freely available census data to build a model that will classify a city into two broad categories: liberal and conservative. This will allow advertisers, political campaigns and non-profits to best target their
ad spend in areas where they will either find a more receptive audience or where they can attempt to change the hearts and minds of their consumers. 

## Data Sources and Cleaning
We first collected the names and locations of the most populous 5000 cities in the United States. We then converted this into a format that would allow us to scrape our two data sources: www.census.gov and bestplaces.net. We built functions that would scrape this data, report on its progress and then combined then into one usable data-frame of close to 4500 entries.

Our starting feature set is as follows:

`CITY` = city <br>
`STATE` = state <br>
`PA` = political affiliation <br>
`POP` = population <br>
`UNDER_5` = % of people under age 5 <br>
`UNDER_18` = % of people under age 18 <br>
`65_OR_OVER` = % of people 65 and older <br>
`FEMALE` = % female <br>
`WHITE` = % white only <br>
`BLACK` = % black only <br>
`AMERICAN_INDIAN` = % american indian or alaska native only <br>
`ASIAN` = % asian only <br>
`PACIFIC_ISLANDER` = % native hawaiian and other pacific islander only <br>
`MULTI_RACE` = % two+ races <br>
`HISPANIC` = % hispanic <br>
`VETERAN` = # of veterans <br>
`FOREIGN` = % foreign born <br>
`HOUSES` = % of owner occupied houses <br>
`HOUSE_VAL` = median value of owner occupied houses <br>
`RENT` = median gross rent <br>
`HOUSEHOLDS` = households <br>
`PPH` = people per household <br>
`YEAR_IN_HOUSE` = % of people living in same house for one year or more <br>
`OTHER_HOME_LANG` = % of people, 5+ years old, who speak a language other than english at home <br>
`COMPUTER` = % of households with computer <br>
`INTERNET` = % of households with internet <br>
`HIGH_SCHOOL` = % of people, 25+ years old, with high school diploma or higher <br>
`BACH_DEGREE` = % of people, 25+ years old, with bachelors degree or higher <br>
`DISABILITY_UNDER_65` = % of people with a disability under 65 years old <br>
`NO_INSURANCE` = % of people without health insurance, under 65 years old <br>
`LABOR` = % of population in civilian labor force, 16+ years old <br>
`FEM_LABOR` = % of female population in civilian labor force, 16+ years old <br>
`HEALTHCARE` = healthcare and social assistance revenue (\$1,000) <br>
`SHIPMENTS` = manufacturers shipments (\$1,000) <br>
`TRAVEL_TIME` = average travel time to work (min), 16+ years old <br>
`HOUSEHOLD_INCOME` = average household income <br>
`INCOME` = per capita income in last 12 months <br>
`POVERTY` = % of people in poverty <br>
`FIRMS` = all firms <br>
`MEN_FIRM` = men owned firms <br>
`FEM_FIRM` = female owned firms <br>
`MINOR_FIRM` = minority owned firms <br>
`NON_MINOR_FIRM` = nonminority owned firms <br>
`VET_FIRM` = veteran owned firms <br>
`NON_VET_FIRM` = non-veteran owned firms <br>
`POP_AREA` = population per square mile <br>
`AREA` = land area in square miles <br>

Our target variable is liberal vs conservative which we will turn into 1 vs 0.

We dont need city or state anymore as they are just identifiers, so we remove them.

The first thing we do is check for and remove null values. We're missing data on around 1000 data points so we are left with about 3500 rows. Next we need to check for zero values that indicate missing rather than real data. We see that we're missing a large amount of data for healthcare revenue and manufacturing shipments so we will drop these as features. Other columns such as % population that is pacific islander we know can be zero so we will leave these. There are also a number of features missing small amounts of data so we remove these rows as it wont meaningfully impact our output. A few of the features can be directly inferred from others: female owned firms vs male owned so we will drop one of each of these sets. 

We have a number of columns that are percentages but currently in whole numbers so we will divide by 100 to put them in percentage form. 

Now we'll look at outliers! We're first going to turn some of our count based columns into percentages where we can to help constrain the range. We turned # of veterans into % of the population as well as used total firms and firm types to make those percentages as well. We're going to use our outlier evaluation function as well as our box plots to help determine outliers. Our non-percentage data looks very skewed so we are going to log-transform our columns to help normalize this.

<img src="Images/continous_charts.png">

After our transforms we still have a significant number of outliers but its much lower than before. We remove these and are left with around 3k data points. 

We've cleaned our data and are now going to focus on EDA.

## EDA
The first thing we want to check for is class imbalance. 

<img src="Images/class_plot.png">

We can clearly see that there is no class imbalance, we in fact have a 49:51 split! Next lets look at our contious variables:

<img src="Images/continuous_charts_logs.png">

We can clearly see that they're less skewed than before the transformation. Our takeaways from these are:

* The vast majority of cities have under 50,000 people. (mean = 30,664)
  *  This (quite obviously) somewhat matches the households and population per square mile distributions.
* The rent and per person income distributions follow each other pretty closely.
  *  The median household income distribution also follows the distribution of the per person income.
* Most owner occupied homes are under \$200,000. (mean = 182,032)
* Most households have 2-3 people. (mean = 2.6)
* Most cities have under 3,000 firms. (mean = 2,666)
* Most cities are under 20 square miles. (mean = 15)

Now we'll look at the rest of our features.

<img src="Images/percent_charts.png">

From these graphs we've made the following insights:

* The following features have a mean percentage of less that 10%. We believe they probably will not be huge factors in our model.
  *  People under age 5 (6%)
  *  American Indian or Alaska Native people (0%)
  *  Asian people (3%)
  *  Native Hawaiian and Other Pacific Islander people (0%)
  *  People of 2+ races (3%)
  *  Foreign born people (9%)
  *  Veterans (6%)
* Most cities have an almost even split of males and females. (`FEMALES` mean = 51%)
* Most cities have a majority population of white people. (mean = 76%)
  *  On the same hand, most cities have a minority population of other races. (`BLACK` mean = 12%, `HISPANIC` = 14%, others covered above)
  *  It also then makes sense a very small number of people don't speak english at home. (mean = 15%)
  *  And a majority of firms are owned by non-minorities. (mean = 73%)
* A little over half of houses are owner occupied. (mean = 60%)
* Most people have been living in the same house for a year or more. (mean = 83%)
* Most households have a computer and internet. (means = 87%, 78%, respectively)
* Most people have a high school diploma. (mean = 87%)
* Most people do not have a bachelor's degree. (mean = 27%)
* A small amount of people have a disability under the age of 65. (mean = 10%)
* Most people under 65 have health insurance. (`NO_INSURANCE` mean = 11%)
* Most cities have around 62% of their population in the labor force.
  *  With a little over half the labor force being female. (mean = 57%)
* Most cities have a low number of people in poverty. (mean = 16%)
* Around a third of firms are owned by females. (mean = 34%)
* Most firms are not owned by veterans. (mean = 82%)

Now that we've made our insights, we'll bin columns with extremely small values (AMERICAN_INDIAN and PACIFIC_ISLANDER) into 1/0 categories. We will set the percent of American Indian or Alaska Natives only column to be of value 1 if it's over 1% of the population, and 0 otherwise. For the percent of Native Hawaiian and other Pacific Islander only column, we'll do a 1 if it is over 0%, and 0 otherwise.

Next, let's compare some of our numeric features with our target variable.

First, we can see how cities having a mostly white population will affect our target. Since WHITE is a percentage value, we'll first round these values to the nearest decimal.

<img src="Images/white_stacked.png">

From this graph, we can clearly see that the "conservative" class grows as the percentage of white people goes up. At the far right we can see that when cities are around 100% white, they are almost 100% conservative.

Next, let's do the same for our female population.

<img src="Images/female_stacked.png">


From this graph we can see that the percentage of females doesn't have a very big change on our target. The vast majority of cities sit at around 50% female, and the class distribution for this looks to be split evenly.

Let's look at the population, this time making bins of 10,000s.

<img src="Images/pop_stacked.png">

We can see a slight shift here in classes. As population goes up, the number of conservative cities goes down. It's important to note this could be due to the smaller sample sizes we get with larger populations.

Lastly, let's see if we can find any relationship between per person income and our target.

<img src="Images/income_stacked.png">

It doesn't seem like per person income affects our target by much. At the tail, we can see that the "liberal" class begins to overcome the other by some but this may be due to less samples at that income value.

Now that we've explored our data, lets make some features!

## Feature Engineering

Now that we've explored our data, we can use the insights we gained to create new features we hope will help our model make better predictions.

The first two we want to make are:

`MAJ_FEMALE` = majority female (0 = less than 50% of population is female, 1 = more than 50% of population is female)
`MAJ_WHITE` = majority white (0 = less than 50% of population is white, 1 = more than 50% of population is white)

<img src="Images/maj_female.png">

<img src="Images/maj_white.png">

From our EDA, we know that having a majority of white people definitely affects our target variable. We'd like to make a feature which notes whether a city has a high percentage of white people and less than 5% of other minorities.

<img src="Images/super_white.png">

Our next feature will be `HOUSE_RATIO`, which will check if median house value over per person income affects our target.

<img src="Images/house_price_inc.png">

Finally, we created a binary category for where the city population is above 50,000.

<img src="Images/large_pop.png">

Later on we'll perform some hypothesis testing to see if there are statistically significant differences in our new features.

## Feature Selection I

### Hypothesis Testing on New Features

We'll run Z tests on these features to check the following, with the groups meaning the features we made: Null Hypothesis: Proportion of different classes among these groups is the same. Alternative Hypothesis: Proportion of different classes among these groups is different.

We won't run a hypothesis test on our `HOUSE_RATIO` column because it is the only non-class variable.

If we can reject the null hypothesis, then we know that feature may be useful for our model. Otherwise, we won't use it. We will use a p-value of 0.05.

Using our two-sample proportion test we arrive at a p value of .06 for MAJ_FEMALE so we do not reject the null. The p values we get for `MAJ_WHITE`, `SUPER_WHITE`, and `POP_LARGE` are essentially 0 so we strongly reject the null for each of these. We will move forward with these columns but decided not to use MAJ_FEMALE in our model.

Next we're going to check for multicollinearity. We get the following sets with very high correlations:
* `WHITE` and `BLACK`
* `HISPANIC` and `OTHER_HOME_LANG`
* `FOREIGN` and `OTHER_HOME_LANG`
* `COMPUTER` and `INTERNET`
* `LABOR` and `FEM_LABOR`
* `POVERTY` and `LOG_HOUSEHOLD_INCOME`
* `LOG_HOUSEHOLD_INCOME` and `LOG_INCOME`
* `LOG_POP`, `LOG_HOUSEHOLDS`, and `LOG_FIRMS`

We cannot remove `WHITE` or `BLACK` because they were both used to make other features. As the collinearity is only a bit higher than 0.85, we'll keep both in, but note that they are highly correlated. Next, we see `OTHER_HOME_LANG` and `LOG_HOUSEHOLD_INCOME` are both correlated to two other features, so we will remove them. We can't remove LOG_POP because it is used to make other features, so we'll remove `LOG_HOUSEHOLDS` and `LOG_FIRMS` from this line. We then looked at the correlations between the rest and our output variable to decide what to drop. We end up with the final heatmap below showing the issues we mentioned above but overall it looks reasonable.

<img src="Images/heatmap.png">

## Initial Models
We built a function to take a variety of models, our data, and gridsearch if desired. Then we built base models for all 9 classification models we've learned. 

<img src="Images/base_models.png">

Next we used GridSearch to tune these models before evaluation.

<img src="Images/grid_models.png">

## Feature Selection II
In the context of our data, there is no need to prioritize precision or recall since the cost for false negatives/positives is the same. Due to this, we focus on accuracy and F1 score as our primary evaluation metrics.

From our initial models we decided that Gradient Boosting Classifier, Support Vector Classification, and Logistic Regression have the best accuracy and F1 score for both base models and models with gridsearched best parameters. We'll focus on these for our future models.

We'll get coefficients for the logistic model to see if we have any features we want to drop which may improve our models. Doing this we decided to drop: 
* `POP_LARGE`
* `AMERICAN_INDIAN`
* `PACIFIC_ISLANDER`
* `SUPER_WHITE`

Now lets see if this improves our models!

<img src="Images/improve_models.png">

Our Logistic Regression model's accuracy decreased very slightly, but every other metric improved, and every metric for our Gradient Boosting Classifier and Support Vector Classification models went up!

Lastly, let's try creating some polynomial features and seeing how these affect our models.

<img src="Images/poly_models.png">

Using polynomial features reduced our model metrics so now we can choose our final model!

## Final Model

Our best model is a Gradient Boosting Classifier on X_improv features! We found an accuracy of ~.84 and F1 the same. The top features from our model were population density, % of the population that is foreign born and % of the population that is white.

<img src="Images/features.png">

Next we looked at a confusion matrix to see how the where the wrong predictions were.

<img src="Images/confusion.png">

We have a very even split between False Negatives/Positives.

<img src="Images/roc.png">

We also built an ROC curve and calculated our AUC score, showing an AUC of 0.9 which means our model is quite accurate!

## Conclusions
We were able to get a relatively accurate broad model using basic census data. With easily over a billion dollars spent on election campaigns, being able to segment potential voters/consumers into voting blocs is potentially very lucrative. The finer you can segment the population the more accurate your model and the more effective you can be with your spend.
