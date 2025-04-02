&emsp;This is my submission for the final project of the Codecademy Data Scientist: Machine Learning Career Path. \*__*Click on the date-a-scientist.ipynb file to view the FULL PROJECT.*__\*

&emsp;profiles.csv.zip is a zip file that contains the file profiles.csv, which is the original data file used for the project.

&emsp;To view __only__ the __INTRODUCTION__ and __SUMMARY__, continue reading below.

&emsp;To view individual __DATA VISUALIZATIONS__, click on the data_visualization folder.



## Introduction

&emsp;This project represents my submission for the final project of the __[Codecademy Data Scientist: Machine Learning Specialist Career Path](https://www.codecademy.com/learn/paths/data-science)__. In this Career Path, students go through ~95 hours of study in order to learn how to write code in the Python programming language, analyze data, communicate findings, and draw predictions using machine learning. Students may also earn a professional certification by passing all associated exams within the Career Path.

&emsp;In this project, I analyze data from OKCupid, a dating app that focuses on using multiple-choice and short-answer questions to match users. First, I explore the many differences between male and female users of the app. Then, I attempt to make a machine learning model that can predict if a user is male or female solely based on their answers to the aforementioned questions. \*\*\*Data was provided by Codecademy in the *Machine Learning Portfolio Project* portion of the Career Path.\*\*\*

&emsp;The data from OKCupid is stored in the accompanying file `profiles.csv`. It has the following columns of multiple-choice data:

- body_type
- diet
- drinks
- drugs
- education
- ethnicity
- height
- income
- job
- offspring
- orientation
- pets
- religion
- sex
- sign
- smokes
- speaks
- status

&emsp;And a set of short-answer responses to :

- essay0 - My self-summary
- essay1 - What I’m doing with my life
- essay2 - I’m really good at…
- essay3 - The first thing people usually notice about me…
- essay4 - Favorite books, movies, show, music, and food
- essay5 - The six things I could never do without
- essay6 - I spend a lot of time thinking about…
- essay7 - On a typical Friday night I am…
- essay8 - The most private thing I am willing to admit
- essay9 - You should message me if…



## Summary

&emsp;To start, all OKCupid user data was imported from `profiles.csv`, and the first 5 rows of data were previewed with the `.head()` method. From this preview, it is shown that there are 31 columns, each corresponding to a single feature of the data (such as age, body type, sex, income, etc.). Using the `.info()` method, it is also shown that data was gathered from **59,946** users. Unfortunately, there are a great deal of null, or empty, values in the data, because not all users wanted to answer all of the questions that OKCupid asked them.

&emsp;To make the data more useful for data analysis and machine learning models, some data cleaning and data wrangling had to be done. So, the first step taken was to remove the rows of data that represent users who did not input values for the following features: `height`, `ethnicity`, `religion`, `drinks`, `smokes`, `drugs`, `education`, `diet`, `body_type`, `job`, `offspring`, and `income`. Then, the `sex` feature was changed from using the strings `'f'` and `'m'` to using the numbers `0` and `1`, respectively. Next, `ethnicity` was broken into 8 separate features (`is_black`, `is_white`, `is_asian`, `is_hispanic_latin`, `is_native_american`, `is_pacific_islander`, `is_indian`, `is_middle_eastern`, `is_other`), and `orientation` was broken into 3 separate features (`is_straight`, `is_gay`, `is_bisexual`). Also, `religion`, `drinks`, `smokes`, `drugs`, `education`, `diet`, `body_type`, `job`, `offspring`, and `status` were changed to simpler binary features (`is_religious`, `is_drinker`, `is_smoker`, `uses_drugs`, `post_secondary_edu`, `adheres_to_diet`, `is_fit`, `has_job`, `has_kid`, `is_single`) that use `0`s and `1`s. Lastly, the `sign` feature was simplified to 12 possible responses, `age_group` was added as a feature, and `age`, `height`, & `income` were scaled down to ranges between -7 and 7 (instead of ranges up to 110 or 1,000,000) for machine learning.

&emsp;After data cleaning and wrangling, the `.info()` method was used again to see that there were only **2,604** users who provided answers to *every* feature used in this project. Fortunately, this number of users still provides a large enough sample to be representative of the full dataset. So, the data was almost ready for analysis at this point. The only thing left to do was to make male-only and female-only copies of the data for even easier analysis. On a side note, it is also learned that the data was collected between July 2011 and July 2012.

&emsp;From analysis of the data, many insights were gained. It is revealed that OKCupid had 67% males and 33% females, ranging in age from 18 to 69, with an average age of 33. Males had an average height of 5'10", while females averaged around 5'5". The Gender Wage Gap was also highlighted, with average pay for 18-19 year old females being ＄6,000 less than males in the same age group, but growing to be ＄45,000 less in the 30-39 year old age group. Furthermore, it is shown that 72% of users identified as White, 12% Asian, 10% Hispanic/Latino, and 8% Black. Religion played a role in only 60% of users' lives, while 92% of users drank alcoholic beverages. Around 49% of male users considered themselves fit, while only 23% of females would have said the same. 93% of users had some form of education after high school, and 86% of users were employed at the time. Only 20% of users had at least one child, with 26% of women being parents and only 18% of men. Overall, 89% of users identified as straight, but 17% of females identified as gay or bisexual, while only 7% of males did. Additionally, Cancer, Taurus, Pisces, Libra, and Capricorn were the top 5 Zodiac signs represented represented amongst users. *For more detailed analysis, including data visualizations, go to the Analyzing & Visualizing Data section of this project.*

&emsp;With data analysis complete, it was time to get into using some machine learning. For this project, the goal was to use different machine learning models in order to predict the sex of users based on their answers to the questions from the OKCupid app. First, 14 predictor variables were chosen (`age_centered`, `height_centered`, `income_centered`, `is_religious`, `is_drinker`,`is_smoker`, `uses_drugs`, `post_secondary_edu`, `adheres_to_diet`, `is_fit`, `has_job`, `has_kid`, `is_single`, `is_straight`) as a best guess for which features would help most with a machine learning model's predictions. Then, the data was split into a training set and testing set, which were used to train machine learning models and identify the levels of accuracy of their predictions. The first model used was a **K-Nearest Neighbors Classifier Model**, which learned to make predictions with 84.83% accuracy. Next, a **Logistic Regression Classifier Model** was used, and it learned to make predictions with 85.99% accuracy. Following this, a **Decision Tree Classifier Model** learned to make predictions with 86.56% accuracy. Then, a **Sequential Backward Selector** was used to see if better predictions could be made from picking its own predictor variables, and it chose 15 features (`age_centered`, `height_centered`, `income_centered`, `is_religious`, `is_drinker`,`is_smoker`, `adheres_to_diet`, `is_fit`, `is_black`, `is_white`, `is_asian`, `is_hispanic_latin`, `is_native_american`, `is_middle_eastern`, `is_bisexual`) that could be used by a **Logistic Regression Classifier Model** to make predictions with 86.67% accuracy. Finally, a **Random Forest Classifier Model** was trained to make predictions with 87.14% accuracy, and a **Stacking Classifier Model** was trained to make predictions with **95.97%** accuracy. *For  a more detailed look and the machine learning used in this project, go to the Applying Machine Learning to Data section.*

&emsp;In conclusion, this project provides a thorough analysis and numerous visualizations of data from the OKCupid dating app. It also explores various differences between female and male users of the app. In addition, it uses machine learning models in order to predict the sex of users based on their answers to specific questions. Many insights were gained from this project, all starting with Python code and 1 data file.