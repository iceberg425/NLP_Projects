# Topic Modeling and Deep Learning 


<img src="./images/wordcloud_1.png" alt="drawing" width="800"/>

## Problem Statement

  Employing text data, we seek to:
      - Predict interest rate movements
      - Explore the content of the Beige books.
  That is, we want to answer the question: using the economic reports of the Federal reserve, can we predict changes in the interest rate. 


## Executive Summary

   The Stock Market Crash of 1987 raised some questions about how the Federal Reserve would respond. "Would the Federal Open Market Committee (FOMC) inject more money into the economy?" "Was such an expansionary monetary policy justified?" "Was a "wait-and-see" attitude more prudent?" Since reliable economic statistics would not be immediately available, policymakers needed to determine the state of the economy in the days following the Crash. The question then was: "on what information would the Fed base its decision making?" 

   A Businessweek article from the same year provided an answer: "Thousands of ... tidbits have poured into the Federal Reserve System's Washington headquarters since Bloody Monday. ... The regional Feds survey businesses in their districts, tapping more than 300 members of various Fed boards, as well as hundreds of informal contacts, to compile the 'Beige Book' on regional business conditions. ... Even 'eyeball evidence'—like Minneapolis Fed President Gary H. Stern's car counts at the local malls—go into the information stream."
    
   This artcile was the public's introduction to the Beige book. The Beige Book is an "anecdotal compilation of economic reports from each Federal Reserve district, from which a national summary is then drawn, and which is submitted to the Federal Reserve Board and released to the public two weeks prior to FOMC meetings, or eight times a year." It should be noted that the Beige Book is *only* one piece of information used in the making of monetary policy; the FOMC relies most heavily on macroeconomic forecasts and on more current data. Still, the release of this document prompts market participants to search for clues to Fed policy. But is such faith in the "prophetic powers" of the Beige Book justified? Also, how did such a document develop and how did it come to take on such importance?
   
   We do not attempt to answer these questions here (they are beyond the scope of this project). Rather we seek to answer the question: *do the contents of the Beige book inform Fed policy? More precisely, does its content affect interest rate movements?* 
   
   To tackle this problem we take two broad approaches:
       1. An unsupervised learning approach
       2. A supervised learning approach
   
   The data was scraped from the Minneapolis Fed [website](https://www.minneapolisfed.org/news-and-events/beige-book-archive). This data includes: 1) the beige book, 2) timestamps, and 3) Districts reports. The data on interest rates were sourced from [Macrotrends](https://www.macrotrends.net/2016/10-year-treasury-bond-rate-yield-chart). The 10 year treasury is the benchmark used to decide mortgage rates across the U.S. and is the most liquid and widely traded bond in the world. Due to the large size of the data, the dataset is not accessible on Github. However, this data can be made available upon request.
   
   We begin with a topic model, using a Latent Dirichlet Allocation (LDA) model with both Sckit-learn and Gensim libraries. The results from both LDA models are similar. We find that the optimal number of topics `n_components` is five. The keywords that arise from this analysis suggest the following topics:
       1. The Primary sector
       2. Retail and Wholesale sectors
       3. Consumer Spending and Business Conditions
       4. Delinquency Rate
       5. Small Businesses
       
   We also conduct a cluster analysis, using the t-distributed Stochastic Neighbor Embedding (t-SNE). We find that even after tuning, by changing the perplexity (from 30, 50, and then 70), the clusters do not change. In all plots, we see *tight and slightly separated clusters*. This possibly suggests that, although we are able to identify five topics in the documents, the vocabulary of these documents are very similar. This we beleive is demonstrated by the tSNE plot.
   
   We then turn to a classification problem, that is we attempt to predict if the interest rates increased, decreased, or remains unchanged. In our dataset, the proportion of observations that remain unchanged is negligible (it is only 0.16%). We ignore this and proceed with a binary classification as opposed to a multiclass problem.
   
   Our findings suggest that predicting interest rate movements may require more **X** features. That is, our model may not include other variables that account for the variation in our dependent variable. To conduct the modeling we employ both parametric and non-parametric models. 
   
   
### Notebooks:
- 1. Data Gathering
    - [1.1. Data Gathering and Cleaning 1](https://github.com/iceberg425/NLP_Projects/tree/master/Data_Gathering_and_Cleaning_1.ipynb)
    - [1.2. Data Gathering and Cleaning 2](https://github.com/iceberg425/NLP_Projects/tree/master/Data_Gathering_and_Cleaning_5.ipynb)
- 2. Data Exploration
    - [2.1. EDA 1](https://github.com/iceberg425/NLP_Projects/tree/master/EDA_2.ipynb)
    - [2.2. EDA 2](https://github.com/iceberg425/NLP_Projects/tree/master/Topic_Model_EDA_District.ipynb)
- 3. Unsupervised Learning
    - [3.1. Topic Modeling with Gensim](https://github.com/iceberg425/NLP_Projects/tree/master/Topic_Modeling_Gensim-District.ipynb)
    - [3.2. Topic Modeling with Sckit-learn](https://github.com/iceberg425/NLP_Projects/tree/master/Topic_Modeling_Sklearn_District.ipynb)
- 4. Modeling and Evaluation
    - [4.1. Preprocessing](https://github.com/iceberg425/NLP_Projects/tree/master/Preprocessing.ipynb)
    - [4.2. Neural Networks](https://github.com/iceberg425/NLP_Projects/tree/master/Predicting_Interest_Rates_with_Text.ipynb)
    - [4.3. Logistic Regression](https://github.com/iceberg425/NLP_Projects/tree/master/Predicting_Interest_Rates_with_Text.ipynb)

TL;DR

### Data Gathering

   The data was gathered from two sources:
   + [Federal Reserve Bank of Minneapolis](https://www.minneapolisfed.org/news-and-events/beige-book-archive)
       
   + [Macrotrends](https://www.macrotrends.net/2016/10-year-treasury-bond-rate-yield-chart)
       
   The first was scraped using Beautiful soup. We collected text documents from January 1970 up until April 2019, for each Federal Reserve bank district. The Beige Book summarizes this information by District and sector. An overall summary of the twelve district reports is prepared by a designated Federal Reserve Bank on a rotating basis. The final outcome is 5148 observations (documents). 
   The second was downloaded as a csv from [macrotrends](https://www.macrotrends.net/2016/10-year-treasury-bond-rate-yield-chart). 
   The availability of a timestamp each observation allowed us to map both datasets into a unified dataset. As pointed out above, due to the size of the data directory, we have excluded the file from the repo; however, the dataset is available upon request. 
   
### Data Exploration

   We explore the data at three levels. First, we explore a sample (the national summary). The national beige book is a summary of the economic conditions gathered for all 12 Fed districts. Then, we proceed to do a full exploration of the full text before embarking on the topic modeling and predictive modeling. Finally, we explore the quantitative data for missing data, its distributions, etc.
   
### 
    
    
   
   
   
   
   
   
   
   
   
   
   