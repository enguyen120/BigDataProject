Nothing to report.
https://medium.com/@enguyen_44217/title-e4bd7db26932 
Could you recognize an article from the New York Times without the source in front of you? What if you only had the words? And if you only had the words, would you be able to distinguish an article from a left-leaning news source from a right-leaning one?
Our team decided to dive into how distinct word choice is in different newspapers using data algorithms - machine learning and natural language processing - to predict the political leaning of a source. We wanted to locate differences and biases of news sources on the political left and right stemming from the words they use and the sentiment that those words convey. Ultimately, our goal was to create concise, informative data visualizations to answer our guiding question: do left-leaning news sources use different words, with different sentiments, than right-leaning sources? 
Though we did not find a clear answer, in the process, we familiarized ourselves with big data tools, the process of cleaning, analyzing, and drawing conclusions from data, and -processing, and made some surprising discoveries about our initial dataset. 


Section 1: Cleaning the Data and Getting a “Bag of Words”
Our first step was to choose a dataset. This was harder than it initially seemed. Although many news sites do have open APIs (interfaces allowing users to access site data), most of them only give users access to the metadata about their articles—not the actual text. The same is true of most online datasets. One exception is the “All the News” dataset, compiled in early 2017 and uploaded to the data collection website Kaggle by Andrew Thompson (found here). (For readers who want to learn more, there is also an expanded version of the dataset with over 2.7 million articles, found here.) This dataset includes the full text of articles from a variety of US news sources, as well as additional information, like the article’s publisher, the date published, the title, and more: perfect for our analysis. 
Next, in order to rate the political leaning of the different sources included, we found a public dataset on GitHub (posted by nsfyn55, found here) of information scraped from Media Bias / Fact Check, a trusted site rating political bias in news media. We had to clean this dataset in order to make it applicable for our project. The dataset originally combined the bias with the source’s ID (for example, the eighth left-center-leaning source was labeled “leftcenter8”), and had more categories of bias than we needed ('left', 'extreme left', 'left center', 'least-biased', 'right center', 'right', and 'extreme right'.) We slimmed this down to a simple, numerical ordering: 0 for any right-leaning source, regardless of extremity, 1 for any left-leaning source, and 0.5 for “least biased” sources. 
Another important step we took was mapping the news sources in the “All the News” dataset to the biases in the MB/FC dataset. We edited the names of the news sources so that they matched the publisher names in our “All the News” dataset—changing “Guardian” to ”The Guardian,” and so on. Using this, we were able to effectively add a new column to the “All the News” dataset: based on the publisher’s name, the new column had a 0, 0.5, or 1 corresponding to that publisher’s bias. 
The most important step, however, was creating a TF-IDF “bag of words” matrix. A “bag of words” is a model used in natural language processing to count how many times a given word appears in a piece of text. TF-IDF (“term frequency–inverse document frequency”) is a statistical transformation that gives a word a larger value based on how ‘important’ it is. For example, the word “the” appears many times in every article, but it does not affect the meaning very much. TF-IDF would therefore reduce the value of the word “the.” The name “Zuckerberg,” on the other hand, probably would not appear that many times throughout all the articles—but it would significantly affect the meaning of the articles where it did appear. “Zuckerberg,” then, would have a high value in articles where it was used many times. There are premade tools in the SciKitLearn library which can extract a TF-IDF bag of words from a dataset like ours.
In the process of trying to run these tools on the “All the News” dataset, we ran into some problems: we quickly ran out of available memory on Google Cloud. To solve this, we decided to clean up our dataset of article texts. We removed “stop words”—words like “a,” “an,” or “the,” which are used for grammar, not meaning. We used a lemmatizer—a program which breaks down words into a simpler root, known as a “lemma,” thus reducing the number of similar words (so “knife” and “knives” would be considered the same word). We also chose to use only articles from one month. We distinctly chose the month of November, 2016, because of the presidential election that occurred during the time in America. In addition, we added extra preprocessing and cleaning in order to eliminate non-English words, as well as any rare words that were used less than ten times overall. These rare words would have no effect on analysis, and, unfortunately, made the bag of words so large that it broke our computers. In this way, we were able to narrow down the number of words to form a better picture of the most commonly used words and pinpoint which news outlets used them. 

Section 2: General Analysis
Our first piece of analysis did not use the TF-IDF bag of words at all. Instead, we made a word cloud of the top 100 most common words that appeared in news articles in November, 2016. The top words were ‘Trump,’ ‘Clinton,’ ‘said,’ and ‘state’. For the left leaning articles the word cloud looked as such: 

For the right leaning articles the word cloud looked like this: 

Lastly, for the articles that were in the center, the word cloud looked like this:

All three word clouds look very similar, and this is not very surprising. At the time that the coverage was scaled from, the media should’ve been discussing presidential elections, and more. What is most interesting is that only two out of the three word clouds mention Clinton, but all include Trump, or Donald Trump. Of course, most of November was after Trump’s presidential victory on the 5th. Nevertheless, this shows just how large a range Trump’s media coverage spanned, covering the top words in almost all the articles. 

Once we had the bag of words and the political bias of our articles, we could do some more in-depth examination of our data. Our distribution of political leanings looked like so:
RIGHT: 2700 articles (33.81%)
LEFT: 4453 articles (55.77%)
CENTER: 832 articles (10.42%)
We quickly see that our data is not equally spread between the three political leanings—there are more left-leaning articles than right-leaning, and more of either than of center-leaning articles. This affects our strategy for our machine learning section (more on that later.)

Similar to the wordclouds, we could also check the most important words overall—that is, the words with the highest overall TF-IDF scores in our dataset. There were:
10 most important: ['trump', 'said', 'clinton', 'state', 'people', 'election', 'year', 'president', 'new', 'say']
10 least important: ['tolerable', 'compulsion', 'zeroed', 'herring', 'pettiness', 'vindictiveness', 'subsumed', 'unproductive', 'preface', 'perpetuates']
	This was very similar for left-leaning sources:
10 most important: ['trump', 'said', 'people', 'clinton', 'state', 'mr', 'say', 'election', 'year', 'like']
10 least important: ['ul', 'vantiv', 'volz', 'walcott', 'warnerthuston', 'worldpay', 'xiaomi', 'yashaswini', 'yt', 'zieminski']
right-leaning sources:
10 most important: ['trump', 'clinton', 'said', 'election', 'state', 'people', 'breitbart', 'news', 'president', 'obama']
10 least important: ['yearned', 'yeats', 'yonhap', 'yt', 'yuan', 'zadie', 'zengerle', 'zeroed', 'zieminski', 'zooming']
and center-leaning sources:
10 most important: ['trump', 'said', 'people', 'clinton', 'state', 'mr', 'say', 'election', 'year', 'like']
10 least important: ['ul', 'vantiv', 'volz', 'walcott', 'warnerthuston', 'worldpay', 'xiaomi', 'yashaswini', 'yt', 'zieminski']
Due to the natural inexactitude of natural language processing, some of the precise importance data might be off. For example, the high importance of the word “Breitbart” in right-leaning sources, but not any other political leaning, may suggest not that right-leaning sources talked about Breitbart that much more than anyone else, but rather that the content of Breitbart articles (all of which come from a right-leaning source) had some kind of reference to their source. 
Some things, however, are interesting. Both left- and center-leaning sources have the word “Mr” among their top 10, but not “president”; right-leaning sources have “president,’ but not “Mr.” We cannot prove any causation given the data that we have, but there is a temptation to read into this a tendency for left-leaning sources refer to the newly-elected Trump as “Mr. Trump,” and or right-leaning sources to use “President Trump.” This may be an interesting avenue of future research, but was outside the scope of this project. 
Generally, however, the most important words seem very reasonable, without anything particularly unexpected.

Finally, we also took a look at the TF-IDF distributions of some particular words we were curious about. First, there is the word “Trump”: the most common word in our dataset.

Setting aside differences in the overall number of articles in the dataset, the left- and right-leaning sources have similar graph shapes, but center-leaning sources have a slightly different shape. Without doing a check for statistical significance, it visually seems that the word “Trump” was marginally less common in center-leaning sources. 
	Our third most common word was “Clinton.” 

The distributions again look very similar. Without checking for statistical significance, there does not seem to be a difference in how important the word “Clinton” was to articles within the three different groups. 

Section 3: Sentiment Analysis
One of the first tasks we completed was finishing the sentiment analysis on our data frame. We created a dataframe that included data of our article IDs, titles, publications, contents, and “leftness” (our measure of the publisher’s political bias.) Using this dataframe, we could do sentiment analysis on the different articles, and compare the articles’ sentiment to the publisher’s political leaning. We concluded with nine different graphs which showed the distribution of sentiment-laden words (positive, negative, or neutral) in articles from the left, the right, and the center. Essentially, we determined the connotation of each word in each article, determining if the article is truly positive or negative in sentiment. Our fundamental purpose was to see if there was any difference in how the right and left used positive or negative words. 
There were some interesting finds. The median lines, and the overall structure of the first 6 graphs were very similar. It seems that there is a similar distribution of sentiment-laden words across left and right political biases, or at least in the articles that we analyzed. For the “mixed” or in the middle-leaning articles, there was a pretty neutral depiction of the graphs. It seems that the bar is more often on the left side than the right (showing a lower mean in the data), but overall there was not much difference between the three graphs other than the median line placement. This mean line placement indicates that most articles did not have a lot of words that indicated the political sentiment and that there were more neutral words. Overall, the sentiment analysis points towards a bias of neutral words. 



Section 4: Machine Learning Analysis
Section 4A: Logistic Regression
Another one of our methods of analysis was machine learning. We chose to use logistic regression and random forests for our analysis. 
Logistic regression is a parametric method (meaning it has a designated input and output) that can be used to find the probability of an outcome based on the cumulation of predictors, or variables. In our case, it uses all the possible words in an article to predict if the article’s publisher was left, center or right leaning based on the data from Media Bias / Fact Check. 
In our analysis, we used stratified k-fold cross validation. This means that we divided our data into groups called “folds,” with an equal number of left- and right-leaning articles in each fold. This lets us account for the fact that our dataset didn’t have an equal number of left- and right-leaning articles. (Unfortunately, the Sci-Kit Learn implementation of stratified k-folds can only divide the data based on a binary variable—in this case, rightness or leftness. We therefore removed all center-leaning articles for the purposes of machine learning.) 
In general, cross validation is a method used to evaluate how accurate a model is by randomly splitting a data set into groups that are called folds, the number of which is given by k, which in our case was 10. For each fold, the data is divided randomly into a training set and test set, with an equal amount of cases from each category per group: the model is trained on k-1 folds (in our case, 9) and tested on the remaining one. Cross validation then averages the k-metrics from each of the k-iterations. 
The training data is used to train the model to make predictions about the data. The test set is used to see how well the model works on data it has not seen before, allowing us to test its accuracy. The resulting evaluation metrics from cross validation are averaged over all of the folds to get a result that encapsulates all the folds and tests that were done.
Our logistic regression had an overall accuracy of 81.12%. This means that our model can correctly sort whether an article is from a right-leaning or left-leaning source, based on the text of the article alone, around 80% of the time. 
We checked the accuracy further by creating a “confusion matrix”—a graph which shows how often the model was correct when it predicted that an article fell into a certain group.

We can see that when our model predicted that an article was left-leaning, it was correct around 80% of the time (415 correct / 520 predictions). If the model predicted that an article was right-leaning, it was correct about 85% of the time (165 correct / 195 predictions).
Generally, if an article was actually left-leaning, it had a 93% chance of being sorted correctly by the model, and if it was actually right-leaning, it had a 61% chance of being sorted correctly.

Once we had our model, we checked for feature importance: how important was each particular word in deciding whether the article as a whole is from a right-leaning or left-leaning source? Large negative feature importances mean that the word was used to predict a right-leaning article, and large positive importances mean that the word was used to predict a left-leaning article.
Most important 10 features
    ('breitbart', -5.306806982034431)
    ('follow', -4.64292371525119)
    ('fox', -3.7617111683186772)
    ('cnn', 3.7045138017588015)
    ('prediction', -3.3611188336224194)
    ('percent', -3.3452453305138596)
    ('mr', 3.0438166852549045)
    ('npr', 2.9099418760226596)
    ('twitter', -2.703878301030436)
    ('map', -2.70057515081719)

We also generated a graph to show our feature importance below. In this figure you can see that most words had around the same effect on making an article more left or right leaning, with only a few stand-out exceptions.


The results of the model’s feature importance are quite interesting, and suggest a number of fascinating further questions.
Many of the most important words are the names of news sources with a bias. Generally, the more a publication’s name was used, the more likely it was for that article to have the same bias as that publication. ‘Breitbart’ and ‘Fox,’ two right-wing sources, have large negative coefficients, meaning that the model uses these words to predict that a source is right-leaning. Similarly, “CNN” and “NPR”, two left-leaning sources, have large positive coefficients, meaning the model uses these words to predict that a source is left-leaning. Perhaps publications tend to cite publications with the same political leaning. Or perhaps the article content from those sources had ‘clues’ to their real sources (e.g., “our reporters from CNN,” or similar phrases.) 
References to the social media site Twitter (the word ‘follow’ and ‘twitter’) seem to be associated with right-leaning news media—definitely something that could merit further investigation. Also, the apparently-innocuous word “Mr” is associated with left-leaning articles, which could be due to a wide variety of reasons. 
Overall, logistic regression for the most part seems to accurately predict the chances of an article being left leaning with a few exceptions. However, since the ability to predict news sources is not the same for left and right leaning sources, and did not take into account the neutral articles that we had in our data set, interpreting how these results are indicative of border news trends must be taken with a grain of salt. Our analysis of course cannot affirmatively answer any questions of causation, but it indicates many tempting avenues of further research.

Section 4B: Random Forests
Our next method of machine learning, random forests, creates a large average of which words fall into which categories based on a method called decision trees. 
Decision trees can be created in large datasets with thresholds of categorical variables. The model splits the data based on whether it falls or contains a category at thresholds. From there, the model is able to sort the data into categories, splitting it on multiple requirements and thresholds until the data is pure—or, in other words, all the data at the ends of the trees are in groups with the same qualities. In our case, a tree would split all articles into “pure” groups of only left-leaning articles and only right-leaning articles.
Random forests are used to avoid overfitting data and having one, single overly complex decision tree. This is done by boot-strapping, or taking many random samples of data and running the decision tree algorithm, and then replacing the articles in the larger dataset and taking the average of the trees in the final outputs. We chose random forests because we wanted to see whether, if we split the data on the most common words used in an article, if it would be able to determine what the political leaning of the publication was. To do this we used the random forest package sk.learn and our testing/training data from the previous part. 
With cross validation, our random forest had a 77.76% accuracy, meaning that from the most common words in an article, we are able to determine the political leaning. 
To examine the accuracy of our model we used a confusion matrix, which shows us the number of articles the algorithm believes to be left or right leaning versus the articles that actually fall into those categories.

For our random first model, it was able to predict an article was left-leaning correctly around 75% of the time (431/ 576 predictions). For right leaning articles, it was correct in its prediction 90% of the time (125/139 predictions. Generally for this model, if an article was actually leaning,, it was correct in soriting it 96% of the time and if the model predicted an article was right leaning, it was correct in sorting in 53% of the time.  
Below is also a list of the words that contributed the most to the splits of the data to be left or right leaning. In random forests, we look at how important a word is to making a split, so there are no positive or negative numbers—only a measure of how well it splits the data. This is a similar selection of words to our logistic regression, which is indicative of what the political climate was in November of 2016. 

Most important 10 features
	('follow', 0.010847651300438809)
	('breitbart', 0.010787164083514428)
	('twitter', 0.007622907518721526)
	('cnn', 0.004558850717966741)
	('fox', 0.004025414422017306)
	('prediction', 0.003713874501682345)
	('like', 0.003279086794815597)
	('time', 0.003229275830119046)
	('projection', 0.0031398204126973465)
	('map', 0.003127812074233638)


Section 5: Conclusion 
	So as we alluded to in some of the above sections of the blog post there are a few key takeaways from this exploration of news data. First and foremost, there is in fact a correlation between the frequency of particular words in a news article and its political leaning, from our random forest model and the decision tree model. From our confusion matrices we can tell that both models had a higher success rate in predicting whether an article was right leaning if it had predicted it to be right wing but was more likely to be correct sorting left wing articles into the right category. Our sentiment analysis tells us that articles contain more neutral words than words that are incredibly negative or positive, but that there is a presence of words that are indicative of political leanings. Most notably, there were a lot of words relating to Twitter that were commonly used in the articles. ‘Trump’ and ‘Donald Trump’ were two of the most popular used words in the articles,  regardless of the political leaning of the source. 
The words that articles use to describe events are so important in shaping the reader’s view, and our analysis indicates that people are getting completely different versions of events based on the news outlets that they read. This is an important result because it highlights the polarization of the American media environment and how people can have misunderstanding of current events, contributing to the extent people do not see eye to eye in the U.S. Indeed, this information has been suspected, especially after the pinnacle 2016 presidential election. However, our data has given some evidential light to the speculated claims that the media has provided an influx of diverse opinions, all creating extreme viewpoints to cater towards an audience that is captured by clickbait, and more. 
