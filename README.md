# The project

The aim of this project is to build models that are able to recommend relevant tags to Stack Overflow issues.
I explored two learning methods :

## Supervised learning

Supervised models are built with a TF-IDF dataset built from a bag of words modeling
I implemented 3 of those models with cross validation :

- Logistic regression
- Support Vector Classifier
- Random Forest

### Results

The results of this first part are presented below :
![alt text](https://github.com/E-tanok/projects_pictures/blob/master/NLP/stack_overflow/supervised/results.png)

## Unsupervised learning

Unsupervised models are built from a bag of words modeling

I implemented 3 of those models :

- NMF
- LDA
- LDA + Word2Vec

Finally I built a process which allows to reccomment tags based on similarity matrices between words and LDA dictionaries which maps messages to main topics. Here is an example with the final model which combines LDA and Word2Vec embedding:

![alt text](https://github.com/E-tanok/projects_pictures/blob/master/NLP/stack_overflow/unsupervised/recommendation_process.png)

I compared the results obtained with this 3 unsupervised approaches thanks to a custom metrics of accuracy which integrate the relevance and the number of recommended tags :

![alt text](https://github.com/E-tanok/projects_pictures/blob/master/NLP/stack_overflow/unsupervised/accuracy_metric.png)

### Results

Accuracy in terms of tags relevance (LDA+Word2Vec wins):

![alt text](https://github.com/E-tanok/projects_pictures/blob/master/NLP/stack_overflow/unsupervised/results_p1.png)


Accuracy in terms of tags number (NMF wins):

![alt text](https://github.com/E-tanok/projects_pictures/blob/master/NLP/stack_overflow/unsupervised/results_p2.png)


Global accuracy (LDA+Word2Vec wins):

![alt text](https://github.com/E-tanok/projects_pictures/blob/master/NLP/stack_overflow/unsupervised/results_p3.png)



# The flask application :

The final model (LDA+Word2Vec) : allowed me to build a [flask application](http://bit.ly/mk_nlp_stack_flask_)

![alt text](https://github.com/E-tanok/NLP_flask_api_stackoverflow_tags_recommender/blob/master/project_instructions/results.png)
