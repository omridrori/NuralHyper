# NuralHyper


## The basic idea
The basic idea is to use neural network to take dataset and use it to find best hyperparameters fast.\
Here, I will only refer to decision trees, but the basic idea can be applied to anything.


![Flowchart Template (1)](https://user-images.githubusercontent.com/59512233/204343939-60abbce5-cb32-4ec7-94ba-ea735cb85d4c.jpg)


## Not working ideas
Predicting the best hyperparameters themselves doesn't work whether we use real data or idenifiters.

### First idea doesnt work 
![Flowchart Template (2)](https://user-images.githubusercontent.com/59512233/204521818-4e5efe50-983d-49a4-a30e-a6b0f50eb4b4.jpg)

### Second try that dosent work
![NOT WORK2](https://user-images.githubusercontent.com/59512233/204521552-5d20fe0d-f699-4794-aef7-962bb9bdadd5.jpg)


## What is do working in high level:

Therefore, directly estimating hyperparameters from data does not work. 

However, predicting **accuracy** for a **specific** configuration of hyperparameters on this data is pretty accurate. 

namly, if we ask the network given data and specific configuration of hyperparameters what would be the accuracy of a model with this configuration of hyperparameters on this data, it predicts it very close.

Then we can use this neural network to do very fast Grid Serach in which in each iteration we predict the accuracy for specific configuration with only one feedforward computation and without the need to run the actuall model (decision tree) on the data from scratch.n

![working](https://user-images.githubusercontent.com/59512233/204522321-55a69ad1-9673-43a4-93bc-b3fa265ba510.jpg)

### What is the identifiers we use for the data:
Using the data itself cause a lot of problems because various datasets has variaty in number of features and examples so it every time change the input size for the network.\
So we have to use idenifires for the data instead of using the data itself. 
So first try which doesnt work was to use statistics for the data like mean,std,etc...
![not working image id](https://user-images.githubusercontent.com/59512233/204526208-539208ce-ed70-4e3c-8ffb-c1708edc6442.jpg)

So in second try, i tried to use specific combination of simple models to describe the data. 
The basic idea was that for example if knn with 3 neighbors work poorly on the data but knn with 8 neighbors run better so that say something on the topology of the data. \
So we use simple models and how they run on the data to tell the "story" of the data.
Here i used simple models which i chose very randomly but i think the biggest improvement of this mechanisem will come with better choosing of models to describe the data.
And the basic flow goes like that:
We start with data, we use something like 10 models (which are the same every time) we run these models on the data and we get accuracy for each model on the data so we get a vector of 10 numbers (the accuracies of each model on the data) and then this is the identifier of the data.
We take this identifier vector and combinitation of hyperparameters that we want to check how decision tree with this configuration of hyperparameter will run on this data and concatenate them together (the identifiers and the combination of hyperparameters) and this is the input for the neural network. 
Then the network give prediction what will be the accuracy of a decision tree model with this configuration of hyperparameters on the data.
So if the network is small this can be very fast prediction on how the model will work with this configuration of hyperparameters instead of running actually decision tree on the model.
![working correct](https://user-images.githubusercontent.com/59512233/204525609-97a98432-dbe4-4fad-a19c-271868c7ee51.jpg)

## Training
The very basic question to ask is how do we train such network, because this will require very big dataset in which each example is dataset in itself, that is we need to get a lot a lot of datasets which is very hard to get. 
So the way is solved it is by the observation that the network itself never sees the data itself it always sees the identifiers of the data which as i explained they are how simple models run on the data. But actually the data itslef it never see. 
So the big observation is that i can  actually randomly sample data from random distributions, namly i have two levels of randomness:
1. randomly sample parameters for creating data. 
2. the randomness of the sampling data with these parameters from the first step.

So i use the make_classification function from sklearn which create artifically data given parameters like n_samples,n_features,n_clusters_per_class etc.. 
So here i first sample parameters for this function and then i used the parameters to actually create the data.
Turns out that it generlize well to real data and not just artificially data. I guess that because the simple models tell something general like on the topolgy of the data.

So the flow goes like that: 
1. sample parameters for creating data
2. create data given these parameters 
3. find id for the data by running these simple models on the data you created
4. randomly choose configuration of hyper parameters to test
5. concatenate the id from step 3 and the configuration of hyperparameters from step 4 and pass it to the neural network
6. get prediction from the neural network how the decision tree will run on the data with this configuration of hyper paramters.
7. in parallel use the data you sampled from step 2 and use real decision tree model on configured with hyper paramters you chose in step 4 get the real accuracy.
8.compute the loss by comparing the real accuracy from step 7 to the predicted accuracy from step 5 and backprop this loss. 

![training](https://user-images.githubusercontent.com/59512233/204536804-0debd79f-86f1-4760-8676-b7d8848a4fa2.jpg)


## The benefits
So now after training you get neural network that can predict how decision tree models configured with specific hyper parameters configuration will run on the data instead of really running the model which take much more time. 
Then you can do grid search using this network to find the best hyper parameters. 




##results:
So one can see the test i did in the folder tests and see it work very well and much faster.
For example on Adults data from kaggle tried to find the best configuration of hyper parameters from the specified hyper parameter space with grid search and i took almost a day to tell the best configuration which gave 0.8571867321867321 accuracy. 
![image](https://user-images.githubusercontent.com/59512233/204535470-7e8afe57-5674-44a3-86de-5caeea52738c.png)

But with my algorithm it took only **3 minutes** to tell the best configuration of hyper parameters with only 0.003 less in accuracy from the best hyper parameters found with grid search. 
![image](https://user-images.githubusercontent.com/59512233/204536155-21f68aa2-7581-4a42-bc70-38f82558e091.png)

