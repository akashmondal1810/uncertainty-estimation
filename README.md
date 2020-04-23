# uncertainty-estimation

## Monte Carlo Model
Here we are running multiple forward passes trough the model with a different dropout masks every time. For a given a trained neural network model 
with certain dropout to derive the uncertainty for one sample we collenct the predictions with different dropout masks.
And by computing the average and the variance of this sample we get an ensemble prediction, which is the mean of the models posterior distribution for this sample and an estimate of the uncertainty of the model regarding the sample.
To achieve this in keras, we have to use the functional API and setup dropout this way: `Dropout(p)(input_tensor, training=True)`.

## Ensemble for Deep Learning Neural Networks
This is an extension of a model averaging ensemble where the contribution of each member to the final prediction is weighted by the performance of the model. In our case i.e. predicting a class probability, the prediction has been calculated as the `argmax` of the summed probabilities for each class label.
Here the value for the weights has been estimated using either holdout validation dataset unseen by the ensemble members during training. For searching the weights we are using the `differential_evolution()` SciPy function by minimizing the classification error (1 – accuracy).
