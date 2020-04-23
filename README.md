# uncertainty-estimation

## Monte Carlo Model
Here we are running multiple forward passes trough the model with a different dropout masks every time. For a given a trained neural network model 
with certain dropout to derive the uncertainty for one sample we collenct the predictions with different dropout masks.
And by computing the average and the variance of this sample we get an ensemble prediction, which is the mean of the models posterior distribution for this sample and an estimate of the uncertainty of the model regarding the sample.
To achieve this in keras, we have to use the functional API and setup dropout this way: `Dropout(p)(input_tensor, training=True)`.
