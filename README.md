## **Steps to build NN using Gradient descent**

* Define Network structure i.e No of layers, No of neurons in each layers, type of activation function in each layers, no of epochs
* Random initialization of weights(W) and bias(b) for each layers
* repeat for no of epochs
	* Calculate Forward propogation in 2 steps
		* linear calculation  Z = WX + b
		* activation function A = g(Z)
	* Calculate cost
	* Calculate back propogation in 2 steps
		* Back propogation for activation part dA
		* Back propogation for linear part which compute dW, db
	* update parameter W and b
* predict accuracy

## **Steps to build NN using Batch Gradient descent**

* Define Network structure i.e No of layers, No of neurons in each layers, type of activation function in each layers, no of epochs, batch   size
* Random initialization of weights(W) and bias(b) for each layers
* Divide training data into mini batches
* repeat for no of epochs
	* repeat for each mini batch
		* Calculate Forward propogation in 2 steps
			* linear calculation  Z = WX + b
			* activation function A = g(Z)
		* Calculate cost
		* Calculate back propogation in 2 steps
			* Back propogation for activation part dA
			* Back propogation for linear part which compute dW, db
		* update parameter W and b
* predict accuracy


