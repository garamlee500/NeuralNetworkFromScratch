from math import exp
from typing import Callable
import torch


class Network:
    def __init__(self) -> None:
 
        self.minibatch_size = 0
        self.layers=[]

    def __repr__(self) -> str:
        return f"Neural network with {len(self.layers)} layers.\n"+\
            f"Has layers: {"".join("\n" + layer.__repr__() for layer in self.layers)}"

    def calculate(self, inputs: torch.Tensor):
        current_output = inputs
        for i in range(len(self.layers)):
            current_output = self.layers[i].calculate(current_output)
        
        return current_output

    def update_weights(self, learning_rate):
        for k in range(len(self.layers)):
            # Want to use mean gradient as this makes learning rate consistent across batch size
            self.layers[k].update_weights(learning_rate/self.minibatch_size)

            # reset gradients
            self.layers[k].gradients.zero_()
        self.minibatch_size = 0


    def calculate_with_gradient(self, inputs: torch.Tensor, target: torch.Tensor, loss_function=torch.nn.CrossEntropyLoss()):
        
        # Calculate another gradient
        self.minibatch_size+=1


        self.calculate(inputs)

        # I could derive the gradient for each loss/activation function mathematically but that wouldn't be in spirit of this project
        # (I already know how to find the gradient of a function by hand)

        yd = self.layers[-1].last_input @ self.layers[-1].weights
        yd.requires_grad= True
        loss = loss_function(self.layers[-1].activation_function(yd), target)
        loss.backward()
        dLossdyk = yd.grad


        # Process gradients for weights on each layer
        # See sgd.md for derivation

        # The input to the ith layer is x^(i) and the output is x^(i+1)
        # And has weights W^{(1)}
        # And has activation function f_{k}
        for k in range(len(self.layers)-1 , -1, -1):
            self.layers[k].gradients += torch.transpose(self.layers[k].last_input,0,1) @  dLossdyk


            # Get gradient of ReLu/activation
            # Again this would be pretty (really) straightforward to derive by hand

            # y ^((k-1))
            if k > 1:
                ykm1 = self.layers[k-1].last_input @ self.layers[k-1].weights
                ykm1.requires_grad = True
                result = self.layers[k-1].activation_function(ykm1)
                result.backward(torch.ones(ykm1.shape))
                activation_grad = torch.transpose(ykm1.grad, 0 , 1)

                # Last row of weights only affected by bias node so not in this derivative
                dLossdyk =  dLossdyk @ torch.transpose(torch.mul(activation_grad,self.layers[k].weights[:-1]), 0,1 ) 


            
            







    

class Layer:
    def __init__(self, input_length: int, output_length: int, activation_function: Callable[[torch.Tensor], torch.Tensor] = torch.relu):
        # self.values = torch.zeros(output_length)
        self.activation_function = activation_function
        self.input_length = input_length
        self.output_length = output_length

        
        self.weights = torch.empty(input_length+1, output_length)
        # TODO: UNDERSTAND THIS
        # (basic randomness seems sensible)
        torch.nn.init.normal_(self.weights, 0, 2/input_length)

        # Clearly I only need one of these really but these make it easier to handle
        self.last_input = torch.empty(input_length+1)
        self.last_output = torch.empty(output_length)

        # Store gradient (temporarily) for sgd
        # Is gradient of loss respect to weights
        self.gradients = torch.zeros(input_length+1, output_length)

    def calculate(self, prev_layer:torch.Tensor):
        # Add bias node
        # Doesn't really affect maths as gradients only depend on weights
        self.last_input = torch.cat((prev_layer, torch.tensor([[1.0]])),1 )
        self.last_output = self.activation_function(torch.mm(self.last_input, self.weights))
        return self.last_output.clone()


    def __repr__(self) -> str:
        return f"Layer with {self.input_length} inputs and {self.output_length} outputs.\n"+\
            f"{self.weights}"
    

    def update_weights(self, learning_rate):
        self.weights = self.weights-self.gradients*learning_rate




if __name__ == "__main__":
    xor_network = Network()
    xor_network.layers.append(Layer(2, 4, torch.sigmoid))
    xor_network.layers.append(Layer(4, 4, torch.sigmoid))
    xor_network.layers.append(Layer(4, 4, torch.sigmoid))
    xor_network.layers.append(Layer(4, 2, torch.nn.Identity()))

    inputs = torch.tensor([[[0.0,0.0]], [[0.,1.]], [[1.,0.]], [[1., 1]]])
    outputs = torch.tensor([[0], [1], [1], [0]])
    import random
    for i in range(100):
        j = random.randint(0,3)
        xor_network.calculate_with_gradient(inputs[j], outputs[j])
        xor_network.update_weights(0.1)


    print (xor_network)