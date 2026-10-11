import numpy as np
import ast
from neuron import Neuron
import activations

inputs = np.array(ast.literal_eval(input("Enter inputs: ")))

activation_func = {
    1: activations.step_function,
    2: activations.sigmoid,
    3: activations.relu
    }
func_key = int(input("Enter 1 for step function, 2 for sigmoid function, 3 for relu function: "))

cell = Neuron(len(inputs), activation_func[func_key])

print("Output: ", cell.forward(inputs))