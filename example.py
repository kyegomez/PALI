import torch  # Importing the torch library for tensor operations
from pali import Pali  # Importing the Pali class from the pali module

model = (
    Pali()
)  # Creating an instance of the Pali class and assigning it to the variable 'model'

img = torch.randn(
    1, 3, 256, 256
)  # Creating a random image tensor with shape (1, 3, 256, 256)
# The shape represents (batch_size, channels, height, width)

prompt = torch.randint(
    0, 256, (1, 1024)
)  # Creating a random text integer tensor with shape (1, 1024)
# The shape represents (batch_size, sequence_length)

output_text = torch.randint(
    0, 256, (1, 1024)
)  # Creating a random target text integer tensor with shape (1, 1024)
# The shape represents (batch_size, sequence_length)

out = model.forward(
    img, prompt, output_text, mask=None
)  # Calling the forward method of the 'model' instance
# The forward method takes the image tensor, prompt tensor, output_text tensor, and an optional mask tensor as inputs
# It performs computations and returns the output tensor

print(out)  # Printing the output tensor
