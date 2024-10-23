import torch
import numpy as np

data =[[1,2],[3,4]]
x_data = torch.tensor(data)

#From a NumPy array
arr = np.array([[3,4],[2,6]],np.uint8)

print("Data: ",data)
print("Torch Data: ",x_data)
print("Numpy Data: ",arr)

#Converting Numpy to tensor
x_np = torch.from_numpy(arr)

print("Torch Numpy Data: ",x_np)

# It Retains the properties of X_data
x_ones = torch.ones_like(x_data)
print("X_Ones: ",x_ones)

#It Overrides the datatype of X_data
x_rand = torch.rand_like(x_data,dtype=torch.float)
print("X_rand: ",x_rand)

#shape is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.
shape = (2,3,)
rand_tensors = torch.rand(shape)
ones_tensors = torch.ones(shape)
zeros_tensors = torch.zeros(shape)

print(f"Rand Tensor: {rand_tensors}\nOnes Tensors: {ones_tensors}\nZeros Tensors: {zeros_tensors}")

print(f"Tensor type: ",rand_tensors.dtype)
print(f"Tensor Shape: ",rand_tensors.shape)
print(f"Tensor is stored on",rand_tensors.device)

tensor = torch.ones(4,4)
print("Rows: ",tensor[0])
print("Columns: ",tensor[:1])
print("Last Columns: ",tensor[...,-1])
tensor[:,1]=0
print(tensor)

#Joining tensors You can use torch.cat to concatenate a sequence of tensors along a given dimension.
#See also torch.stack, another tensor joining operator that is subtly different from torch.cat.

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

#Bridge with NumPy
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")