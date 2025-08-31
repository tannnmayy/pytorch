import numpy as np
array_1d = np.arange(16)
print(array_1d)
print(array_1d.shape)
array_2d = np.array([[1, 2, 3], [5, 56, 4]])
print(array_2d.shape)
array1= np.array([1,2,3,4])
array2= np.array([3,4,5,5])
result_array= array1 +array2
print(result_array)
reshaped_array= array_1d.reshape(4,4)
print(reshaped_array)
reshaped_array2= array_2d.reshape(3,2)
print(reshaped_array2)
identity_matrix= np.eye(10)
print(identity_matrix)


