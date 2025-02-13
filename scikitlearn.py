from sklearn import datasets
from matplotlib import pyplot as plt 

digit = datasets.load_digits()
data = digit.data
print(data.shape)
plt.imshow(digit.images[-1], cmap='gray')
plt.show()