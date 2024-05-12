import numpy as np
import matplotlib.pyplot as plt


batman = np.array([[0, 0], [1, 0.2], [0.4, 1], [0.5, 0.4], [0, 0.8], [-0.5, 0.4], [-0.4, 1], [-1, 0.2], [0, 0]])
triangle = np.array([[2, 4], [1, 1], [3, 1], [2, 4]])
rectangle = np.array([[1, 3], [1, 1], [4, 1], [4, 3], [1, 3]])
x = triangle[:, 0]
y = triangle[:, 1]

plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o', linestyle='-')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.axis('equal')
plt.show()




