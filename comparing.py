import part1 as p1
import part2 as p2
import numpy as np
import matplotlib.pyplot as plt

triangle = np.array([[2, 4], [1, 1], [3, 1], [2, 4]])
rectangle = np.array([[1, 3], [1, 1], [4, 1], [4, 3], [1, 3]])

changed_figure, changed_figure_matrix = p1.rotate(triangle, 45)
changed_figure_cv2 = p2.rotate_cv2(triangle, 45)


plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.title('Original Figure')
plt.plot(triangle[:, 0], triangle[:, 1], 'b.-', label='Original')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)


plt.subplot(1, 3, 2)
plt.title('Rotated')
plt.plot(changed_figure[:, 0], changed_figure[:, 1], 'r.-', label='Changed')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.title('using CV2')
plt.plot(changed_figure_cv2[:, 0], changed_figure_cv2[:, 1], 'g.-', label='by CV2')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()