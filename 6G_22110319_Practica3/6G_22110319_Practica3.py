import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen original
img = cv2.imread('samus.png', cv2.IMREAD_GRAYSCALE)

# --- Imagen modificada desde práctica 2 (Rotada) ---
(h, w) = img.shape
center = (w // 2, h // 2)
matriz_rot = cv2.getRotationMatrix2D(center, 45, 1.0)
img_modificada = cv2.warpAffine(img, matriz_rot, (w, h))

# --- Histograma original ---
hist_original = cv2.calcHist([img_modificada], [0], None, [256], [0, 256])

# --- Ecualización ---
img_eq = cv2.equalizeHist(img_modificada)

# --- Histograma ecualizado ---
hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256])

# --- Mostrar resultados con matplotlib ---
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.imshow(img_modificada, cmap='gray')
plt.title('Imagen Modificada (Rotada)')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.plot(hist_original, color='black')
plt.title('Histograma Original')
plt.xlim([0, 256])

plt.subplot(2, 2, 3)
plt.imshow(img_eq, cmap='gray')
plt.title('Imagen Ecualizada')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.plot(hist_eq, color='black')
plt.title('Histograma Ecualizado')
plt.xlim([0, 256])

plt.tight_layout()
plt.show()
