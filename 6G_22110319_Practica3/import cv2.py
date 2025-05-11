import cv2
import numpy as np

# Cargar imagen en escala de grises
img = cv2.imread('samus.png', cv2.IMREAD_GRAYSCALE)

# ------------------------ Operaciones Aritméticas ------------------------

# 1. Suma (brillo aumentado)
img_suma = cv2.add(img, 50)

# 2. Resta (oscurecer)
img_resta = cv2.subtract(img, 50)

# 3. Multiplicación (contraste aumentado)
img_mult = cv2.multiply(img, 1.5)

# 4. División (reducir intensidad)
img_div = cv2.divide(img, 1.5)

# 5. Negación (inverso)
img_neg = cv2.bitwise_not(img)

# 6. Transpuesta (gira imagen en diagonal)
img_trans = cv2.transpose(img)

# 7. Aumento de tamaño
img_ampliada = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

# 8. Reducción de tamaño
img_reducida = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

# 9. Rotación (45 grados)
(h, w) = img.shape
center = (w // 2, h // 2)
matriz_rot = cv2.getRotationMatrix2D(center, 45, 1.0)
img_rotada = cv2.warpAffine(img, matriz_rot, (w, h))

# 10. Traslación (mover 50px derecha, 30px abajo)
M = np.float32([[1, 0, 50], [0, 1, 30]])
img_traslada = cv2.warpAffine(img, M, (w, h))

# ------------------------ Mostrar resultados ------------------------
cv2.imshow('Original', img)
cv2.imshow('Suma', img_suma)
cv2.imshow('Resta', img_resta)
cv2.imshow('Multiplicación', img_mult)
cv2.imshow('División', img_div)
cv2.imshow('Negación', img_neg)
cv2.imshow('Transpuesta', img_trans)
cv2.imshow('Ampliada', img_ampliada)
cv2.imshow('Reducida', img_reducida)
cv2.imshow('Rotada', img_rotada)
cv2.imshow('Trasladada', img_traslada)

cv2.waitKey(0)
cv2.destroyAllWindows()
