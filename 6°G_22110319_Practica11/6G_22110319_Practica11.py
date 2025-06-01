import cv2  # Librería OpenCV para procesamiento de imágenes

# --------- Cargar imágenes ---------
img1 = cv2.imread('template.png', 0)  # Imagen plantilla (recorte que se desea encontrar)
img2 = cv2.imread('samus.png', 0)     # Imagen donde se realizará la búsqueda
# Ambas se cargan en escala de grises (0) para simplificar el análisis

# --------- Inicializar el detector ORB ---------
orb = cv2.ORB_create()
# ORB (Oriented FAST and Rotated BRIEF) detecta puntos clave robustos a rotación y escala

# --------- Detectar puntos clave (keypoints) y descriptores ---------
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
# kp1, kp2 = puntos clave en cada imagen
# des1, des2 = vectores que describen las características locales en torno a cada keypoint

# --------- Comparar descriptores con Brute-Force matcher ---------
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# NORM_HAMMING es la distancia utilizada para descriptores binarios como los de ORB
# crossCheck=True asegura que la coincidencia sea recíproca (más precisa)

matches = bf.match(des1, des2)
# Obtenemos una lista de coincidencias entre descriptores

# --------- Ordenar coincidencias por distancia (mejor a peor) ---------
matches = sorted(matches, key=lambda x: x.distance)
# Entre menor sea la distancia, mejor es la coincidencia entre características

# --------- Dibujar las mejores 20 coincidencias ---------
resultado = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)
# Se dibujan las líneas que conectan los puntos clave coincidentes entre ambas imágenes

# --------- Mostrar el resultado ---------
cv2.imshow('Similitudes con ORB', resultado)
cv2.waitKey(0)  # Esperar hasta que el usuario presione una tecla
cv2.destroyAllWindows()  # Cerrar la ventana mostrada
