"""
HISTOGRAMA


from matplotlib import pyplot as plt
import cv2
img= cv2.imread('way.jpg')
img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#converte P&B
cv2.imshow("ImagemP&B", img)
cv2.imwrite("way_pb.jpg", img)


#Função calcHistpara calcularo hisogramadaimagem
h = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.figure()
plt.title("HistogramaP&B")
plt.xlabel("Intensidade")
plt.ylabel("Qtdede Pixels")
plt.plot(h)
plt.xlim([0, 256])
plt.show()
cv2.waitKey(0)

"""

"""

#HISTOGRAMA CLAHE

import cv2 as cv
img = cv.imread('way.jpg',0)
# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
#cv.imwrite('histogramaclahe.jpg',cl1)
cv.imshow('Não equalizada', img)
cv.imshow('Equalizada', cl1)
cv.waitKey(0)

"""


"""

#SUAVIZAÇÃO GAUSSIANA

import cv2
import numpy as np
img = cv2.imread('histogramaclahe.jpg')
img = img[::2,::2] # Diminui a imagem
suave = np.vstack([
np.hstack([img,
cv2.GaussianBlur(img, ( 3, 3), 0)]),
np.hstack([cv2.GaussianBlur(img, ( 5, 5), 0),
cv2.GaussianBlur(img, ( 7, 7), 0)]),
np.hstack([cv2.GaussianBlur(img, ( 9, 9), 0),
cv2.GaussianBlur(img, (11, 11), 0)]),
])
cv2.imshow("Imagem original e suavisadas pelo filtro Gaussiano", suave)
#cv2.imwrite('SuaveGaussiana.jpg', suave)
cv2.waitKey(0)


"""

"""

#FILTRO LAPLACIANO

import numpy as np
import cv2
img = cv2.imread('SuaveGaussiana.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img[::2,::2] # Diminui a imagem
lap = cv2.Laplacian(img, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
resultado = np.vstack([img, lap])
cv2.imshow("Filtro Laplaciano", resultado)
#cv2.imwrite("laplaciano.jpg", resultado)
cv2.waitKey(0)

"""


"""

#MORFOLOGIA EROSAO E DILATACAO

import cv2
import numpy as np
img = cv2.imread('laplaciano.jpg', 0)
kernel = np.ones((5, 5), np.uint8)
img = img[::2,::2] # Diminui a imagem
img_erosion = cv2.erode(img, kernel, iterations=1)
img_dilation = cv2.dilate(img, kernel, iterations=1)
cv2.imshow('Input', img)
cv2.imshow('Erosion', img_erosion)
cv2.imshow('Dilation', img_dilation)
#cv2.imwrite("erosao.jpg", img_erosion)
#cv2.imwrite("dilatacao.jpg", img_dilation)
cv2.waitKey(0)

"""



"""

#TOP HAT

import cv2

# Getting the kernel to be used in Top-Hat
filterSize =(3, 3)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
input_image = cv2.imread("dilatacao.jpg")
input_image = input_image[::2,::2] # Diminui a imagem
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
tophat_img = cv2.morphologyEx(input_image, cv2.MORPH_TOPHAT,
kernel)
cv2.imshow("original", input_image)
cv2.imshow("tophat", tophat_img)
#cv2.imwrite("tophat.jpg", tophat_img)
cv2.waitKey(0)

"""


"""

#BLACK HAT

import cv2
# Defining the kernel to be used in Top-Hat
filterSize = (3, 3)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)

input_image = cv2.imread("dilatacao.jpg")
input_image = input_image[::2,::2] # Diminui a imagem
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Applying the Black-Hat operation
blackhat_img = cv2.morphologyEx(input_image,
cv2.MORPH_BLACKHAT, kernel)
cv2.imshow("original", input_image)
cv2.imshow("blackhat", blackhat_img)
#cv2.imwrite("blackhat.jpg", blackhat_img)
cv2.waitKey(0)


"""


"""
#Limiar adaptativo

import cv2
import numpy as np
img = cv2.imread('way.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converte
img = img[::2,::2] # Diminui a imagem
suave = cv2.GaussianBlur(img, (7, 7), 0) # aplica blur
bin1 = cv2.adaptiveThreshold(suave, 255,
cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5)
bin2 = cv2.adaptiveThreshold(suave, 255,
cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
resultado = np.vstack([
np.hstack([img, suave]),
np.hstack([bin1, bin2])
])
cv2.imshow("Binarização adaptativa da imagem", resultado)
#cv2.imwrite("LimiarizacaoAdapt.jpg", resultado)
cv2.waitKey(0)


"""




"""

#BINARIZAÇÂO

import cv2
import numpy as np
img = cv2.imread('way.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img[::2,::2] # Diminui a imagem


suave = cv2.GaussianBlur(img, (7, 7), 0) # aplica blur
(T, bin) = cv2.threshold(suave, 130, 255, cv2.THRESH_BINARY)
(T, binI) = cv2.threshold(suave, 130, 255, cv2.THRESH_BINARY_INV)
resultado = np.vstack([
    np.hstack([suave, bin]),
    np.hstack([binI, cv2.bitwise_and(img, img, mask = binI)])
])
cv2.imshow("Binarizacao da imagem", binarizacao)
#cv2.imwrite("binarizacaoaposTophat.jpg", resultado)
cv2.waitKey(0)


"""