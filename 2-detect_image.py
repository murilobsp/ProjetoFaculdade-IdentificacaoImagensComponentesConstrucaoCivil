import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

#Escolher arquivo de imagem
root = tk.Tk()
root.withdraw()
img = filedialog.askopenfilename(title="Escolha um arquivo", filetypes=[ ("Imagens", "*.jpg *.jpeg *.png ")])


#Teste com melhor modelo já treinado
model = YOLO("runs/detect/train/weights/best.pt")
result = model(img,) #save=True
impressao = result[0].plot()


#Redimensiona a exibicao do cv2
altura_orig, largura_orig, _ = impressao.shape
largura_desejada = 900
# Calcula a proporção para manter a imagem esticada corretamente
proporcao = largura_desejada / largura_orig
altura_desejada = int(altura_orig * proporcao)
dimensoes = (largura_desejada, altura_desejada)
# Redimensiona a imagem apenas para exibição
imagem_redimensionada = cv2.resize(impressao, dimensoes, interpolation=cv2.INTER_AREA)


cv2.imshow("Detector Yolo", imagem_redimensionada )
cv2.waitKey(0)
cv2.destroyAllWindows()


print("[+] Concluido")