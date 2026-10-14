from ultralytics import  YOLO
import  multiprocessing

if __name__=="__main__":
    multiprocessing.freeze_support()
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=("construction/data.yaml"),  #Arquivo de configuracao
        epochs=50,                        #Ciclos de treinamento (vai rodar x as imagens)
        imgsz=640,                        #Tamanho das imagens
        batch=16,                         #Quantas imagens por vez
        workers=0,                        #Forcar o carregamento
        patience=0,                       #Forca rodar as x epochs, se nao para quando ele achar que ja deu
        device=0                          #Para usar a GPU
    )

    print("[+] Treinamento Concluido")
