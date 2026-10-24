
import cv2
import yaml
import shutil
import numpy as np

from pathlib import Path
from ultralytics import YOLO

# Define a pasta principal do projeto
base = Path(__file__).resolve().parent

# Define a origem e o destino dos datasets
origem = base / "Construction"
destino = base / "filtromascara"

# Carrega o modelo de segmentação
modelo = YOLO("yolo11m-seg.pt")

# Define o tamanho do kernel para expandir a máscara
kernel = np.ones((9, 9), dtype=np.uint8)

# Extensões de imagens aceitas
extensoes = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# Contabiliza os resultados
total = 0
segmentadas = 0
sem_pessoa = 0
erros = 0

# Percorre as pastas do dataset
for pasta in ["test", "train", "val"]:

    # Define os caminhos das imagens
    imagens = origem / "images" / pasta
    saida_imagens = destino / "images" / pasta

    # Cria a pasta de destino
    saida_imagens.mkdir(parents=True, exist_ok=True)

    # Percorre os arquivos da pasta
    for arquivo in imagens.iterdir():

        # Ignora arquivos que não são imagens
        if arquivo.suffix.lower() not in extensoes:
            continue

        # Lê a imagem original
        imagem = cv2.imread(str(arquivo))

        if imagem is None:
            print(f"Erro ao ler: {arquivo}")
            erros += 1
            continue

        # Executa a segmentação
        resultado = modelo(
            imagem,
            imgsz=1280,
            conf=0.05,
            retina_masks=True,
            verbose=False
        )[0]

        # Cria uma máscara inicialmente preta
        mascara = np.zeros(imagem.shape[:2], dtype=np.uint8)

        # Verifica se existem máscaras e detecções
        if resultado.masks is not None and resultado.boxes is not None:

            # Percorre as classes detectadas
            for indice, classe in enumerate(
                resultado.boxes.cls.cpu().numpy()
            ):

                # Seleciona somente a classe pessoa
                if int(classe) == 0:

                    # Obtém o contorno da pessoa
                    pontos = resultado.masks.xy[indice].astype(np.int32)

                    # Preenche a região da pessoa em branco
                    cv2.fillPoly(mascara, [pontos], 255)

        # Define o caminho da imagem de saída
        caminho_saida = saida_imagens / arquivo.name

        # Se nenhuma pessoa foi segmentada, preserva a imagem original
        if cv2.countNonZero(mascara) == 0:

            imagem_final = imagem
            sem_pessoa += 1

            print(f"AVISO: sem máscara, imagem original salva: {arquivo.name}")

        else:

            # Expande a máscara para preservar melhor as bordas
            mascara = cv2.dilate(
                mascara,
                kernel,
                iterations=2
            )

            # Mantém as pessoas e torna o fundo preto
            imagem_final = cv2.bitwise_and(
                imagem,
                imagem,
                mask=mascara
            )

            segmentadas += 1

        # Salva a imagem processada
        if cv2.imwrite(str(caminho_saida), imagem_final):
            total += 1
        else:
            print(f"Erro ao salvar: {caminho_saida}")
            erros += 1

    print(f"{pasta}: processamento concluído.")

# Copia as anotações originais para o novo dataset
for pasta in ["test", "train", "val"]:

    origem_labels = origem / "labels" / pasta
    destino_labels = destino / "labels" / pasta

    if origem_labels.exists():

        shutil.copytree(
            origem_labels,
            destino_labels,
            dirs_exist_ok=True
        )

    else:
        print(f"AVISO: pasta de labels não encontrada: {origem_labels}")

# Lê a configuração original do dataset
with open(origem / "data.yaml", "r", encoding="utf-8") as arquivo_yaml:
    configuracao = yaml.safe_load(arquivo_yaml)

# Atualiza os caminhos do novo dataset
configuracao["path"] = "filtromascara"
configuracao["train"] = "images/train"
configuracao["val"] = "images/val"
configuracao["test"] = "images/test"

# Gera o data.yaml do novo dataset
with open(destino / "data.yaml", "w", encoding="utf-8") as arquivo_yaml:
    yaml.safe_dump(
        configuracao,
        arquivo_yaml,
        sort_keys=False,
        allow_unicode=True
    )

# Exibe o resumo do processamento
print("\nPROCESSAMENTO FINALIZADO!")
print(f"Imagens salvas: {total}")
print(f"Imagens segmentadas: {segmentadas}")
print(f"Imagens sem pessoa segmentada: {sem_pessoa}")
print(f"Erros de leitura ou gravação: {erros}")
