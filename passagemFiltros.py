import cv2
import yaml
from pathlib import Path

#PASTA BASE DOS ARQUIVOS
base = Path(__file__).resolve().parent.parent

#PASSA A CAMINHO PARA AS VARIAVEIS
origem = base / "construction" / "images"
destino =  base / "filtrocinzaconstruction" / "images"


#CRIA O CLAHE
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8,8)
)

#PERCORRE PASTA POR PASTA
for pasta in ["test", "train", "val"]:

    imagens = origem / pasta
    saida_imagens = destino / pasta

    quantidade = 0

    # A CADA ARQUIVO NA PASTA IMG
    for arquivo in imagens.iterdir():

        #LE O ARQUIVO
        imagem = cv2.imread(str(arquivo))

        #TRANSFORMA DE BGR PARA LAB
        lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)

        #FAZ O SPLIT
        l, a , b = cv2.split(lab)

        #APLCIA O CLAHE
        l = clahe.apply(l)

        #FAZ A JUNCAO DO SPLIT E TRANSFORMA EM BGR DNV
        imagem_clahe = cv2.cvtColor(
            cv2.merge((l, a, b)),
            cv2.COLOR_LAB2BGR
        )

        #PASSA PELA GALSSIANA EM MATRIZ DE 3X3 (MAIS SUAVE)
        imagem_final = cv2.GaussianBlur(
            imagem_clahe,
            (3, 3),
            0
        )

        camino_saida = saida_imagens / arquivo.name

        #A CADA IMAGEM SOMA +1
        if cv2.imwrite(str(camino_saida), imagem_final):
            quantidade += 1
        else:
            print(f"Erro ao salvar: {camino_saida}")


    print(f"{pasta}: {quantidade} imagens processadas.")

    # ABRE E LÊ O YAML
    with open(base / "construction" / "data.yaml", "r", encoding="utf-8") as f:
        configuracao = yaml.safe_load(f)

    #CONFIGURA O INTERNO DO YAML
    configuracao["path"] = "filtrocinzaconstruction"
    configuracao["test"] = "images/test"
    configuracao["train"] = "images/train"
    configuracao["val"] = "images/val"

    #ESCREVE O NOVO YAML
    with open(destino/"data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            configuracao,
            f,
            sort_keys=False,
            allow_unicode=True
        )


    print("PROCESSAMENTO FINALIZADO !!!")





