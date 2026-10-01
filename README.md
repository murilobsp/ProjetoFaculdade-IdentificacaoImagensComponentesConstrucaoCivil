## Guia para uso

### Arquivos

 **introducao.py**<br>
 - Teste para ver versão do Torch (para aprendizado de maquina).
 - Ver se o cuda ta disponivel (para uso da GPU).
 - Modelo da GPU.

  **detect-image.py**<br>
 - Vai ser usado para avaliar as imagens.
 - Seleciona o melhor modelo já treinado (best.pt).
 - Faz o redimensionamento e exibe a imagem com o cv2.

  **treinamento.py**<br>
 - Primeiro IF não deixa criar vários processos pais e filhos.
 - Seleciona o versão do Yolo, no caso yolo11n.pt.
 - model.train contem os parâmetros para o treinamento.

<br>


### Pastas

  **Construction**<br>
 - Modelo com dataset, ainda não treinado.
 - Contém as imagens, labels e validation.
 - Contém o arquivo data.yaml (indicação da localização das imagens e nomes para o treino)

  **runs**<br>
 - Vezes que foi rodado(treinado)
 - Cada pasta "train" foi uma vez.
 - weigths contém o melhor treino dessa vez(best.pt) e o ultimo (last.pt)
 - Contém todo os parâmetros de resultados.

<br>

<img width="580" alt="obra" src="https://github.com/user-attachments/assets/85008010-f51f-4cd6-ac6f-550dcc1e76be" />
