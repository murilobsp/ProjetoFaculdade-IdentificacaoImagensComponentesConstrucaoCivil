import tkinter as tk 
from tkinter import filedialog 
from ultralytics import YOLO
import cv2  #lê e processa imagens
from PIL import Image, ImageTk #precisa da lib Pillow para exibir as imagens na interface do Tkinter
from pathlib import Path  # Permite trabalhar com caminhos de pastas e arquivos

janela = tk.Tk() #cria janela principal
janela.title("Compara Detecções")
janela.geometry("1200x600") #tamando da janela
janela.configure(bg="#121315")
area_resultados = tk.Frame(janela, bg="#1C1E22")  # cria um conteiner para organizar os resultados na janela
area_resultados.pack(side="top", fill="both", expand=True, padx=10, pady=10)  # Posiciona os resultados no topo e permite que ocupem o espaço disponível
area_resultados.grid_columnconfigure((0, 1, 2, 3), weight=1)



def selecionar_imagem(): #será executada quando o usuário clicar no botão
    caminho = filedialog.askopenfilename(
        initialdir="Construction/images/test", #caminho de pasta inicial que será aberto
        title="Selecione Uma Imagem",
        filetypes=[("Imagens", "*.jpg *.jpeg *.png")]
    )
    
    if caminho:
        arquivo = Path(caminho) #pega o caminho e nome do arquivo selecionado
        nome_imagem = arquivo.name #extrai o nome completo, incluindo a extensão (jpeg,jpg, etc...)  
        pastas = [ #lista dos caminhos das pastas onde será pego as imagens
            arquivo.parent, #pasta da imagem original selecionada
            Path("filtrocinzaconstruction/images/test"),
            Path("filtroconstruction/images/test"),
            Path("filtromascara/images/test")
    ]

    imagens = [] #cria lista vazia para armazenar as 5 imagens
    for pasta in pastas: #percorre as 5 pastas
        caminho_imagem = pasta / nome_imagem #monta o caminho da imagem com o mesmo nome
        imagem_cv = cv2.imread(str(caminho_imagem)) #carrega a img usando openvc

        if imagem_cv is not None: #verifica se o opencv carregou a imagem mesmo
            imagens.append((caminho_imagem, imagem_cv)) #guarda o caminho e os pixels da imagem na lista
        else:  # Executa caso o OpenCV não consiga ler o arquivo
            print(f"Erro ao carregar: {caminho_imagem}")  # Informa qual imagem apresentou problema

    resultados_yolo = [] #cria lista para guardar os resultados das detecções
    for caminho_imagem, imagem_cv in imagens: #percorre as 5 imagens carregadas
        resultado = model(imagem_cv)[0] #executa yolo na imagem atual
        impressao = resultado.plot() #desenha as identificações do yolo
        
        legendas = [] #lista para guardar as legendas
        for caixa in resultado.boxes: #percorre cada objeto detectado pelo yolo
            classe = int(caixa.cls[0]) #obtem o identificador numerico da classe detectada
            confiança = float(caixa.conf[0]) #salva a confiança da detecção
            nome_classe = model.names[classe] #obtem o nome correspondente à classe
            legendas.append(f"{nome_classe.upper()}: {confiança:.2f}")# monta o texto da legenda com dois decimais
        
        if len(resultado.boxes) == 0: #para caso o yolo não identifique nada na imagem
            legendas.append("Nenhum Objeto Detectado") #ele vai adicionar esse texto na legenda
        
        texto_legenda = "\n".join(legendas) #junta as detecções em linhas separadas
        resultados_yolo.append((caminho_imagem, impressao, texto_legenda)) #guarda o caminho e a imagem com as detecções
       

    for elemento in area_resultados.winfo_children(): #retorna os elementos que estão dentro da área de resultados
        elemento.destroy() #remove o elemento atual

    nomes_filtros = [  # Define os títulos que serão exibidos em cada imagem
            "Original",   
            "Clahe e Gaussiana em Cinza",
            "Clahe e Gaussiana",
            "Binarização e Máscara"
        ]

    for indice, (caminho_imagem, impressao, texto_legenda) in enumerate(resultados_yolo):  # percorre as cinco imagens com as detecções
        imagem_rgb = cv2.cvtColor(impressao, cv2.COLOR_BGR2RGB)  #converte de BGR para RGB
        imagem_pil = Image.fromarray(imagem_rgb)  #converte o array do OpenCV para uma imagem Pillow
        imagem_pil.thumbnail((220, 220))  #redimensiona a imagem para 220x220 para caber na coluna
        imagem_tk = ImageTk.PhotoImage(imagem_pil)  #precisa passar para esse formato para poder ser exibida no tkinter
        
        rotulo_titulo = tk.Label(area_resultados, text=nomes_filtros[indice], bg="#2A3045", fg="#F2EFE9", font=("Arial", 10, "bold"))  # Cria os títulos 
        rotulo_titulo.grid(row=0, column=indice, padx=5, pady=5)  # Posiciona o título na coluna correspondente
        
        rotulo_imagem = tk.Label(area_resultados, image=imagem_tk, bg="#1C1E22")  # Cria um elemento para exibir a imagem
        rotulo_imagem.image = imagem_tk # Mantém uma referência à imagem para que ela não desapareça
        rotulo_imagem.grid(row=1, column=indice, padx=5, pady=5)  # Posiciona a imagem abaixo do título
       
        rotulo_legenda = tk.Label(area_resultados, text=texto_legenda, justify="left", anchor="w", width=30, bg="#222736", fg="#FFB547", font=("Consolas", 9))  # Cria o elemento que irá exibir as legendas 
        rotulo_legenda.grid(row=2, column=indice, padx=5, pady=5)  # Posiciona a legenda abaixo da imagem

model = YOLO("runs/detect/train/weights/best.pt")
model.model.names = {
    0: 'Capacete',
    1: 'Luvas',
    2: 'Colete',
    3: 'Botas',
    4: 'Óculos',
    5: 'Sem Colete',
    6: 'Pessoa',
    7: 'Sem Capacete',
    8: 'Sem Óculos',
    9: 'Sem Luvas',
    10: 'Sem Botas'
}

botao = tk.Button(  #cria o botão de seleção
    janela,  #define a janela principal como contêiner
    text="Selecionar Imagem",  #define o texto do botão
    command=selecionar_imagem,  #define a função executada ao clicar
    bg="#FF7A1A",  #define o fundo azul
    fg="#121315",  #define o texto branco
    activebackground="#E66500",  #define o fundo ao pressionar
    activeforeground="#121315",  #define a cor do texto ao pressionar
    relief="flat",  #demove o efeito de relevo
    borderwidth=0,  #demove a borda padrão
    font=("Arial", 10, "bold"),  #define a fonte do botão
    padx=12,  #adiciona espaço horizontal interno
    pady=6  # a diciona espaço vertical interno
)

botao.pack(side="bottom", pady=15) #adiciona botão fixo e deixa um espaçamento vertical de 15px ao redor

janela.mainloop() #inicia a janela, mantendo aberta e esperando as ações do usuário


