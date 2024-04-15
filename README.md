# Reconhecimento Facial em Python

Este repositório contém dois scripts Python que demonstram o uso de técnicas de reconhecimento facial com a biblioteca OpenCV.

## Scripts

### 1. `facerecognite_load_video.py`

Este script é responsável por carregar um vídeo em tempo real (como uma transmissão de câmera) e identificar faces utilizando o reconhecimento facial. O script usa Haar Cascades para detectar as faces e o método LBPH (Local Binary Patterns Histograms) para reconhecimento facial. Ele exibe as faces reconhecidas com um retângulo verde e o nome da pessoa, se reconhecida.

#### Funcionalidades:

- Captura de vídeo através de uma câmera conectada.
- Uso de múltiplos Haar Cascades para detecção facial, com opção de alternar entre eles.
- Reconhecimento facial com modelo treinado previamente.
- Exibição dos frames com faces detectadas e reconhecidas em tempo real.
- Finalização do programa ao pressionar a tecla 'esc'.

### 2. `facerecognite_save.py`

Este script realiza a coleta de imagens para treinar o reconhecedor de faces. Ele percorre uma pasta específica que contém imagens de pessoas, detecta faces nelas, e as processa para serem usadas no treinamento do modelo de reconhecimento facial.

#### Funcionalidades:

- Leitura de imagens de um diretório específico.
- Detecção de faces nas imagens utilizando Haar Cascade.
- Processamento das faces detectadas para um formato e tamanho uniforme.
- Armazenamento das faces processadas para treinamento.
- Treinamento do modelo LBPH com as faces coletadas.
- Salvamento do modelo treinado para uso futuro.

## Requisitos

Para executar os scripts, você precisará instalar a biblioteca OpenCV em Python. Isso pode ser feito através do comando pip:

```bash
pip install opencv-python
pip install opencv-contrib-python
```

Garanta também que os arquivos Haar Cascade XML estejam no diretório correto, conforme especificado nos scripts.

## Executando os Scripts
Você pode executar cada script a partir da linha de comando:

```bash
python facerecognite_load_video.py
python facerecognite_save.py
```

Assegure-se de que o ambiente onde os scripts serão executados tenha acesso a uma câmera compatível e que as permissões de leitura de diretórios estejam corretas.

