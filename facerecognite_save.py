# Importando os modulos necessarios
import cv2 ,os
import numpy as np

# Para a deteccao facial utilizamos o Haar Cascade do OpenCV
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Para o reconhecimento facial utilizamos o LBPH Face Recognizer 
recognizer = cv2.face.createLBPHFaceRecognizer()

# Pasta com as imagens
path = '.\peoples'

max_height = 128
max_width = 128

# Metodo para retornar as imagens e suas etiquetas(labels) para o treinamento de reconhecimento
def get_images_and_labels(path):
    # Vetor com os respectivos nomes
    image_names = [str.title(os.path.basename(os.path.join(path, f).replace("-"," "))) for f in os.listdir(path)]
    saveNames(image_names)
    # Vetor com a localizacao de cada imagem
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    
    
    images = []
    labels = []
    face = []
    j = 0
    
    for i in range(len(image_paths)):  
        image_path = [os.path.join(image_paths[i], g) for g in os.listdir(image_paths[i])]
        for image_path in image_path:
            # Ler a imagem em escala de cinza
            image = cv2.imread(image_path,0)
            
            # Determinar a label da imagem como um int [people3.4 = 3]
            label = i

            # Detectar faces na imagem
            faces = faceCascade.detectMultiScale(
                image,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Buscar apenas o maior
            size = 0
            for (x, y, w, h) in faces:
                if (w * h) > size: 
                    size = w * h
                    face = [(x, y, w, h)]     

            # Se for detectado alguem, armazena a imagem e a label num vetor
            for (x, y, w, h) in face:                              
                resized = cv2.resize(image[y: y + h, x: x + w], (max_width,max_height), interpolation = cv2.INTER_AREA)
                cv2.equalizeHist(resized, resized);
                images.append(resized)
                labels.append(label)
                print "Face found. Path = " + str(image_path)
                #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
                #cv2.imshow(image_path, resized)
                
                cv2.imwrite( "faces/face"+str(j)+".jpg", resized);
                j += 1
            
    # Retornando as imagens e suas labels
    return images, labels

def saveNames(names): 
    src = open("names.txt", 'w')
    for i in range(len(names)):
        src.write(names[i] + "\n")



# Chamar as imagens e suas repectivas labels
images, labels = get_images_and_labels(path)

# Realizar o treinamento
recognizer.train(images, np.array(labels))

# Salvar o recognizer
recognizer.save("faces.yml")
print("\nSalvo!")

cv2.waitKey(0);
cv2.destroyAllWindows()