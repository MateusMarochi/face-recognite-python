import cv2, json
import numpy as np

# Definindo entrada da camera
cap = cv2.VideoCapture(1)

# Para a deteccao facial utilizamos o Haar Cascade do OpenCV
cascadePath = ["haarcascade_frontalface_alt_tree.xml","haarcascade_frontalface_alt.xml","haarcascade_frontalface_default.xml"]
cascade_last = -1

# Criar um objeto do tipo LBPHFaceRecognizer
recognizer = cv2.face.createLBPHFaceRecognizer()

#Carregar arquivo de treino
recognizer.load("faces.yml")

fileNames = open("names.txt", "r") 
names = fileNames.read().splitlines()

max_height = 128
max_width = 128

def nothing(x):
    pass
# Trackbar com as Cascades disponiveis
cv2.namedWindow("Reconhecimento")
cv2.createTrackbar('Cascade',"Reconhecimento",0,2,nothing)

# Inicializar a camera
while(True):
    cascade = cv2.getTrackbarPos('Cascade',"Reconhecimento")
    if cascade_last != cascade:
        faceCascade = cv2.CascadeClassifier(cascadePath[cascade])
        cascade_last = cascade
    
    # Captura de cada frame
    ret, frame = cap.read()

    frame = cv2.flip(frame,1)
    
    # Converter o frame em escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    
    # Detectar faces no frame da camera
    faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
    count_F = 0
    for (x, y, w, h) in faces:
        resized = cv2.resize(gray[y: y + h, x: x + w], (max_width,max_height), interpolation = cv2.INTER_AREA)
        cv2.equalizeHist( resized, resized );
        predicted_label = -1
        predicted_confidence = 0
        prediction = recognizer.predict(resized)
        #jsonRec = json.dumps(recognizer)
        #print jsonRec
        cv2.imshow("Face " +str(count_F),resized)
        count_F += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.putText(frame, names[prediction], (x,y), 0, 1 ,(255,255,0))
        
    cv2.imshow("Reconhecimento", frame)
    # Se a tecla 'esc' for pressionada encerrar o programa
    if cv2.waitKey(1) == 27:
        break