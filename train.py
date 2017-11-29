# GenData.py

import sys
import numpy as np
import cv2
import os

# modula el nivel de las variables  ##########################################################################
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

###################################################################################################
def main():
    imgTrainingNumbers = cv2.imread("training_chars.png")            # lee la imagen con los caracteres de prueba

    if imgTrainingNumbers is None:                          # si el archivo no se lee correctamente
         # manda un mensaje de error en la salida estandar
        raw_input("Error la imagen puede ser leida del archivo") # pausa para que el usuario vea el mensaje
        return                                              # sale del programa
    # end if

    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)          # obtiene la imagen en escala de grises
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                        # se difumina la imagen

                                                        # filtra la imagen de escala de grises a blanco y negro
    imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # imagen de entrada
                                      255,                                  # convierte a los pixeles que pasan el umbral en blancos
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # usa un filtro gausiano, que la media, parece tener mejores resultados
                                      cv2.THRESH_BINARY_INV,                # invierte el fondo por el primer plano  (fondo negro)
                                      11,                                   # tamanio de un pixel vecino para calcular el umbral
                                      2)                                    # constante sustraida de la media o de la media ponderada

    cv2.imshow("imgThresh", imgThresh)      # muestra la imagen umbral para referencia

    imgThreshCopy = imgThresh.copy()        # hace una copia de la imagen umbral, es necesario ya que findContours modifica la imagen
# imagen de entrada, asegurese de usar una copia ya que la funcion modificara esta imagen el en curso de enccontrar los contornos
    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                                 cv2.RETR_EXTERNAL,                 # regresa el contono mas externo
                                                 cv2.CHAIN_APPROX_SIMPLE)           # comprime horizontalmente, verticalmente y diagonalmente segmentos y deja unicamente sus puntos finales
                                # declara un arreglo numpy vacio, se usara despues para escribir en un archivo
                                # cero renglones, suficientes columnas para mantener toda la informacion de la imagen
    npaFlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
# declara un lista de clasificaciones vacia esta sera nuestra lista de como clasificaremos nuestros caracteres de entrada del usuario, se escribira un archivo al final
    intClassifications = []
                    #posibles caracteres en los que estamos interesados son digitos de 0 a 9,etc
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

    for npaContour in npaContours:                          # para cada contorno
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:          # si el contorno es suficientemente grande para considerarlo
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)         # se obtiene y encierra en un rectangul

                                                # se encierra en un rectangulo y se le pregunta al usuario por el caracter
            cv2.rectangle(imgTrainingNumbers,           # se dibuja un rectangulo en la imagen de entrenamiento original
                          (intX, intY),                 # incia en la esquina superior izquierda
                          (intX+intW,intY+intH),        # hasta la esquina inferior derecha
                          (0, 0, 255),                  # rojo
                          2)                            # el grueso

            imgROI = imgThresh[intY:intY+intH, intX:intX+intW]                                  # se corta la imagen
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))     # se aumenta el tamanio de la imagen

            cv2.imshow("imgROI", imgROI)                    #muestra el caracter en una nueva ventana para referencia
            cv2.imshow("imgROIResized", imgROIResized)      # muestra la image con el tamanio modificado
            cv2.imshow("training_numbers.png", imgTrainingNumbers)      # muestra la imagen de entrenamiento con el rectangulo dibujado

            intChar = cv2.waitKey(0)                     # se presiona una tecla

            if intChar == 27:                   # si se oprimio esc
                sys.exit()                      # se sale del programa
            elif intChar in intValidChars:      # si el caracter esta en la lista de caracteres de busca

                intClassifications.append(intChar)  # se apendiza el caracter entero en el arreglo de caracteres, este se conertira a float despues

                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # se aplana la imagen para apendizarla al arreglo numpy
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)                    # se agrega la imagen aplanada a l arreglo numpy
            # end if
        # end if
    # end for

    fltClassifications = np.array(intClassifications, np.float32)                   # se convierte la lista de clasificaciones a un arreglo de flotante

    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))   # se aplana el arreglo para escribirlo en un archivo despues

    print "\n\nEntrenamiento completo !!\n"

    np.savetxt("classifications.txt", npaClassifications)           # se escriben las imagenes aplanadas en el archivo
    np.savetxt("flattened_images.txt", npaFlattenedImages)          #

    cv2.destroyAllWindows()             # remueve todas las ventanas

    return

###################################################################################################
if __name__ == "__main__":
    main()
# end if
