# Importamos las librerías necesarias para el control del dron, procesamiento de imágenes y manejo de matrices.
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from djitellopy import tello
import cv2
import time

# Definición de la clase PIDController
class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0):
        """
        Inicializa un controlador PID.

        Args:
            kp, ki, kd: Coeficientes del controlador PID.
            setpoint: Valor objetivo que el PID intentará alcanzar.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

    def update(self, current_value):
        """
        Calcula la salida del PID dado un valor actual.

        Args:
            current_value: Valor actual del sistema (por ejemplo, desplazamiento).

        Returns:
            output: Salida del PID para ajustar el sistema hacia el setpoint.
        """
        # Cálculo del tiempo transcurrido
        current_time = time.time()
        elapsed_time = current_time - self.last_time

        # Calcula el error actual
        error = self.setpoint - current_value
        self.integral += error * elapsed_time
        derivative = (error - self.prev_error) / elapsed_time if elapsed_time > 0 else 0

        # Calcula la salida del PID
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        # Actualiza el estado previo
        self.prev_error = error
        self.last_time = current_time

        return output


# Inicialización del dron
me = tello.Tello()  # Creación de la instancia del dron.
me.connect()  # Conectamos con el dron Tello.
me.streamon()  # Activamos la transmisión de video.
me.takeoff()  # Ordenamos el despegue del dron.z
print(me.get_battery())  # Imprimimos el nivel de batería actual del dron.
me.move_down(30)  # Bajamos el dron 30 cm para ajustar su altura inicial.

# Parámetros de color para detectar la línea (HSV) en el entorno específico.
HsvLine = [3,86,131,16,239,206] # Filtro HSV para la pista naranja en un entorno específico.
width, height = 800, 500  # Tamaño de la ventana de visualización.
imx, imy = width // 2, height // 2  # Centro de la imagen para referencia en el cálculo de vectores.

# Inicialización de los controladores PID para cada eje
pid_lr = PIDController(kp=0.5, ki=0.01, kd=0.15)  # Control de movimiento lateral (izquierda/derecha)
pid_fb = PIDController(kp=1.1, ki=0.01, kd=0.3)  # Control de movimiento adelante/atrás
pid_yaw = PIDController(kp=0.68, ki=0.03, kd=0.18)  # Control de rotación (yaw)

# Listas para almacenar los valores de error y las señales de control.
LR_Error, FB_Error, YAW_Error = [],[],[]
Control_LR, Control_FB, Control_Yaw = [],[],[]

#Updates the data and graphs
def cntrl_plot():
    # this sets up the Matplotlib window:
    matplotlib.use('TkAgg')
    fig, (LRax, FBax, YAWax) = plt.subplots(3, 1, sharex=True, constrained_layout=True)

    # Grafica 1
    LRax.plot(LR_Error, label="Error DX", color="blue")
    LRax.plot(Control_LR, label="Control Left/Right", color="cyan")
    LRax.set_title("Error DX & Control Lateral (Left/Right)", loc='left', fontfamily="serif")
    LRax.legend()
    LRax.grid(True)

    # Gráfica 2: Error en Y y señal FB
    FBax.plot(FB_Error, label="Error DY", color="green")
    FBax.plot(Control_FB, label="Control Forward/Backward", color="lime")
    FBax.set_title("Error DY & Control Forward/Backward", loc='left', fontfamily="serif")
    FBax.legend()
    FBax.grid(True)

    # Gráfica 3: Error en ángulo y señal YAW
    YAWax.plot(YAW_Error, label="Error Ángulo", color="red", )
    YAWax.plot(Control_Yaw, label="Control YAW", color="orange")
    YAWax.set_title("Error Ángulo & Control Yaw", loc='left', fontfamily="serif")
    YAWax.legend()
    YAWax.grid(True)

    plt.suptitle("Drone Seguidor de Línea Señales de Control & Error", fontweight="bold",
                    fontsize=18, fontfamily="serif")
    plt.show()

def thresholding(img):
    """
    Aplica un filtro HSV a la imagen para extraer la línea de color especificado.

    Args:
        img: Imagen de entrada en formato BGR.

    Returns:
        mask: Máscara binaria donde el color especificado aparece en blanco y el resto en negro.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convierte la imagen a espacio de color HSV.
    lower = np.array(HsvLine[0:3])  # Valores inferiores de HSV para el filtro.
    upper = np.array(HsvLine[3:6])  # Valores superiores de HSV para el filtro.
    mask = cv2.inRange(hsv, lower, upper)  # Genera la máscara binaria con el rango de color especificado.
    return mask


def getContours(imgThres, img):
    """
    Detecta el contorno más grande en la máscara para identificar la posición de la línea.

    Args:
        imgThres: Imagen de máscara binaria donde se detectan los contornos.
        img: Imagen original donde se dibuja el contorno.

    Returns:
        cx, cy: Coordenadas x e y del centro del contorno más grande.
    """
    cx, cy = 0, 0  # Coordenadas iniciales del centro.
    contours, _ = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Detecta contornos.
    if len(contours) != 0:
        biggest = max(contours, key=cv2.contourArea)  # Encuentra el contorno más grande.
        x, y, w, h = cv2.boundingRect(biggest)  # Calcula el cuadro delimitador del contorno.
        cx = x + w // 2  # Calcula el centro del contorno en x.
        cy = y + h // 2  # Calcula el centro del contorno en y.
        cv2.drawContours(img, biggest, -1, (255, 0, 255), 7)  # Dibuja el contorno en la imagen original.
        cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)  # Marca el centro del contorno.
    return cx, cy

def getVector(cx, cy, img, imx, imy):
    """
    Crea un vector desde el centro de la imagen al centro del contorno detectado.

    Args:
        cx, cy: Coordenadas del centro del contorno.
        img: Imagen en la que se dibuja el vector.
        imx, imy: Coordenadas del centro de la imagen.

    Returns:
        dx, dy, angulo: Desplazamientos en x e y y el ángulo del vector en grados.
    """
    cv2.circle(img, (imx, imy), 10, (0, 0, 255), cv2.FILLED)  # Dibuja el centro de referencia en la imagen.
    dx, dy = imx - cx, imy - cy  # Calcula desplazamientos en x e y.
    cv2.arrowedLine(img, (imx, imy), (cx, cy), (0, 255, 0), 3)  # Dibuja el vector en la imagen.
    if dx == 0 or dy == 0:  # Verifica si el centro del contorno está alineado con el centro de la imagen.
        return dx, dy, 0
    else:
        angulo = np.degrees(np.arctan2(dx, dy))  # Calcula el ángulo en grados usando atan2 para tener el signo correcto.
        return dx, dy, angulo

def sendCommands(dx, dy, angulo):
    """
    Envía comandos de movimiento al dron usando control PID en cada dirección.
    dx, dy: Desplazamientos en x e y del centro del contorno respecto al centro de la imagen.
    angulo: Ángulo del vector formado entre el centro de la imagen y el centro del contorno.
    """
    # Calculamos las salidas PID para cada dirección y limitamos la salidas.
    LR  = int(np.clip(pid_lr.update(dx), -25, 25))
    fb_output = int(np.clip(pid_fb.update(dy), 0, 35))
    FB = -fb_output + 35
    YAW = int(np.clip(pid_yaw.update(angulo), -45, 45))

    #Agregamos los valores de error y de control de la iteración a sus listas.
    global LR_Error, FB_Error, YAW_Error, Control_LR, Control_FB, Control_Yaw
    LR_Error.append(0.1*dx)
    FB_Error.append(0.1*dy)
    YAW_Error.append(0.1*angulo)
    Control_LR.append(LR)
    Control_FB.append(FB)
    Control_Yaw.append(YAW)

    #Enviamos los valores de control al drone.
    me.send_rc_control(LR, FB, 0, YAW)  # Envía los comandos ajustados al dron.

while True:
    img = me.get_frame_read().frame  # Captura el cuadro actual del video transmitido por el dron.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convierte la imagen de BGR a RGB.
    img = cv2.flip(cv2.resize(img, (width, height)), 0)  # Redimensiona y voltea la imagen.

    # Procesamiento de la imagen para la detección de la línea.
    imgTresh = thresholding(img)  # Aplica el filtro HSV para extraer la línea.
    cx, cy = getContours(imgTresh, img)  # Obtiene el centro del contorno más grande de la línea.
    dx, dy, angulo = getVector(cx, cy, img, imx, imy)  # Calcula el vector hacia el contorno.
    sendCommands(dx, dy, angulo)  # Envía comandos de control en función del vector calculado.

    # Información en pantalla
    battery = "Battery Level " + str(me.get_battery()) + "%"
    cv2.putText(img, "PID Tello Drone Line Follower", (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1)
    cv2.putText(img, "-Aramis Cell", (10, 55), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1)
    cv2.putText(img, battery, (width - 190, height - 32), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1)

    cv2.imshow("Out", img)  # Muestra la imagen procesada en una ventana.
    cv2.imshow("BW", imgTresh)

    if cv2.waitKey(1) & 0xff == ord("z"):
        me.send_rc_control(0, 0, 0, 0)
        me.streamoff()
        cv2.destroyAllWindows()
        me.land()
        cntrl_plot()
        break
