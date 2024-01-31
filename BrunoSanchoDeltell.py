from tensorflow import keras
import time
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
#2c
from sklearn.model_selection import GridSearchCV
#2d
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
#2e
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist


#############################################
#
#
#   CARGAR DATOS
#
#
#############################################

def load_MNIST_for_adaboost(class_to_train): #DATOS BALANCEADOS 
    # Cargar los datos de entrenamiento y test tal y como nos los sirve keras (MNIST de Yann Lecun)
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
    
    # Filtrar las imágenes y etiquetas solo para la clase a entrenar
    X_train_class = X_train[Y_train == class_to_train]
    Y_train_class = np.ones(len(X_train_class), dtype="int8")  # Etiquetas positivas para la clase de interés
    
    # Filtrar las imágenes y etiquetas para las clases restantes
    X_train_rest = X_train[Y_train != class_to_train]
    Y_train_rest = np.ones(len(X_train_rest), dtype="int8") * -1  # Etiquetas negativas para las otras clases
    # Elegir todas las imágenes de X_train_class
    indices_class = np.arange(len(X_train_class))
    
    # Elegir la misma cantidad de imágenes aleatorias de X_train_rest
    indices_rest = np.random.choice(len(X_train_rest), len(X_train_class), replace=False)
    
    X_train_balanced = np.concatenate([X_train_class[indices_class], X_train_rest[indices_rest]])
    Y_train_balanced = np.concatenate([Y_train_class[indices_class], Y_train_rest[indices_rest]])
    
    # Formatear imágenes a vectores de floats y normalizar
    X_train_balanced = X_train_balanced.reshape((X_train_balanced.shape[0], 28*28)).astype("float32") / 255.0
    X_test = X_test.reshape((X_test.shape[0], 28*28)).astype("float32") / 255.0
    Y_test = Y_test.astype("int8")
    
    return X_train_balanced, Y_train_balanced, X_test, Y_test



def load_MNIST_for_adaboost_random(): #DATOS SIN BALANCEAR
    # Cargar los datos de entrenamiento y test tal y como nos los sirve keras
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
    # Formatear imágenes a vectores de floats y normalizar
    X_train = X_train.reshape((X_train.shape[0], 28*28)).astype("float32") / 255.0
    X_test = X_test.reshape((X_test.shape[0], 28*28)).astype("float32") / 255.0
    #X_train = X_train.astype("float32") / 255.0
    #X_test = X_test.astype("float32") / 255.0
    # Formatear las clases a enteros con signo para aceptar clase -1
    Y_train = Y_train.astype("int8")
    Y_test = Y_test.astype("int8")
    return X_train, Y_train, X_test, Y_test

def load_MNIST_for_MPL_and_CNN():
    # Cargar los datos MNIST
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

    # Aplanar las imágenes y normalizar los valores de píxeles a un rango entre 0 y 1
    X_train = X_train.reshape((X_train.shape[0], -1)) / 255.0
    X_test = X_test.reshape((X_test.shape[0], -1)) / 255.0

    # Convertir las etiquetas a formato one-hot encoding
    Y_train_one_hot = to_categorical(Y_train)
    Y_test_one_hot = to_categorical(Y_test)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_val, Y_train_one_hot, Y_val_one_hot = train_test_split(
        X_train, Y_train_one_hot, test_size=0.2, random_state=42
    )
    return X_train, X_val, Y_train_one_hot, Y_val_one_hot

#############################################
#
#
#   CREAR GRÁFICAS
#
#
#############################################

TA = [(1,900),(2,450),(3,300), (4,225), (5,180), (6,150), (9,100), (10,90), (15,60), (18,50), (25,36), (30,30), (45,20), (50,18), (60,15), (75,12), (90,10), (100,9), (150,6), (180,5), (225,4), (300,3), (450,2), (900,1)]

def crear_grafica_adaboostBin():
    CLASE = 3
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost(CLASE)
    Y_test = np.where(Y_test == CLASE, 1, -1)

    result = []
    for T, A in TA:
        print(f"Entrenando clasificadores Adaboost para el dígito {CLASE}, T={T}, A={A}")

        accuracies = []
        train_times = []

        for _ in range(10):
            classifier = Adaboost(T, A)
            start = time.time()
            classifier.fit(X_train, Y_train)
            end = time.time()

            test = classifier.predict(X_test)
            accuracy = np.mean(Y_test == test) * 100
            accuracies.append(accuracy)
            train_time = (end - start) * 1000  # Convertir a milisegundos
            train_times.append(train_time)

            print(f"Tasa de aciertos (test, tiempo): {accuracy}% y {train_time} ms")

        mean_accuracy = np.mean(accuracies)
        mean_train_time = np.mean(train_times)
        result.append((T, A, mean_train_time, mean_accuracy))
        print(f"Precisión promedio (test, tiempo): {mean_accuracy}% y {mean_train_time} ms")

    T_values, A_values, train_time, tasa_test = [], [], [], []

    for T, A, _time, test in result:
        T_values.append(T)
        A_values.append(A)
        train_time.append(_time)
        tasa_test.append(test)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('T')
    ax1.set_ylabel('Tasa de aciertos (%)', color=color)
    ax1.plot(T_values, tasa_test, color=color, marker='o', label='Tasa de aciertos')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Tiempo de entrenamiento (ms)', color=color)
    ax2.plot(T_values, train_time, color=color, marker='s', label='Tiempo de entrenamiento')
    ax2.tick_params(axis='y', labelcolor=color)
    

    fig.tight_layout()
    plt.xscale('log')  # Escala logarítmica en el eje x
    plt.legend(loc='upper left')
    plt.show()
    
    
def tarea_2B_graficas_rendimiento():
    CLASE = 3
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost(CLASE)
    Y_test = np.where(Y_test == CLASE,1,-1)

    result = []
    for T,A in TA:
        print(f"Entrenando clasificador Adaboost de scikit-learn para el dígito {CLASE}, T={T}")
        #weak_learner = DecisionTreeClassifier(max_depth=1,max_leaf_nodes=A) #A entre 2 - infinito
        classifier = AdaBoostClassifier(n_estimators=T)
        start = time.time()
        classifier.fit(X_train,Y_train)
        end = time.time()
        train_time = end - start
        predictions = classifier.predict(X_test)
        accuracy= accuracy_score(Y_test,predictions)*100
        result.append((T,A,train_time,accuracy))
        print(f"Tasa de aciertos (test, tiempo):{accuracy}% y {end-start}")
        

    T_values, A_values, train_time, tasa_test = [], [], [], []
    for T,A,_time,test in result:
        T_values.append(T)
        A_values.append(A)
        train_time.append(_time)
        tasa_test.append(test)

    fig, ax1 = plt.subplots()

    
    color = 'tab:red'
    ax1.set_xlabel('T')
    ax1.set_ylabel('Tasa de aciertos (%)', color=color)
    ax1.plot(T_values, tasa_test, color=color, marker='o', label='Tasa de aciertos')
    ax1.tick_params(axis='y', labelcolor=color)

    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Tiempo de entrenamiento (s)', color=color)
    ax2.plot(T_values, train_time, color=color, marker='s', label='Tiempo de entrenamiento')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.xscale('log')  # Escala logarítmica en el eje x
    plt.legend(loc='upper left')
    plt.show()
    
    crear_grafica_adaboostBin()


#############################################
#
#
#   MODELOS
#
#
#############################################
    
def tareas_1A_y_1B_adaboost_binario(clase, T, A, verbose=False):
    print(f"Entrenando clasificador Adaboost para el dígito {clase}, T={T}, A={A}")
    adaboost = Adaboost(T=T, A=A)
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost(clase)
    """Y_train = np.where(Y_train == clase,1,-1)"""
    Y_test = np.where(Y_test == clase,1,-1)
    adaboost.fit(X_train, Y_train, verbose=verbose)

    predictions_train = adaboost.predict(X_train, True)
    predictions_test = adaboost.predict(X_test, True)
    
    a_train_calc = np.where(predictions_train == Y_train, 1, 0)
    a_test_calc = np.where(predictions_test == Y_test, 1, 0)
    accuracy_train = (np.sum(a_train_calc)/len(Y_train))*100
    accuracy_test = (np.sum(a_test_calc)/len(Y_test))*100
    
    adaboost.set_accuracy(accuracy_test)

    print(f"Tasas acierto (train, test): {accuracy_train:.4f}%, {accuracy_test:.4f}%")
    return accuracy_train, accuracy_test


class Adaboost:
    def __init__(self, T=20, A=10):
        # Dar valores a los parámetros del clasificador e iniciar la lista de clasificadores débiles vacía
        self.T = T
        self.A = A
        self.classifiers = []
        self.accuracy = None
    
    def set_accuracy(self, accu):
        self.accuracy = accu

    def fit(self, X, Y, verbose=False):
        start_time = time.time()
        # Obtener el número de observaciones y de características por observación de X
        n_samples, n_features = X.shape

        # Iniciar pesos de las observaciones a 1/n_observaciones
        w = np.ones(n_samples) / n_samples
        if verbose:
            print("Entrenando clasificadores de umbral (con dimensión, umbral, dirección y error):")

        for t in range(self.T):
            # Bucle de búsqueda de un buen clasificador débil: desde 1 hasta A repetir
            errors = []
            classifiers = []

            for a in range(self.A):
                # Crear un nuevo clasificador débil aleatorio
                classifier = DecisionStump(n_features)

                # Calcular predicciones de ese clasificador para todas las observaciones
                predictions = classifier.predict(X)

                # Calcular el error: comparar predicciones con los valores deseados
                error = np.sum(w[predictions != Y])

                errors.append(error)
                classifiers.append(classifier)
                """if verbose:
                    print(f"Añadido clasificador {t}: {classifier.feature_index}, {classifier.threshold:.4f}, {classifier.polarity}, {error:.6f}")"""

            # Actualizar mejor clasificador hasta el momento: el que tenga menor error
            best_classifier = classifiers[np.argmin(errors)]
            # Calcular el valor de alfa y las predicciones del mejor clasificador débil
            min_error = min(errors)
            if min_error <= 0:
                alpha = 0
            else:
                alpha = 0.5 * np.log2((1 - min_error) / (min_error))
                

            predictions = best_classifier.predict(X)

            # Actualizar pesos de las observaciones en función de las predicciones, los valores deseados y alfa
            w *= np.exp(-alpha * Y * predictions)
            
            # Normalizar a 1 los pesos
            w /= np.sum(w)

            # Guardar el clasificador en la lista de clasificadores de Adaboost
            self.classifiers.append((best_classifier, alpha))
            if verbose:
                print(f"Mejor clasificador {t}: {best_classifier.feature_index}, {best_classifier.threshold:.4f}, {best_classifier.polarity}, {min_error:.6f}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        if verbose:
            print(f"Tiempo de entrenamiento: {elapsed_time} segundos")
            
    def predict(self, X, sign=True):
        # Calcular las predicciones de cada clasificador débil para cada input multiplicadas por su alfa
        # Sumar para cada input todas las predicciones ponderadas y decidir la clase en función del signo
        predictions = np.zeros((X.shape[0], len(self.classifiers)))

        for i, (classifier, alpha) in enumerate(self.classifiers):
            predictions[:, i] = alpha * classifier.predict(X)

        if not sign:
            return np.sum(predictions, axis=1)  # Se usará para el multiclase
        else:
            return np.sign(np.sum(predictions, axis=1))
        
        
class DecisionStump:
    def __init__(self, n_features):
        # Seleccionar al azar una característica, un umbral y una polaridad.
        self.feature_index = np.random.choice(n_features)
        self.threshold = np.random.uniform(0, 1) 
        self.polarity = np.random.choice([-1, 1])

    def predict(self, X):
        if len(X.shape) == 1:
            # Si es una sola imagen, conviértela en una matriz de una sola fila
            X = X.reshape(1, -1)
        
        # Si la característica que comprueba este clasificador es mayor que el umbral y la polaridad es 1
        # o si es menor que el umbral y la polaridad es -1, devolver 1 (pertenece a la clase)
        # Si no, devolver -1 (no pertenece a la clase)
        predictions = np.ones(X.shape[0])
        predictions[X[:, self.feature_index] < self.threshold] = -1 * self.polarity
        return predictions
    

class AdaboostMulticlass:
    def __init__(self, T, A):
        self.T = T  # Número de clasificadores débiles
        self.A = A  # Factor de ponderación de los clasificadores débiles
        self.classifiers = []
        self.accuracy = None
        
    def fit(self):
        Y_train_ind = None
        Y_test_ind = None
        for clase in range(0, 10):
            X_train, Y_train_ind, _, _ = load_MNIST_for_adaboost(clase)
            
            adaboost = Adaboost(T=self.T, A=self.A)  # Crea una nueva instancia de Adaboost para cada clase
            adaboost.fit(X_train, Y_train_ind, verbose=False)
            
            # Calcular la precisión de cada clasificador
            predictions_train = adaboost.predict(X_train)
            
            a_train_calc = np.where(predictions_train == Y_train_ind, 1, 0)
            accuracy_train = (np.sum(a_train_calc) / len(Y_train_ind)) * 100
            adaboost.set_accuracy(accuracy_train)
            print(f"Classifier {clase} has an accuracy of: {adaboost.accuracy}")
            self.classifiers.append(adaboost)
        
    def predict_1_image(self,X):
       # Predict para todas las imágenes en X_test con cada clasificador Adaboost
        all_predictions = np.zeros((len(X), len(self.classifiers)))

        for i, adaboost in enumerate(self.classifiers):
            all_predictions[:, i] = adaboost.predict(X, sign=False)
            
        # Determina la predicción final como la clase con el máximo valor en cada fila
        final_prediction = np.argmax(all_predictions[0])
        
        image = X.reshape((28, 28))  # Ajusta las dimensiones según tu conjunto de datos

        # Muestra la imagen
        plt.imshow(image, cmap='gray')
        plt.title('MNIST Image')
        plt.show()
        print(f"La predicción del Multiclase para la imagen es: {final_prediction}")
            
            
    def calculate_accuracy(self):
        _, _, X_test, Y_test = load_MNIST_for_adaboost(0)#X_test e Y_test son aleatorios no ponderados
        # Predict para todas las imágenes en X_test con cada clasificador Adaboost
        all_predictions = np.zeros((len(X_test), len(self.classifiers)))

        for i, adaboost in enumerate(self.classifiers):
            all_predictions[:, i] = adaboost.predict(X_test, sign=False)
            
        # Determina la predicción final como la clase con el máximo valor en cada fila
        final_predictions = np.argmax(all_predictions, axis=1)
        # Calcula el porcentaje final de acierto comparando con las etiquetas reales en Y_test
        accuracy = np.mean(final_predictions == Y_test) * 100
        print(f"Porcentaje final de acierto: {accuracy:.4f}%")
        # Imprimir recuento de imágenes acertadas por clase
        correct_by_class = np.zeros(10)
        for i in range(10):
            correct_by_class[i] = np.sum((final_predictions == Y_test) & (Y_test == i))

        print("Recuento de imágenes acertadas por clase:")
        for i in range(10):
            print(f"Clase {i}: {correct_by_class[i]} imágenes acertadas")
        

def tarea_1D_adaboost_multiclase(T, A):
    _, _, X_test, Y_test = load_MNIST_for_adaboost(0)#X_test e Y_test son aleatorios no ponderados
    multi = AdaboostMulticlass(T,A)
    multi.fit()
    multi.calculate_accuracy()
    multi.predict_1_image(X_test[2])
     
    
def tarea_2A_AdaBoostClassifier_default():
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost_random()
    # Crear un clasificador AdaBoost
    classifier = AdaBoostClassifier()
    # Entrenar el clasificador con los conjuntos de entrenamiento
    classifier.fit(X_train, Y_train)
    predictions = classifier.predict(X_test)
    accuracy= accuracy_score(Y_test,predictions)*100
    print(f"Accuracy on Adaboost SKLearn:{accuracy:.3f} %")
    return accuracy
    
    
def tarea_2C_AdaBoostClassifier_DecisionTree():
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost_random()

    # Configura el DecisionTreeClassifier
    decision_tree = DecisionTreeClassifier(max_depth=3)  # Ajusta los parámetros según sea necesario

    # Configura el AdaBoostClassifier con el DecisionTreeClassifier
    classifier = AdaBoostClassifier(base_estimator=decision_tree)
    start_time = time.time()
    # Entrena el clasificador con los conjuntos de entrenamiento
    classifier.fit(X_train, Y_train)
    end_time = time.time()
    elapsed_time = end_time - start_time
    

    # Realiza predicciones en el conjunto de prueba
    predictions = classifier.predict(X_test)

    # Calcula la precisión
    accuracy = accuracy_score(Y_test, predictions) * 100

    print(f"Accuracy on AdaBoost with Decision Tree: {accuracy:.3f} %")
    print(f"Training Time: {elapsed_time:.3f} seconds")
    return accuracy

def tarea_2D_MLP_Keras():
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost_random()
    Y_train_one_hot = to_categorical(Y_train)
    Y_test_one_hot = to_categorical(Y_test)
    
    model = Sequential()
    # Aplanar las imágenes antes de la capa densa
    model.add(Dense(units=128, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    start_time = time.time()
    model.fit(X_train, to_categorical(Y_train), epochs=10, batch_size=64, validation_split=0.3, verbose=1)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    predictions = np.argmax(model.predict(X_test), axis=-1)
    accuracy = accuracy_score(Y_test, predictions) * 100
    print(f"Accuracy on MLP: {accuracy:.3f}%")
    print(f"Training Time: {elapsed_time:.2f} seconds")
 

def tarea_2E_CNN_Keras():

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # Preprocesar los datos
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)) / 255.0
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)) / 255.0

    # Convertir etiquetas a formato one-hot encoding
    Y_train_one_hot = to_categorical(Y_train)
    Y_test_one_hot = to_categorical(Y_test)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_val, Y_train_one_hot, Y_val_one_hot = train_test_split(
        X_train, Y_train_one_hot, test_size=0.2, random_state=42
    )

    # Crear el modelo CNN
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compilar el modelo
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    start_time = time.time()
    # Entrenar el modelo
    model.fit(X_train, Y_train_one_hot, epochs=10, batch_size=32, validation_data=(X_val, Y_val_one_hot), verbose=1)
    end_time = time.time()
    # Evaluar el modelo en el conjunto de prueba
    accuracy = model.evaluate(X_test, Y_test_one_hot, verbose=0)[1] * 100
    print(f"Accuracy on CNN: {accuracy:.2f}%")
    training_time = end_time - start_time
    print(f"Training Time: {training_time:.3f} seconds")

def tarea_2F_graficas_rendimiento():#Realizada en la Memoria
    pass


if __name__== "__main__":
    #rend_1A = tareas_1A_y_1B_adaboost_binario(3,50,50, verbose = True)
    #rend_1C = crear_grafica_adaboostBin()
    rend_1D = tarea_1D_adaboost_multiclase(50, 50)
    #rend_1E = tarea_1E_adaboost_multiclase_mejorado(T=incognita, A=incognita)
    #tarea_1E_graficas_rendimiento(rend_1D, rend_1E)
    
    #rend_2A = tarea_2A_AdaBoostClassifier_default()
    #tarea_2B_graficas_rendimiento()
    #rend_2C = tarea_2C_AdaBoostClassifier_DecisionTree()
    #rend_2D = tarea_2D_MLP_Keras()
    #rend_2E = tarea_2E_CNN_Keras()
    #rend_2F = tarea_2F_graficas_rendimiento()

