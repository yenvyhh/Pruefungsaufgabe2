# Prüfungsaufgabe 2: Unit Testing and Logging for Data Science via Logistic Regression

**Quellen:**
Unit Testing & Logging: https://towardsdatascience.com/unit-testing-and-logging-for-data-science-d7fb8fd5d217 
Logistic Regression: Udemy Kurs - Python für Data Science, Machine Learning & Visualization

**Zu Beginn bitte unter "Cell" -> "Run All" auswählen.**

**Die Funktionen my_logger und my_timer werden hinzugefügt:**
from functools import wraps
def my_logger(orig_func):
    import logging
    logging.basicConfig(filename='{}.log'.format(orig_func.__name__), level=logging.INFO)

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logging.info(
            'Ran with args: {}, and kwargs: {}'.format(args, kwargs))
        return orig_func(*args, **kwargs)

    return wrapper

def my_timer(orig_func):
    import time

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        print('{} ran in: {} sec'.format(orig_func.__name__, t2))
        return result

    return wrapper

**Anwendung von my_logger und my_timer auf ausgewählte Funktionen.**
***Erste Funktion auf die Fit-Funktion
@my_logger
@my_timer
def logmodelFit(): 
    return logmodel.fit(X_train,y_train)
logmodelFit()
**Nach Ausführung wird folgenden Ergebnis angezeigt: 
***logmodelFit ran in: 0.09561443328857422 sec
***LogisticRegression()"

***Zweite Funktion auf die Prediction-Funktion
@my_logger
@my_timer

def lm_predict():
    return logmodel.predict(X_test)
predict = lm_predict()

**Nach Ausführung wird folgenden Ergebnis angezeigt: 
***lm_predict ran in: 0.002523660659790039 sec

**Für die my_logger Funktion wird eine Log-Datei "logmodelFit.log" erzeugt, welcher im Ordner gespeichert wird

#Testfälle 1 und 2 

**Für den Testfall 1 steht zum einen der testdatenfile_1.txt sowie der testdatenfile_2.txt zur Verfügung. **Die textfiles werden an folgender Stelle eingelesem
with open("testdatenfile_1.txt", 'r') as f:
Inhalt = f.read()
**Bei Verwendung des ersten Textfiles (testdaten_score=0.85) ist der Test erfolgreich (Anzeige von "OK"). Zur Verwendung des zweiten Testfiles (testdaten_score=0.90) muss in der obigen Funktion der "testdatenfile_1.txt" durch "testdatenfile_2.txt" ersetzt werden. Als Ergebnis werden wir erhalten, dass der Test durchgefallen (Anzeige von "FAILED") ist. 

**Für Testfall 2 wird überprüft, ob die Laufzeit der Trainingsfunktion, die 120%-Grenze der repräsentativen Laufzeit überschreitet.**
time_1=time.time()
logmodel_test.fit(X_train,y_train)
self.time_2 = time.time() - time_1
**Durch Ausführen der oben gezeigten Zeilen, wird im Notebook die Laufzeit für die Trainingsfunktion ermittelt, in dem die aktuelle Zeit (time_1) erfasst wird, das Model gefittet wird und im Anschluss  erneut die Zeit ermittelt wird und time_1 subtrahiert wird. Das ergenis entspricht dann time_2.
Die 120%ige repräsentative Laufzeit wird ermittelt, indem der erhaltene Wert bei:**
@my_logger
@my_timer
def logmodelFit(): 
    return logmodel.fit(X_train,y_train)
logmodelFit()
**Mit dem Ergebnis: logmodelFit ran in: 0.031914472579956055 sec mit dem Faktor 1,2 multipliziert wird.
Im vorliegenden Ergebnis ist die Lautzeit der Trainingsfunktion ***0.03780817985534668*** und der Grenzwert der repräsentativen Lautzeit bei **0.0637061119079589***
