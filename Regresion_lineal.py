import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tabulate import tabulate


import csv
from sklearn import linear_model


def dataSet():

    with open ('precios.csv','r') as f:
       data=list(csv.reader(f,delimiter=","))

    data=np.array(data,dtype=np.float)

   
  
    distancia=data[:,0]
    tiempo=data[:,1]
    precio=data[:,2]

    return distancia,tiempo,precio

def Recargo_nocturno(precio):
    print("\n¿Se aplica recargo nocturno a la tarifa?")
    recargo_n=input(" si o no: ")

    if( recargo_n =="si" or recargo_n=="Si"):
        precio+=500
        print("\nprecio final: ",precio)
    else:
        print("\nprecio final: ",precio)
        

if __name__ == "__main__":
    distancia,tiempo,precio=dataSet()

    X=np.c_[distancia,tiempo]
    x_train,x_test,y_train,y_test=train_test_split(X,precio,test_size=0.3)

    lr_multiple=linear_model.LinearRegression()

    #entrenamiento del modelo
    lr_multiple.fit(x_train,y_train)

    #se realiza lapredicción 
    y_pred=lr_multiple.predict(x_test)

    print("\nLos originales: ",y_test )
    print("\nLos que se predijeron: ",y_pred)

    print("Valor de las pendientes o ceficientes a: ",lr_multiple.coef_)

    print("Valor de la intersección o ceficientes b: ",lr_multiple.intercept_)

    #mientras sea más cercano a 1 mejor
    print("\nPrecisión del modelo:", lr_multiple.score(x_train,y_train))
    print("\nPrecisión del modelo r2:", r2_score(y_test,y_pred))

    print("\nPrecisión error cuadratico medio:", mean_squared_error(y_test,y_pred))

    m=np.c_[y_test,y_pred]
    headers=["y test","y pred"]
    table=tabulate(m,headers,tablefmt="fancy_grid")

    print (table)

    x1=float(input("digite los km: "))
    x2=int(input("digite el tiempo en m: "))

    h=lr_multiple.intercept_ + (lr_multiple.coef_[0]*x1)+(lr_multiple.coef_[1]*x2)
    print("\nprecio parcial: ",h)

    #función de recargo nocturno 
    Recargo_nocturno(h)

   

  

   

   