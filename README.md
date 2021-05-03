# YOGESWARAN-POORMEKAH-1PT2-TP4-Methode-de-Dichotomie-Secante

"""---------------------------------------------------------------------------
YOGESWARAN POORMEKAH 1PT2
LAHRECH ABDEL 1PT2
Aéro 1 — Analyse numérique (MA123, 2020-2021)
TP 4 — Méthode de Dichotomie
---------------------------------------------------------------------------"""
"===================Code Poormekah================="
"-----------------------1 Programmation des méthodes-------------------------"
###QUESTION 1###
print('='*10,"1)Programmation des méthodes",'='*10)
print('='*10,"QUESTION 1",'='*10)
from math import cos
from math import sin
from math import tan
from math import exp
from math import log
import matplotlib.pyplot as plt


def Dichotomie(f,a0,b0,epsilon,Nitermax):
    n=1
    liste_des_n = []
    liste_des_m = []
    liste_des_erreurs = []
    a=a0
    b=b0
    while abs(b-a)> epsilon and n<Nitermax:
        m=(a+b)/2
        n+=1
        if (f(a)*f(m)<=0):
            b=m
        else:
            a=m
        erreur = abs((b - a)/2**(n +1))
        liste_des_n.append(n)
        liste_des_m.append(m)
        liste_des_erreurs.append(erreur)
    plt.semilogy(liste_des_n, liste_des_erreurs)
    plt.title("évolution des erreurs avec methode de dichotomie")
    plt.xlabel('n')
    plt.ylabel('erreurs')
    plt.grid()
    plt.show()
    return print("La solution par la méthode de la sécante de la fonction "
          ,f.__name__," est: ",m,". Le nombre d'itération est:",n,".")



    
"===fonction 1==="
def f(x):
    return x**4 + 3*x - 9

    
"===fonction 2==="
def g1(x):
    return x-3*cos(x)+2


"===fonction 3==="
def h(x):
    return x*exp(x)-7

"===fonction 4==="
def i(x):
    return exp(x)-x-10

"===fonction 5==="
def j(x):
    return 2*tan(x)-x-5

"===fonction 6==="
def k(x):
    return exp(x)-x**2-3

"===fonction 7==="
def l(x):
    return 3*x+4*log(x)-7 

"===fonction 8==="
def m(x):
    return x**4-(2*x)**2+4*x-17 

"===fonction 9==="
def n(x):
    return exp(x)-2*sin(x) - 7

"===fonction 10==="
def o(x):
    return log(x**2+4)*exp(x)-10

print("---fonction 1---")
print(Dichotomie(f, 1 ,2, 1e-10,10**4))
print(Dichotomie(f, -1 ,-2, 1e-10,10**4))
print("\n")

print("---fonction 2---")
print(Dichotomie(g1, 1, 2, 1e-10,10**4))
print("\n")

print("---fonction 3---")
print(Dichotomie(h, 1, 2, 1e-10,10**4))
print("\n")

print("---fonction 4---")
print(Dichotomie(i,2, 3, 10**-10,10**4))
print("\n")

print("---fonction 5---")
print(Dichotomie(j,1, 2, 10**-10,10**4))
print("\n")

print("---fonction 6---")
print(Dichotomie(k,1, 2, 10**-10,10**4))
print("\n")

print("---fonction 7---")
print(Dichotomie(l,1, 2, 10**-10,10**4))
print("\n")

print("---fonction 8---")
print(Dichotomie(m,2, 3, 10**-10,10**4))
print("\n")

print("---fonction 9---")
print(Dichotomie(n,2, 3, 10**-10,10**4))
print("\n")

print("---fonction 10---")
print(Dichotomie(o,1, 2, 10**-10,10**4))
print("\n")




###QUESTION 2###
print('='*10,"QUESTION 2",'='*10)

  
def Secante(f,x0,x1,epsilon,Nitermax):
    t = x0
    u = x1
    n = 0
    v = u - f(u)*(u-t)/(f(u)-f(t))
    liste_des_n = []
    liste_des_m = []
    liste_des_erreurs = []
    while abs(v-u) > epsilon:
        n = n+1
        if n > Nitermax:
            message = "Echec apres {} itérations".format(Nitermax)
            return message
        u, v = v, v-f(v)*(v-u)/(f(v)-f(u))
    """erreur = abs((t - u)/2**(n +1))
    
    plt.semilogy(liste_des_n, liste_des_erreurs)
    plt.title("évolution des erreurs avec methode de secante")
    plt.xlabel('n')
    plt.ylabel('erreurs')
    plt.grid()
    plt.show()"""
    liste_des_n.append(n)
    liste_des_m.append(v)
    #liste_des_erreurs.append(erreur)
    print("La solution par la méthode de la sécante de la fonction "
          ,f.__name__," est: ",v,". Le nombre d'itération est:",n,".")

print("---fonction 1---")
print(Secante(f, 1 ,2, 1e-10,10**4))
print("\n")

print("---fonction 2---")
print(Secante(g1, 1, 2, 1e-10,10**4))
print("\n")

print("---fonction 3---")
print(Secante(h, 1, 2, 1e-10,10**4))
print("\n")

print("---fonction 4---")
print(Secante(i,2, 3, 10**-10,10**4))
print("\n")

print("---fonction 5---")
print(Secante(j,1, 2, 10**-10,10**4))
print("\n")

print("---fonction 6---")
print(Secante(k,1, 2, 10**-10,10**4))
print("\n")

print("---fonction 7---")
print(Secante(l,1, 2, 10**-10,10**4))
print("\n")

print("---fonction 8---")
print(Secante(m,2, 3, 10**-10,10**4))
print("\n")

print("---fonction 9---")
print(Secante(n,2, 3, 10**-10,10**4))
print("\n")

print("---fonction 10---")
print(Secante(o,1, 2, 10**-10,10**4))
print("\n")

"-------------------------2 Comparaison des méthodes------------------------"
print('='*10,"2)Comparaison des méthodes",'='*10)

###QUESTION 1###
print('='*10,"QUESTION 1",'='*10)
def PointFixe(g,x0,epsilon,Nitermax):
    n = 0
    xold = x0
    xnew = g(xold)
    erreur = xnew-xold
    liste_des_n = []
    liste_des_m = []
    liste_des_erreurs = []
    while abs(erreur)>epsilon and n<Nitermax:      
        xnew = g(xold)
        n+= 1        # n=n+1
        erreur = xnew-xold
        xold = xnew
        print(xnew,n,(xnew-xold)) 
    liste_des_n.append(n)
    liste_des_m.append(xnew)
    liste_des_erreurs.append(erreur)
    print("Pour",g,"Le nombre d'itérations est de", n)
    print("|xn - xn-1| est inférieur à",epsilon," et |xn - xn-1|=", erreur,".")
    print("La suite tend vers",xnew)
    return xnew

def Newton(f,fder,x0,epsilon,Nitermax):
    n = 1
    xold = x0
    xnew = xold-(f(xold)/fder(xold))
    erreur = abs(xnew-xold)
    liste_des_n = []
    liste_des_m = []
    liste_des_erreurs = []
    while erreur >=epsilon and n<Nitermax:
        xold=xnew
        xnew = xold-(f(xold)/fder(xold))
        n+= 1        # n=n+1
        erreur = abs(xnew-xold)
        xold = xnew
        print(xnew,n,(xnew-xold)) 
        
    liste_des_n.append(n)
    liste_des_m.append(xnew)
    liste_des_erreurs.append(erreur)
    print("Pour",f,"Le nombre d'itérations est de", n)
    print("|xn - xn-1| est inférieur à",epsilon," et |xn - xn-1|=", erreur,".")
    print("La suite tend vers",xnew)
    return xnew



"""print("---fonction 1---")
liste_des_n=
liste_des_xn
liste_des_en
"""
print("\n")

###QUESTION 2###
print('='*10,"QUESTION 2",'='*10)

print("-"*10,"METHODE DU POINT FIXE","-"*10)
def g(x):
    return (1+sin(x))/2

PointFixe(g,0,10**-10,1000)
print("\n")

print("-"*10,"METHODE DE DICHOTOMIE","-"*10)
def f1(x):
    return 2*x - (1 + sin(x))
print(Dichotomie(f1,0,1,10**-10,10**4))
print("\n")

print("-"*10,"METHODE DE NEWTON","-"*10)
def f1der(x):
    return 2 - cos(x)
Newton(f1,f1der,0,10**-10,10**4)
print("\n")

print("-"*10,"METHODE DE LA SECANTE","-"*10)
Secante(f1,0,1,10**-10,10**4)



"================Code Abdel===================="
import math
from math import *
import matplotlib.pyplot as plt
import numpy as np

def Dichotomie(f, a0, b0, epsilon, Nitermax):
    n = 1
    liste_n = []
    liste_m = []
    liste_erreurs = []
    while abs(b0 - a0) > epsilon and n < Nitermax:
        n = n + 1
        m = (a0 + b0)/2
        if f(a0) * f(m) <= 0:
            b0 = m
        else:
            a0 = m
        print(n)
        erreur = abs((b0-a0)/2-0.88)
        liste_n.append(n)
        liste_m.append(m)
        liste_erreurs.append(erreur)
    plt.semilogy(liste_n, liste_erreurs)
    plt.title("évolution des erreurs")
    x=np.array([n])
    y=np.array([m])
    plt.plot(x,y)
    plt.show()
    return m


def f0(x):
    return 2*x - (1 + sin(x))


def f1(x):
    return x**4 +3*x - 9


def f2(x):
    return 3*cos(x) - x - 2


def f3(x):
    return x*exp(x) - 7



print("="*10, "fonction 0", "="*10)
z = Dichotomie(f0, 0, 1, 1E-10, 1E4)
print(z)

print("="*10, "fonction 1", "="*10)
a = Dichotomie(f1, 1, 2, 1E-10, 1E4)
b = Dichotomie(f1, -2, -1,1E-10, 1E4)
print(a)
print(b)

print("="*10, "fonction 2", "="*10)
c = Dichotomie(f2, -2, -1, 1E-10, 1E4)
d = Dichotomie(f2, 0, 1, 1E-10, 1E4)
print(c)
print(d)

print("="*10, "fonction 3", "="*10)
e = Dichotomie(f3, 1, 2, 1E-10, 1E4)
print(e)

def Newton(f, fder, x0, epsilon, Nitermax):
    n = 1
    liste_n = []
    liste_xold= []
    liste_erreurs = []
    xold = x0
    xnew = xold - ((f(xold))/fder(xold))
    erreur = xnew - xold
    while abs(erreur) > epsilon and n < Nitermax:
        xnew = xold - ((f(xold))/fder(xold))
        n = n +1
        erreur = xnew - xold
        calculs_erreurs = abs(xold - xnew)
        xold = xnew
        print("xnew =",xnew, "n =", n, "xnew - xold =", xnew-xold)
        liste_n.append(n)
        liste_xold.append(xold)
        liste_erreurs.append(calculs_erreurs)
    plt.plot(liste_n, liste_erreurs)
    plt.title("évolution des erreurs")
    x=np.array([n])
    y=np.array([xnew])
    plt.grid()
    plt.show()
    return xnew



def f0(x):
    return 2*x - (1 + sin(x))
def f0der(x):
    return 2 - cos(x)


def f1(x):
    return x**4 +3*x - 9

def f1der(x):
    return 4*x**3 + 3


def f2(x):
    return 3*cos(x) - x - 2

def f2der(x):
    return -3*sin(x) -1


def f3(x):
    return x*exp(x) - 7

def f3der(x):
    return exp(x) + x*exp(x)




print("====fonction 0=======")
z = Newton(f0, f0der, 0, 1E-10, 1E4)
print("les solutions de l'équation sont:", z)

print("===fonction 1====")
a = Newton(f1, f1der, -2, 1E-10, 1E4)
b = Newton(f1, f1der, 0, 1E-10, 1E4)
print("les solutions de l'équation sont: ", a, ";", b)

print("===fonction 2====")
a = Newton(f2, f2der, -5, 1E-10, 1E4)
b = Newton(f2, f2der, -2, 1E-10, 1E4)
c = Newton(f2, f2der, 1, 1E-10, 1E4) 
print("les solutions de l'équation sont: ", a, ";", b, ";", c)

print("===fonction 3====")
a = Newton(f3, f3der, 2, 1E-10, 1E4)
print("les solutions de l'équation sont: ", a)


def PointFixe(g, x0, epsilon, Nitermax):
    n = 1
    xold = x0
    xnew = g(xold)
    erreur = xnew - xold
    liste_n = []
    liste_xold = []
    liste_erreurs = []
    while abs(erreur) > epsilon and n < Nitermax:
        xnew = g(xold)
        n = n +1
        erreur = xnew - xold
        calcul_erreur = abs(xold - xnew)
        xold = xnew
        print(xnew, n,  xnew-xold)
        liste_n.append(n)
        liste_xold.append(xold)
        liste_erreurs.append(calcul_erreur)
    plt.semilogy(liste_n, liste_erreurs)
    plt.title("évolution des erreurs")
    x=np.array([n])
    y=np.array([xnew])
    plt.plot(x,y)
    plt.show()

    return xnew



def g0(x):
    return (1+ sin(x))/2



def g1(x):
    return(9-3*x)**0.25 

def g12(x):
    return -(9-3*x)**0.25



def g2(x):
    return acos((2+x)/3)

def g22(x):
    return -(acos((2+x)/3))



def g3(x):
    return log(7/x)






print("========fonction 0========")
v = PointFixe(g0, 0, 1E-10, 1E4)

print("========fonction 1========")
a1 = PointFixe(g1plus,0,1E-10,1E4)
a2 = PointFixe(g1neg,0,1E-10,1E4)
print("les solutions de l'équation sont:")
print(a1)
print(a2)

print("========fonction 2========")
b1= PointFixe(g2plus,0,1E-10,1E4)
b2= PointFixe(g2neg,0,1E-10,1E4)
print("les solutions de l'équation sont:")
print(b1)
print(b2)

print("========fonction 3========")
c= PointFixe(g3,1,1E-10,1E4)
print("les solutions de l'équation sont:")
print(c)


def Secante(f, x0, x1, epsilon, Nitermax):
    n = 1
    liste_n = []
    liste_x0 = []
    liste_erreurs = []
    liste_x2 = []
    while abs(x1 - x0) > epsilon and n < Nitermax:
        n = n + 1
        x2 = x0-f(x0)*(x1-x0)/(f(x1)-f(x0))
        x0 = x1
        x1 = x2
        liste_n.append(n)
        liste_x0.append(x0)
        liste_x2.append(x1)

    for x in liste_x2:
        erreur = abs(x - x0)
        liste_erreurs.append(erreur)
    print( liste_n)
    print( liste_x0)
    print(liste_erreurs)

    plt.figure()
    plt.semilogy(liste_n, liste_erreurs)
    plt.title("évolution des erreurs")
    x=np.array([n])
    y=np.array([x2])
    plt.plot(x,y)
    plt.show()
    return x0


def f0(x):
    return 2*x - (1 + sin(x))



def f1(x):
    return x**4 + 3*x - 9


def f2(x):
    return 3*cos(x) - x - 2


def f3(x):
    return x*exp(x) - 7


#
print("="*10, "fonction 0", "="*10)
v = Secante(f0, 0, 1, 1E-10, 1E4)
print(z)

print("="*10, "fonction 1", "="*10)
a = Secante(f1, 1, 2, 1E-10, 1E4)
b = Secante(f1, -2, -1,1E-10, 1E4)
print(a)
print(b)

print("="*10, "fonction 2", "="*10)
c = Secante(f2, -2, -1, 1E-10, 1E4)
d = Secante(f2, 0, 1, 1E-10, 1E4)
print(c)
print(d)

print("="*10, "fonction 3", "="*10)
e = Secante(f3, 1, 2, 1E-10, 1E4)
print(e)


