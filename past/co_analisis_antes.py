import csv
from itertools import islice
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


belisario_co_a = []
carapungo_co_a = []
centro_co_a = []
cotocollao_co_a = []
elcamal_co_a = []
guamani_co_a = []
chillos_co_a = []
tumbaco_co_a = []

#Read File == Resultados 1 mes durante la pandemia 2020
with open("C:\\Users\\gaibo\\OneDrive\\Escritorio\\Python Projects\\Air Pollution\Historical_data_analysis\\CO DATA-N.csv", "r") as csv_file_antes:
    z = csv.reader(csv_file_antes, delimiter=";")
    for line in islice(z, 141387, 142107):
        belisario_co_a.append(float(line[1]))
        carapungo_co_a.append(float(line[2]))
        centro_co_a.append(float(line[3]))
        cotocollao_co_a.append(float(line[4]))
        elcamal_co_a.append(float(line[5]))
        guamani_co_a.append(float(line[6]))
        chillos_co_a.append(float(line[7]))
        tumbaco_co_a.append(float(line[8]))


#Plot belisario_co_a CO
belisario_co_a = np.array(belisario_co_a)
tiempo_j = range(1, len(belisario_co_a) + 1)
tiempo_j = np.array(tiempo_j)
mask_j = (belisario_co_a > 0)

#Plot carapungo_co_a CO
carapungo_co_a = np.array(carapungo_co_a)
tiempo_k = range(1, len(carapungo_co_a) + 1)
tiempo_k = np.array(tiempo_k)
mask_k = (carapungo_co_a > 0)

#Plot centro_co_a CO
centro_co_a = np.array(centro_co_a)
tiempo_l = range(1, len(centro_co_a) + 1)
tiempo_l = np.array(tiempo_l)
mask_l = (centro_co_a > 0)

#Plot cotocollao_co_a CO
cotocollao_co_a = np.array(cotocollao_co_a)
tiempo_m = range(1, len(cotocollao_co_a) + 1)
tiempo_m = np.array(tiempo_m)
mask_m = (cotocollao_co_a > 0)

#Plot elcamal_co_a CO
elcamal_co_a = np.array(elcamal_co_a)
tiempo_n = range(1, len(elcamal_co_a) + 1)
tiempo_n = np.array(tiempo_n)
mask_n = (elcamal_co_a > 0)

#Plot guamani_co_a CO
guamani_co_a = np.array(guamani_co_a)
tiempo_o = range(1, len(guamani_co_a) + 1)
tiempo_o = np.array(tiempo_o)
mask_o = (guamani_co_a > 0)

#Plot chillos_co_a CO
chillos_co_a = np.array(chillos_co_a)
tiempo_p = range(1, len(chillos_co_a) + 1)
tiempo_p = np.array(tiempo_p)
mask_p = (chillos_co_a > 0)

#Plot tumbaco_co_a CO
tumbaco_co_a = np.array(tumbaco_co_a)
tiempo_q = range(1, len(tumbaco_co_a) + 1)
tiempo_q = np.array(tiempo_q)
mask_q = (tumbaco_co_a > 0)