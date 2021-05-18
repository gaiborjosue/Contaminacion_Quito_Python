import csv
from itertools import islice
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import random
import numpy as np
from scipy.stats import ttest_rel, kstest, normaltest
import scipy.stats as stats

from sklearn.preprocessing import PowerTransformer


plt.style.use('ggplot')

belisario_o3 = []
carapungo_o3 = []
centro_o3 = []
cotocollao_o3 = []
elcamal_o3 = []
guamani_o3 = []
chillos_o3 = []
tumbaco_o3 = []

belisario_o3_a = []
carapungo_o3_a = []
centro_o3_a = []
cotocollao_o3_a = []
elcamal_o3_a = []
guamani_o3_a = []
chillos_o3_a = []
tumbaco_o3_a = []

def medias_antes_durante(belisario_co, carapungo_co, centro_co, cotocollao_co, elcamal_co, guamani_co, chillos_co, tumbaco_co, belisario_co_a, carapungo_co_a, centro_co_a, cotocollao_co_a, elcamal_co_a, guamani_co_a, chillos_co_a, tumbaco_co_a):
    bel_mean = statistics.mean(belisario_co)
    car_mean = statistics.mean(carapungo_co)
    cen_mean = statistics.mean(centro_co)
    coto_mean = statistics.mean(cotocollao_co)
    cama_mean = statistics.mean(elcamal_co)
    gua_mean = statistics.mean(guamani_co)
    chillos_mean = statistics.mean(chillos_co)
    tum_mean = statistics.mean(tumbaco_co)
    
    bel_mean_a = statistics.mean(belisario_co_a)
    car_mean_a = statistics.mean(carapungo_co_a)
    cen_mean_a = statistics.mean(centro_co_a)
    coto_mean_a = statistics.mean(cotocollao_co_a)
    cama_mean_a = statistics.mean(elcamal_co_a)
    gua_mean_a = statistics.mean(guamani_co_a)
    chillos_mean_a = statistics.mean(chillos_co_a)
    tum_mean_a = statistics.mean(tumbaco_co_a)
    
    return print(f"{bel_mean, car_mean, cen_mean, coto_mean, cama_mean, gua_mean, chillos_mean, tum_mean} Antes: {bel_mean_a, car_mean_a, cen_mean_a, coto_mean_a, cama_mean_a, gua_mean_a, chillos_mean_a, tum_mean_a}")

belisario_o3_19 = []
carapungo_o3_19 = []
centro_o3_19 = []
cotocollao_o3_19 = []
elcamal_o3_19 = []
guamani_o3_19 = []
chillos_o3_19 = []
tumbaco_o3_19 = []


#Read File == Resultados 1 mes durante la pandemia 2020
with open("/home/edward/DataScience_Projects/Air Pollution/Historical_data_analysis/O3/O3-N.csv", "r") as csv_file:
    f = csv.reader(csv_file, delimiter=";")
    for line in islice(f, 142131, 146570):
        belisario_o3.append(float(line[1]))
        carapungo_o3.append(float(line[2]))
        centro_o3.append(float(line[3]))
        cotocollao_o3.append(float(line[4]))
        elcamal_o3.append(float(line[5]))
        guamani_o3.append(float(line[6]))
        chillos_o3.append(float(line[7]))
        tumbaco_o3.append(float(line[8]))

#Read File == Resultados 1 mes antes la pandemia 2020
with open("/home/edward/DataScience_Projects/Air Pollution/Historical_data_analysis/O3/O3-N.csv", "r") as csv_file_antes:
    z = csv.reader(csv_file_antes, delimiter=";")
    for line in islice(z, 137691, 142130):
        belisario_o3_a.append(float(line[1]))
        carapungo_o3_a.append(float(line[2]))
        centro_o3_a.append(float(line[3]))
        cotocollao_o3_a.append(float(line[4]))
        elcamal_o3_a.append(float(line[5]))
        guamani_o3_a.append(float(line[6]))
        chillos_o3_a.append(float(line[7]))
        tumbaco_o3_a.append(float(line[8]))

#Read File == Resultados 1 mes 2019
with open("/home/edward/DataScience_Projects/Air Pollution/Historical_data_analysis/O3/O3-N.csv", "r") as csv_file_2019:
    c = csv.reader(csv_file_2019, delimiter=";")
    for line in islice(c, 133347, 137786):
        belisario_o3_19.append(float(line[1]))
        carapungo_o3_19.append(float(line[2]))
        centro_o3_19.append(float(line[3]))
        cotocollao_o3_19.append(float(line[4]))
        elcamal_o3_19.append(float(line[5]))
        guamani_o3_19.append(float(line[6]))
        chillos_o3_19.append(float(line[7]))
        tumbaco_o3_19.append(float(line[8]))

#!!!!FIGURE!!!!!

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, figsize=(16,10))

#!!!!!!!!!!!!!!!


#Plot belisario_o3 CO
belisario_o3 = np.array(belisario_o3)
tiempo_b = range(1, len(belisario_o3) + 1)
tiempo_b = np.array(tiempo_b)
mask = (belisario_o3 > 0)

belisario_o3_a = np.array(belisario_o3_a)
tiempo_j = range(1, len(belisario_o3_a) + 1)
tiempo_j = np.array(tiempo_j)
mask_j = (belisario_o3_a > 0)

belisario_o3_19 = np.array(belisario_o3_19)
tiempo_r = range(1, len(belisario_o3_19) + 1)
tiempo_r = np.array(tiempo_r)
mask_r = (belisario_o3_19 > 0)

ax1.plot(tiempo_b[mask], belisario_o3[mask])
ax1.plot(tiempo_j[mask_j], belisario_o3_a[mask_j])
#ax1.plot(tiempo_r[mask_r], belisario_o3_19[mask_r])
ax1.set_title("Belisario")

# pt = PowerTransformer(method='box-cox')
# data = belisario_o3.reshape(-1, 1)
# pt.fit(data)
# transformed_data = pt.transform(data)

# k2, p = normaltest(data)
# transformed_k2, transformed_p = normaltest(transformed_data)

wiltest_belisario = stats.wilcoxon(belisario_o3, belisario_o3_a)

#Plot carapungo_o3 CO
carapungo_o3 = np.array(carapungo_o3)
tiempo_c = range(1, len(carapungo_o3) + 1)
tiempo_c = np.array(tiempo_c)
mask_c = (carapungo_o3 > 0)

carapungo_o3_a = np.array(carapungo_o3_a)
tiempo_k = range(1, len(carapungo_o3_a) + 1)
tiempo_k = np.array(tiempo_k)
mask_k = (carapungo_o3_a > 0)

carapungo_o3_19 = np.array(carapungo_o3_19)
tiempo_s = range(1, len(carapungo_o3_19) + 1)
tiempo_s = np.array(tiempo_s)
mask_s = (carapungo_o3_19 > 0)

wiltest_carapungo = stats.wilcoxon(carapungo_o3, carapungo_o3_a)

ax2.plot(tiempo_c[mask_c], carapungo_o3[mask_c])
ax2.plot(tiempo_k[mask_k], carapungo_o3_a[mask_k])
#ax2.plot(tiempo_s[mask_s], carapungo_o3_19[mask_s])
ax2.set_title("Carapungo")


#Plot centro_o3 CO
centro_o3 = np.array(centro_o3)
tiempo_d = range(1, len(centro_o3) + 1)
tiempo_d = np.array(tiempo_d)
mask_d = (centro_o3 > 0)

centro_o3_a = np.array(centro_o3_a)
tiempo_l = range(1, len(centro_o3_a) + 1)
tiempo_l = np.array(tiempo_l)
mask_l = (centro_o3_a > 0)

centro_o3_19 = np.array(centro_o3_19)
tiempo_t = range(1, len(centro_o3_19) + 1)
tiempo_t = np.array(tiempo_t)
mask_t = (centro_o3_19 > 0)

wiltest_centro = stats.wilcoxon(centro_o3, centro_o3_a)

ax3.plot(tiempo_d[mask_d], centro_o3[mask_d])
ax3.plot(tiempo_l[mask_l], centro_o3_a[mask_l])
#ax3.plot(tiempo_t[mask_t], centro_o3_19[mask_t])
ax3.set_title("Centro")

#Plot cotocollao_o3 CO
cotocollao_o3 = np.array(cotocollao_o3)
tiempo_e = range(1, len(cotocollao_o3) + 1)
tiempo_e = np.array(tiempo_e)
mask_e = (cotocollao_o3 > 0)

cotocollao_o3_a = np.array(cotocollao_o3_a)
tiempo_m = range(1, len(cotocollao_o3_a) + 1)
tiempo_m = np.array(tiempo_m)
mask_m = (cotocollao_o3_a > 0)

cotocollao_o3_19 = np.array(cotocollao_o3_19)
tiempo_u = range(1, len(cotocollao_o3_19) + 1)
tiempo_u = np.array(tiempo_u)
mask_u = (cotocollao_o3_19 > 0)

wiltest_cotocollao = stats.wilcoxon(cotocollao_o3, cotocollao_o3_a)

ax4.plot(tiempo_e[mask_e], cotocollao_o3[mask_e])
ax4.plot(tiempo_m[mask_m], cotocollao_o3_a[mask_m])
#ax4.plot(tiempo_u[mask_u], cotocollao_o3_19[mask_u])
ax4.set_title("Cotocollao")


# for ax in fig.get_axes():
#     ax.label_outer()

fig.text(0.5, 0.04, 'Tiempo (Horas)', ha='center')
fig.text(0.04, 0.5, 'Índices O3 (mg/m3)', va='center', rotation='vertical')

fig.legend(["Durante", "Antes"])
plt.savefig("/home/edward/DataScience_Projects/Air Pollution/Historical_data_analysis/O3/Gráficos O3/1.png")
plt.show()


fig2, ((ax1, ax2)) = plt.subplots(1,2, sharex=True, figsize=(16,10))

#Plot elcamal_o3 CO
elcamal_o3 = np.array(elcamal_o3)
tiempo_f = range(1, len(elcamal_o3) + 1)
tiempo_f = np.array(tiempo_f)
mask_f = (elcamal_o3 > 0)

elcamal_o3_a = np.array(elcamal_o3_a)
tiempo_n = range(1, len(elcamal_o3_a) + 1)
tiempo_n = np.array(tiempo_n)
mask_n = (elcamal_o3_a > 0)

elcamal_o3_19 = np.array(elcamal_o3_19)
tiempo_v = range(1, len(elcamal_o3_19) + 1)
tiempo_v = np.array(tiempo_v)
mask_v = (elcamal_o3_19 > 0)

wiltest_elcamal = stats.wilcoxon(elcamal_o3, elcamal_o3_a)

ax1.plot(tiempo_f[mask_f], elcamal_o3[mask_f])
ax1.plot(tiempo_n[mask_n], elcamal_o3_a[mask_n])
#ax1.plot(tiempo_u[mask_v], elcamal_o3_19[mask_v])
ax1.set_title("El camal")

#Plot guamani_o3 CO
guamani_o3 = np.array(guamani_o3)
tiempo_g = range(1, len(guamani_o3) + 1)
tiempo_g = np.array(tiempo_f)
mask_g = (guamani_o3 > 0)

guamani_o3_a = np.array(guamani_o3_a)
tiempo_o = range(1, len(guamani_o3_a) + 1)
tiempo_o = np.array(tiempo_o)
mask_o = (guamani_o3_a > 0)

guamani_o3_19 = np.array(guamani_o3_19)
tiempo_w = range(1, len(guamani_o3_19) + 1)
tiempo_w = np.array(tiempo_w)
mask_w = (guamani_o3_19 > 0)

wiltest_guamani = stats.wilcoxon(guamani_o3, guamani_o3_a)

ax2.plot(tiempo_g[mask_g], guamani_o3[mask_g])
ax2.plot(tiempo_o[mask_o], guamani_o3_a[mask_o])
#ax2.plot(tiempo_w[mask_w], guamani_o3_19[mask_w])
ax2.set_title("Guamní")

#Will take the same label axis
# for ax in fig2.get_axes():
#     ax.label_outer()
fig2.text(0.5, 0.04, 'Tiempo (Horas)', ha='center')
fig2.text(0.04, 0.5, 'Índices O3 (mg/m3)', va='center', rotation='vertical')


fig2.legend(["Durante", "Antes"])
plt.savefig("/home/edward/DataScience_Projects/Air Pollution/Historical_data_analysis/O3/Gráficos O3/2.png")
plt.show()

fig3, ((ax3, ax4)) = plt.subplots(1,2, sharex=True, figsize=(16,10))
#Plot chillos_o3 CO
chillos_o3 = np.array(chillos_o3)
tiempo_h = range(1, len(chillos_o3) + 1)
tiempo_h = np.array(tiempo_h)
mask_h = (chillos_o3 > 0)

chillos_o3_a = np.array(chillos_o3_a)
tiempo_p = range(1, len(chillos_o3_a) + 1)
tiempo_p = np.array(tiempo_p)
mask_p = (chillos_o3_a > 0)

chillos_o3_19 = np.array(chillos_o3_19)
tiempo_x = range(1, len(chillos_o3_19) + 1)
tiempo_x = np.array(tiempo_x)
mask_x = (chillos_o3_19 > 0)

wiltest_chillos = stats.wilcoxon(chillos_o3, chillos_o3_a)

ax3.plot(tiempo_h[mask_h], chillos_o3[mask_h])
ax3.plot(tiempo_p[mask_p], chillos_o3_a[mask_p])
#ax3.plot(tiempo_x[mask_x], chillos_o3_19[mask_x])
ax3.set_title("Chillos")

#Plot tumbaco_o3 CO
tumbaco_o3 = np.array(tumbaco_o3)
tiempo_i = range(1, len(tumbaco_o3) + 1)
tiempo_i = np.array(tiempo_i)
mask_i = (tumbaco_o3 > 0)

tumbaco_o3_a = np.array(tumbaco_o3_a)
tiempo_q = range(1, len(tumbaco_o3_a) + 1)
tiempo_q = np.array(tiempo_q)
mask_q = (tumbaco_o3_a > 0)


tumbaco_o3_19 = np.array(tumbaco_o3_19)
tiempo_y = range(1, len(tumbaco_o3_19) + 1)
tiempo_y = np.array(tiempo_y)
mask_y = (tumbaco_o3_19 > 0)

wiltest_tumbaco = stats.wilcoxon(tumbaco_o3, tumbaco_o3_a)

ax4.plot(tiempo_i[mask_i], tumbaco_o3[mask_i])
ax4.plot(tiempo_q[mask_q], tumbaco_o3_a[mask_q])
#ax4.plot(tiempo_y[mask_y], tumbaco_o3_19[mask_y])
ax4.set_title("Tumbaco")


fig3.text(0.5, 0.04, 'Tiempo (Horas)', ha='center')
fig3.text(0.04, 0.5, 'Índices O3 (mg/m3)', va='center', rotation='vertical')

fig3.legend(["Durante", "Antes"])

plt.savefig("/home/edward/DataScience_Projects/Air Pollution/Historical_data_analysis/O3/Gráficos O3/3.png")
plt.show()


# belisario_o3 = belisario_o3.tolist()

# hmean = np.mean(belisario_o3)
# hstd = np.std(belisario_o3)
# pdf = stats.norm.pdf(belisario_o3, hmean, hstd)
# plt.plot(belisario_o3, pdf)
# plt.show()

# data = [belisario_o3, belisario_o3_19]
# plt.boxplot(data)
# plt.savefig("/home/edward/DataScience_Projects/Air Pollution/Historical_data_analysis/O3/Gráficos O3/4.png")
# plt.show()

# districts = gpd.read_file(r"C:\Users\gaibo\Downloads\provincias\provincias.shp")

# fig, ax = plt.subplots(1,1)

# districts[districts.Id == "000026"].plot(ax=ax, legend=True)

# plt.show()

Wilcoxon_result_test = []

app_list = [wiltest_belisario, wiltest_carapungo, wiltest_centro, wiltest_chillos, wiltest_cotocollao, wiltest_elcamal, wiltest_guamani, wiltest_tumbaco]

for app in app_list:
    Wilcoxon_result_test.append(app)

print(Wilcoxon_result_test)