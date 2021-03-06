import csv
from itertools import islice
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import random
import numpy as np
from scipy.stats import ttest_rel, kstest, normaltest
import scipy.stats as stats
import statistics
from sklearn.preprocessing import PowerTransformer


plt.style.use('ggplot')

belisario_so2 = []
carapungo_so2 = []
centro_so2 = []
cotocollao_so2 = []
elcamal_so2 = []
guamani_so2 = []
chillos_so2 = []
tumbaco_so2 = []

belisario_so2_a = []
carapungo_so2_a = []
centro_so2_a = []
cotocollao_so2_a = []
elcamal_so2_a = []
guamani_so2_a = []
chillos_so2_a = []
tumbaco_so2_a = []

belisario_so2_19 = []
carapungo_so2_19 = []
centro_so2_19 = []
cotocollao_so2_19 = []
elcamal_so2_19 = []
guamani_so2_19 = []
chillos_so2_19 = []
tumbaco_so2_19 = []

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


#Read File == Resultados 1 mes durante la pandemia 2020
with open("/home/edward/DataScience_Projects/Air Pollution/SO2/SO2-N.csv", "r") as csv_file:
    f = csv.reader(csv_file, delimiter=";")
    for line in islice(f, 142131, 146570):
        belisario_so2.append(float(line[1]))
        carapungo_so2.append(float(line[2]))
        centro_so2.append(float(line[3]))
        cotocollao_so2.append(float(line[4]))
        elcamal_so2.append(float(line[5]))
        guamani_so2.append(float(line[6]))
        chillos_so2.append(float(line[7]))
        tumbaco_so2.append(float(line[8]))

#Read File == Resultados 1 mes antes la pandemia 2020
with open("/home/edward/DataScience_Projects/Air Pollution/SO2/SO2-N.csv", "r") as csv_file_antes:
    z = csv.reader(csv_file_antes, delimiter=";")
    for line in islice(z, 137691, 142130):
        belisario_so2_a.append(float(line[1]))
        carapungo_so2_a.append(float(line[2]))
        centro_so2_a.append(float(line[3]))
        cotocollao_so2_a.append(float(line[4]))
        elcamal_so2_a.append(float(line[5]))
        guamani_so2_a.append(float(line[6]))
        chillos_so2_a.append(float(line[7]))
        tumbaco_so2_a.append(float(line[8]))

#Read File == Resultados 1 mes 2019
with open("/home/edward/DataScience_Projects/Air Pollution/SO2/SO2-N.csv", "r") as csv_file_2019:
    c = csv.reader(csv_file_2019, delimiter=";")
    for line in islice(c, 133347, 137786):
        belisario_so2_19.append(float(line[1]))
        carapungo_so2_19.append(float(line[2]))
        centro_so2_19.append(float(line[3]))
        cotocollao_so2_19.append(float(line[4]))
        elcamal_so2_19.append(float(line[5]))
        guamani_so2_19.append(float(line[6]))
        chillos_so2_19.append(float(line[7]))
        tumbaco_so2_19.append(float(line[8]))

#!!!!FIGURE!!!!!

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex= True, figsize=(16,10))

#!!!!!!!!!!!!!!!


#Plot belisario_so2 CO
belisario_so2 = np.array(belisario_so2)
tiempo_b = range(1, len(belisario_so2) + 1)
tiempo_b = np.array(tiempo_b)
mask = (belisario_so2 > 0)

belisario_so2_a = np.array(belisario_so2_a)
tiempo_j = range(1, len(belisario_so2_a) + 1)
tiempo_j = np.array(tiempo_j)
mask_j = (belisario_so2_a > 0)

belisario_so2_19 = np.array(belisario_so2_19)
tiempo_r = range(1, len(belisario_so2_19) + 1)
tiempo_r = np.array(tiempo_r)
mask_r = (belisario_so2_19 > 0)

ax1.plot(tiempo_b[mask], belisario_so2[mask])
ax1.plot(tiempo_j[mask_j], belisario_so2_a[mask_j])
#ax1.plot(tiempo_r[mask_r], belisario_so2_19[mask_r])
ax1.set_title("Belisario")

# pt = PowerTransformer(method='box-cox')
# data = belisario_so2.reshape(-1, 1)
# pt.fit(data)
# transformed_data = pt.transform(data)

# k2, p = normaltest(data)
# transformed_k2, transformed_p = normaltest(transformed_data)

wiltest_belisario = stats.wilcoxon(belisario_so2, belisario_so2_a)

#Plot carapungo_so2 CO
carapungo_so2 = np.array(carapungo_so2)
tiempo_c = range(1, len(carapungo_so2) + 1)
tiempo_c = np.array(tiempo_c)
mask_c = (carapungo_so2 > 0)

carapungo_so2_a = np.array(carapungo_so2_a)
tiempo_k = range(1, len(carapungo_so2_a) + 1)
tiempo_k = np.array(tiempo_k)
mask_k = (carapungo_so2_a > 0)

carapungo_so2_19 = np.array(carapungo_so2_19)
tiempo_s = range(1, len(carapungo_so2_19) + 1)
tiempo_s = np.array(tiempo_s)
mask_s = (carapungo_so2_19 > 0)

wiltest_carapungo = stats.wilcoxon(carapungo_so2, carapungo_so2_a)

ax2.plot(tiempo_c[mask_c], carapungo_so2[mask_c])
ax2.plot(tiempo_k[mask_k], carapungo_so2_a[mask_k])
#ax2.plot(tiempo_s[mask_s], carapungo_so2_19[mask_s])
ax2.set_title("Carapungo")


#Plot centro_so2 CO
centro_so2 = np.array(centro_so2)
tiempo_d = range(1, len(centro_so2) + 1)
tiempo_d = np.array(tiempo_d)
mask_d = (centro_so2 > 0)

centro_so2_a = np.array(centro_so2_a)
tiempo_l = range(1, len(centro_so2_a) + 1)
tiempo_l = np.array(tiempo_l)
mask_l = (centro_so2_a > 0)

centro_so2_19 = np.array(centro_so2_19)
tiempo_t = range(1, len(centro_so2_19) + 1)
tiempo_t = np.array(tiempo_t)
mask_t = (centro_so2_19 > 0)

wiltest_centro = stats.wilcoxon(centro_so2, centro_so2_a)

ax3.plot(tiempo_d[mask_d], centro_so2[mask_d])
ax3.plot(tiempo_l[mask_l], centro_so2_a[mask_l])
#ax3.plot(tiempo_t[mask_t], centro_so2_19[mask_t])
ax3.set_title("Centro")

#Plot cotocollao_so2 CO
cotocollao_so2 = np.array(cotocollao_so2)
tiempo_e = range(1, len(cotocollao_so2) + 1)
tiempo_e = np.array(tiempo_e)
mask_e = (cotocollao_so2 > 0)

cotocollao_so2_a = np.array(cotocollao_so2_a)
tiempo_m = range(1, len(cotocollao_so2_a) + 1)
tiempo_m = np.array(tiempo_m)
mask_m = (cotocollao_so2_a > 0)

cotocollao_so2_19 = np.array(cotocollao_so2_19)
tiempo_u = range(1, len(cotocollao_so2_19) + 1)
tiempo_u = np.array(tiempo_u)
mask_u = (cotocollao_so2_19 > 0)

wiltest_cotocollao = stats.wilcoxon(cotocollao_so2, cotocollao_so2_a)

ax4.plot(tiempo_e[mask_e], cotocollao_so2[mask_e])
ax4.plot(tiempo_m[mask_m], cotocollao_so2_a[mask_m])
#ax4.plot(tiempo_u[mask_u], cotocollao_so2_19[mask_u])
ax4.set_title("Cotocollao")


# for ax in fig.get_axes():
#     ax.label_outer()

fig.text(0.5, 0.04, 'Tiempo (Horas)', ha='center')
fig.text(0.04, 0.5, '??ndices SO2 (mg/m3)', va='center', rotation='vertical')

fig.legend(["Durante", "Antes"])
plt.savefig("/home/edward/DataScience_Projects/Air Pollution/SO2/Gr??ficos SO2/1.png")
plt.show()


fig2, ((ax1, ax2)) = plt.subplots(1,2, sharex=True, figsize=(16,10))

#Plot elcamal_so2 CO
elcamal_so2 = np.array(elcamal_so2)
tiempo_f = range(1, len(elcamal_so2) + 1)
tiempo_f = np.array(tiempo_f)
mask_f = (elcamal_so2 > 0)

elcamal_so2_a = np.array(elcamal_so2_a)
tiempo_n = range(1, len(elcamal_so2_a) + 1)
tiempo_n = np.array(tiempo_n)
mask_n = (elcamal_so2_a > 0)

elcamal_so2_19 = np.array(elcamal_so2_19)
tiempo_v = range(1, len(elcamal_so2_19) + 1)
tiempo_v = np.array(tiempo_v)
mask_v = (elcamal_so2_19 > 0)

wiltest_elcamal = stats.wilcoxon(elcamal_so2, elcamal_so2_a)

ax1.plot(tiempo_f[mask_f], elcamal_so2[mask_f])
ax1.plot(tiempo_n[mask_n], elcamal_so2_a[mask_n])
#ax1.plot(tiempo_u[mask_v], elcamal_so2_19[mask_v])
ax1.set_title("El camal")

#Plot guamani_so2 CO
guamani_so2 = np.array(guamani_so2)
tiempo_g = range(1, len(guamani_so2) + 1)
tiempo_g = np.array(tiempo_f)
mask_g = (guamani_so2 > 0)

guamani_so2_a = np.array(guamani_so2_a)
tiempo_o = range(1, len(guamani_so2_a) + 1)
tiempo_o = np.array(tiempo_o)
mask_o = (guamani_so2_a > 0)

guamani_so2_19 = np.array(guamani_so2_19)
tiempo_w = range(1, len(guamani_so2_19) + 1)
tiempo_w = np.array(tiempo_w)
mask_w = (guamani_so2_19 > 0)

wiltest_guamani = stats.wilcoxon(guamani_so2, guamani_so2_a)

ax2.plot(tiempo_g[mask_g], guamani_so2[mask_g])
ax2.plot(tiempo_o[mask_o], guamani_so2_a[mask_o])
#ax2.plot(tiempo_w[mask_w], guamani_so2_19[mask_w])
ax2.set_title("Guamn??")

#Will take the same label axis
# for ax in fig2.get_axes():
#     ax.label_outer()

fig2.text(0.5, 0.04, 'Tiempo (Horas)', ha='center')
fig2.text(0.04, 0.5, '??ndices SO2 (mg/m3)', va='center', rotation='vertical')

fig2.legend(["Durante", "Antes"])
plt.savefig("/home/edward/DataScience_Projects/Air Pollution/SO2/Gr??ficos SO2/2.png")
plt.show()

fig3, ((ax3, ax4)) = plt.subplots(1,2, sharex=True, figsize=(16,10))
#Plot chillos_so2 CO
chillos_so2 = np.array(chillos_so2)
tiempo_h = range(1, len(chillos_so2) + 1)
tiempo_h = np.array(tiempo_h)
mask_h = (chillos_so2 > 0)

chillos_so2_a = np.array(chillos_so2_a)
tiempo_p = range(1, len(chillos_so2_a) + 1)
tiempo_p = np.array(tiempo_p)
mask_p = (chillos_so2_a > 0)

chillos_so2_19 = np.array(chillos_so2_19)
tiempo_x = range(1, len(chillos_so2_19) + 1)
tiempo_x = np.array(tiempo_x)
mask_x = (chillos_so2_19 > 0)

wiltest_chillos = stats.wilcoxon(chillos_so2, chillos_so2_a)

ax3.plot(tiempo_h[mask_h], chillos_so2[mask_h])
ax3.plot(tiempo_p[mask_p], chillos_so2_a[mask_p])
#ax3.plot(tiempo_x[mask_x], chillos_so2_19[mask_x])
ax3.set_title("Chillos")

#Plot tumbaco_so2 CO
tumbaco_so2 = np.array(tumbaco_so2)
tiempo_i = range(1, len(tumbaco_so2) + 1)
tiempo_i = np.array(tiempo_i)
mask_i = (tumbaco_so2 > 0)

tumbaco_so2_a = np.array(tumbaco_so2_a)
tiempo_q = range(1, len(tumbaco_so2_a) + 1)
tiempo_q = np.array(tiempo_q)
mask_q = (tumbaco_so2_a > 0)


tumbaco_so2_19 = np.array(tumbaco_so2_19)
tiempo_y = range(1, len(tumbaco_so2_19) + 1)
tiempo_y = np.array(tiempo_y)
mask_y = (tumbaco_so2_19 > 0)

wiltest_tumbaco = stats.wilcoxon(tumbaco_so2, tumbaco_so2_a)

ax4.plot(tiempo_i[mask_i], tumbaco_so2[mask_i])
ax4.plot(tiempo_q[mask_q], tumbaco_so2_a[mask_q])
#ax4.plot(tiempo_y[mask_y], tumbaco_so2_19[mask_y])
ax4.set_title("Tumbaco")


fig3.text(0.5, 0.04, 'Tiempo (Horas)', ha='center')
fig3.text(0.04, 0.5, '??ndices SO2 (mg/m3)', va='center', rotation='vertical')

fig3.legend(["Durante", "Antes"])
plt.savefig("/home/edward/DataScience_Projects/Air Pollution/SO2/Gr??ficos SO2/3.png")
plt.show()


# belisario_so2 = belisario_so2.tolist()

# hmean = np.mean(belisario_so2)
# hstd = np.std(belisario_so2)
# pdf = stats.norm.pdf(belisario_so2, hmean, hstd)
# plt.plot(belisario_so2, pdf)
# plt.show()



# data = [belisario_so2, belisario_so2_19]
# plt.boxplot(data)
# plt.savefig("C:\\Users\\gaibo\\OneDrive\\Escritorio\\Python Projects\\Air Pollution\\Historical_data_analysis\\SO2\\Gr??ficos SO2\\4.png")
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

medias_antes_durante(belisario_so2, carapungo_so2, centro_so2, cotocollao_so2, elcamal_so2, guamani_so2, chillos_so2, tumbaco_so2, belisario_so2_a, carapungo_so2_a, centro_so2_a, cotocollao_so2_a, elcamal_so2_a, guamani_so2_a, chillos_so2_a, tumbaco_so2_a)