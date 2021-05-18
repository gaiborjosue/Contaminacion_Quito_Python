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


plt.style.use('seaborn')

belisario_pm = []
carapungo_pm = []
centro_pm = []
cotocollao_pm = []
elcamal_pm = []
guamani_pm = []
chillos_pm = []
tumbaco_pm = []

belisario_pm_a = []
carapungo_pm_a = []
centro_pm_a = []
cotocollao_pm_a = []
elcamal_pm_a = []
guamani_pm_a = []
chillos_pm_a = []
tumbaco_pm_a = []

belisario_pm_19 = []
carapungo_pm_19 = []
centro_pm_19 = []
cotocollao_pm_19 = []
elcamal_pm_19 = []
guamani_pm_19 = []
chillos_pm_19 = []
tumbaco_pm_19 = []


#Read File == Resultados 1 mes durante la pandemia 2020
with open("/home/edward/DataScience_Projects/Air Pollution/PM2.5/PM2.5-N.csv", "r") as csv_file:
    f = csv.reader(csv_file, delimiter=",")
    for line in islice(f, 136404, 140843):
        belisario_pm.append(float(line[1]))
        carapungo_pm.append(float(line[2]))
        centro_pm.append(float(line[3]))
        cotocollao_pm.append(float(line[4]))
        elcamal_pm.append(float(line[5]))
        guamani_pm.append(float(line[6]))
        chillos_pm.append(float(line[7]))
        tumbaco_pm.append(float(line[8]))

#Read File == Resultados 1 mes antes la pandemia 2020
with open("/home/edward/DataScience_Projects/Air Pollution/PM2.5/PM2.5-N.csv", "r") as csv_file_antes:
    z = csv.reader(csv_file_antes, delimiter=",")
    for line in islice(z, 131964, 136403):
        belisario_pm_a.append(float(line[1]))
        carapungo_pm_a.append(float(line[2]))
        centro_pm_a.append(float(line[3]))
        cotocollao_pm_a.append(float(line[4]))
        elcamal_pm_a.append(float(line[5]))
        guamani_pm_a.append(float(line[6]))
        chillos_pm_a.append(float(line[7]))
        tumbaco_pm_a.append(float(line[8]))

#Read File == Resultados 1 mes 2019
with open("/home/edward/DataScience_Projects/Air Pollution/PM2.5/PM2.5-N.csv", "r") as csv_file_2019:
    c = csv.reader(csv_file_2019, delimiter=",")
    for line in islice(c, 127620, 132059):
        belisario_pm_19.append(float(line[1]))
        carapungo_pm_19.append(float(line[2]))
        centro_pm_19.append(float(line[3]))
        cotocollao_pm_19.append(float(line[4]))
        elcamal_pm_19.append(float(line[5]))
        guamani_pm_19.append(float(line[6]))
        chillos_pm_19.append(float(line[7]))
        tumbaco_pm_19.append(float(line[8]))

#!!!!FIGURE!!!!!

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, figsize=(16,10))

#!!!!!!!!!!!!!!!
print(len(belisario_pm), len(belisario_pm_a))

#Plot belisario_pm CO
belisario_pm = np.array(belisario_pm)
tiempo_b = range(1, len(belisario_pm) + 1)
tiempo_b = np.array(tiempo_b)
mask = (belisario_pm > 0)

belisario_pm_a = np.array(belisario_pm_a)
tiempo_j = range(1, len(belisario_pm_a) + 1)
tiempo_j = np.array(tiempo_j)
mask_j = (belisario_pm_a > 0)

belisario_pm_19 = np.array(belisario_pm_19)
tiempo_r = range(1, len(belisario_pm_19) + 1)
tiempo_r = np.array(tiempo_r)
mask_r = (belisario_pm_19 > 0)

ax1.plot(tiempo_b[mask], belisario_pm[mask])
ax1.plot(tiempo_j[mask_j], belisario_pm_a[mask_j])
#ax1.plot(tiempo_r[mask_r], belisario_pm_19[mask_r])
ax1.set_title("Belisario")

# pt = PowerTransformer(method='box-cox')
# data = belisario_pm.reshape(-1, 1)
# pt.fit(data)
# transformed_data = pt.transform(data)

# k2, p = normaltest(data)
# transformed_k2, transformed_p = normaltest(transformed_data)

wiltest_belisario = stats.wilcoxon(belisario_pm, belisario_pm_a)


#Plot carapungo_pm CO
carapungo_pm = np.array(carapungo_pm)
tiempo_c = range(1, len(carapungo_pm) + 1)
tiempo_c = np.array(tiempo_c)
mask_c = (carapungo_pm > 0)

carapungo_pm_a = np.array(carapungo_pm_a)
tiempo_k = range(1, len(carapungo_pm_a) + 1)
tiempo_k = np.array(tiempo_k)
mask_k = (carapungo_pm_a > 0)

carapungo_pm_19 = np.array(carapungo_pm_19)
tiempo_s = range(1, len(carapungo_pm_19) + 1)
tiempo_s = np.array(tiempo_s)
mask_s = (carapungo_pm_19 > 0)

wiltest_carapungo = stats.wilcoxon(carapungo_pm, carapungo_pm_a)

ax2.plot(tiempo_c[mask_c], carapungo_pm[mask_c])
ax2.plot(tiempo_k[mask_k], carapungo_pm_a[mask_k])
#ax2.plot(tiempo_s[mask_s], carapungo_pm_19[mask_s])
ax2.set_title("Carapungo")


#Plot centro_pm CO
centro_pm = np.array(centro_pm)
tiempo_d = range(1, len(centro_pm) + 1)
tiempo_d = np.array(tiempo_d)
mask_d = (centro_pm > 0)

centro_pm_a = np.array(centro_pm_a)
tiempo_l = range(1, len(centro_pm_a) + 1)
tiempo_l = np.array(tiempo_l)
mask_l = (centro_pm_a > 0)

centro_pm_19 = np.array(centro_pm_19)
tiempo_t = range(1, len(centro_pm_19) + 1)
tiempo_t = np.array(tiempo_t)
mask_t = (centro_pm_19 > 0)

wiltest_centro = stats.wilcoxon(centro_pm, centro_pm_a)

ax3.plot(tiempo_d[mask_d], centro_pm[mask_d])
ax3.plot(tiempo_l[mask_l], centro_pm_a[mask_l])
#ax3.plot(tiempo_t[mask_t], centro_pm_19[mask_t])
ax3.set_title("Centro")

#Plot cotocollao_pm CO
cotocollao_pm = np.array(cotocollao_pm)
tiempo_e = range(1, len(cotocollao_pm) + 1)
tiempo_e = np.array(tiempo_e)
mask_e = (cotocollao_pm > 0)

cotocollao_pm_a = np.array(cotocollao_pm_a)
tiempo_m = range(1, len(cotocollao_pm_a) + 1)
tiempo_m = np.array(tiempo_m)
mask_m = (cotocollao_pm_a > 0)

cotocollao_pm_19 = np.array(cotocollao_pm_19)
tiempo_u = range(1, len(cotocollao_pm_19) + 1)
tiempo_u = np.array(tiempo_u)
mask_u = (cotocollao_pm_19 > 0)

wiltest_cotocollao = stats.wilcoxon(cotocollao_pm, cotocollao_pm_a)

ax4.plot(tiempo_e[mask_e], cotocollao_pm[mask_e])
ax4.plot(tiempo_m[mask_m], cotocollao_pm_a[mask_m])
#ax4.plot(tiempo_u[mask_u], cotocollao_pm_19[mask_u])
ax4.set_title("Cotocollao")


# for ax in fig.get_axes():
#     ax.label_outer()

fig.text(0.5, 0.04, 'Tiempo (Horas)', ha='center')
fig.text(0.04, 0.5, 'Índices PM2.5 (mg/m3)', va='center', rotation='vertical')

fig.legend(["Durante", "Antes"])
plt.savefig("/home/edward/DataScience_Projects/Air Pollution/PM2.5/Gráficos PM2.5/1.png")
plt.show()


fig2, ((ax1, ax2)) = plt.subplots(1,2, sharex=True, figsize=(16,10))

#Plot elcamal_pm CO
elcamal_pm = np.array(elcamal_pm)
tiempo_f = range(1, len(elcamal_pm) + 1)
tiempo_f = np.array(tiempo_f)
mask_f = (elcamal_pm > 0)

elcamal_pm_a = np.array(elcamal_pm_a)
tiempo_n = range(1, len(elcamal_pm_a) + 1)
tiempo_n = np.array(tiempo_n)
mask_n = (elcamal_pm_a > 0)

elcamal_pm_19 = np.array(elcamal_pm_19)
tiempo_v = range(1, len(elcamal_pm_19) + 1)
tiempo_v = np.array(tiempo_v)
mask_v = (elcamal_pm_19 > 0)

wiltest_elcamal = stats.wilcoxon(elcamal_pm, elcamal_pm_a)

ax1.plot(tiempo_f[mask_f], elcamal_pm[mask_f])
ax1.plot(tiempo_n[mask_n], elcamal_pm_a[mask_n])
#ax1.plot(tiempo_u[mask_v], elcamal_pm_19[mask_v])
ax1.set_title("El camal")

#Plot guamani_pm CO
guamani_pm = np.array(guamani_pm)
tiempo_g = range(1, len(guamani_pm) + 1)
tiempo_g = np.array(tiempo_f)
mask_g = (guamani_pm > 0)

guamani_pm_a = np.array(guamani_pm_a)
tiempo_o = range(1, len(guamani_pm_a) + 1)
tiempo_o = np.array(tiempo_o)
mask_o = (guamani_pm_a > 0)

guamani_pm_19 = np.array(guamani_pm_19)
tiempo_w = range(1, len(guamani_pm_19) + 1)
tiempo_w = np.array(tiempo_w)
mask_w = (guamani_pm_19 > 0)

wiltest_guamani = stats.wilcoxon(guamani_pm, guamani_pm_a)

ax2.plot(tiempo_g[mask_g], guamani_pm[mask_g])
ax2.plot(tiempo_o[mask_o], guamani_pm_a[mask_o])
#ax2.plot(tiempo_w[mask_w], guamani_pm_19[mask_w])
ax2.set_title("Guamní")

#Will take the same label axis
# for ax in fig2.get_axes():
#     ax.label_outer()
fig2.text(0.5, 0.04, 'Tiempo (Horas)', ha='center')
fig2.text(0.04, 0.5, 'Índices PM2.5 (mg/m3)', va='center', rotation='vertical')

fig2.legend(["Durante", "Antes"])
plt.savefig("/home/edward/DataScience_Projects/Air Pollution/PM2.5/Gráficos PM2.5/2.png")
plt.show()

fig3, ((ax3, ax4)) = plt.subplots(1,2, sharex=True, figsize=(16,10))
#Plot chillos_pm CO
chillos_pm = np.array(chillos_pm)
tiempo_h = range(1, len(chillos_pm) + 1)
tiempo_h = np.array(tiempo_h)
mask_h = (chillos_pm > 0)

chillos_pm_a = np.array(chillos_pm_a)
tiempo_p = range(1, len(chillos_pm_a) + 1)
tiempo_p = np.array(tiempo_p)
mask_p = (chillos_pm_a > 0)

chillos_pm_19 = np.array(chillos_pm_19)
tiempo_x = range(1, len(chillos_pm_19) + 1)
tiempo_x = np.array(tiempo_x)
mask_x = (chillos_pm_19 > 0)

wiltest_chillos = stats.wilcoxon(chillos_pm, chillos_pm_a)

ax3.plot(tiempo_h[mask_h], chillos_pm[mask_h])
ax3.plot(tiempo_p[mask_p], chillos_pm_a[mask_p])
#ax3.plot(tiempo_x[mask_x], chillos_pm_19[mask_x])
ax3.set_title("Chillos")

#Plot tumbaco_pm CO
tumbaco_pm = np.array(tumbaco_pm)
tiempo_i = range(1, len(tumbaco_pm) + 1)
tiempo_i = np.array(tiempo_i)
mask_i = (tumbaco_pm > 0)

tumbaco_pm_a = np.array(tumbaco_pm_a)
tiempo_q = range(1, len(tumbaco_pm_a) + 1)
tiempo_q = np.array(tiempo_q)
mask_q = (tumbaco_pm_a > 0)


tumbaco_pm_19 = np.array(tumbaco_pm_19)
tiempo_y = range(1, len(tumbaco_pm_19) + 1)
tiempo_y = np.array(tiempo_y)
mask_y = (tumbaco_pm_19 > 0)

wiltest_tumbaco = stats.wilcoxon(tumbaco_pm, tumbaco_pm_a)

ax4.plot(tiempo_i[mask_i], tumbaco_pm[mask_i])
ax4.plot(tiempo_q[mask_q], tumbaco_pm_a[mask_q])
#ax4.plot(tiempo_y[mask_y], tumbaco_pm_19[mask_y])
ax4.set_title("Tumbaco")

fig3.text(0.5, 0.04, 'Tiempo (Horas)', ha='center')
fig3.text(0.04, 0.5, 'Índices PM2.5 (mg/m3)', va='center', rotation='vertical')


fig3.legend(["Durante", "Antes"])

plt.savefig("/home/edward/DataScience_Projects/Air Pollution/PM2.5/Gráficos PM2.5/3.png")
plt.show()


# belisario_pm = belisario_pm.tolist()

# hmean = np.mean(belisario_pm)
# hstd = np.std(belisario_pm)
# pdf = stats.norm.pdf(belisario_pm, hmean, hstd)
# plt.plot(belisario_pm, pdf)
# plt.show()




# data = [belisario_pm, belisario_pm_19]
# plt.boxplot(data)
# plt.savefig("C:\\Users\\gaibo\\OneDrive\\Escritorio\\Python Projects\\Air Pollution\\Historical_data_analysis\\PM2.5\\Gráficos PM2.5\\4.png")
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