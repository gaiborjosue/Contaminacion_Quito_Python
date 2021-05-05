import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import cartopy.crs as ccrs
from scipy import stats

csv_path = 'C:\\Users\\gaibo\\OneDrive\\Escritorio\\Python Projects\\Air Pollution\\Historical_data_analysis\\guamaní,-quito-air-quality.csv'

h = pd.read_csv(csv_path)

modified_h = h.dropna()
modified_h.to_csv("guamaní_n,-quito-air-quality.csv", index=False)

new_csvpath = 'C:\\Users\\gaibo\\OneDrive\\Escritorio\\Python Projects\\Air Pollution\\Historical_data_analysis\\guamaní_n,-quito-air-quality.csv'
#!!!!!!Extracting and analyzing days of coronavirus!!!!!!
df = pd.read_csv(csv_path)


#columns of the csv file
df.columns
df = df.rename(columns = {" pm25": "pm25",
" pm10":"pm10", " o3":"o3", " no2":"no2", " so2":"so2", " co":"co"})      


plt.style.use('ggplot')

# print(plt.style.available)

#Stablish the range of time that we want to analize, in this case: 22 days.
df['date'] = pd.to_datetime(df.date)

df50 = (df['date'] > '2020-03-12') & (df['date'] < '2020-04-04')
df50 = df.loc[df50]


df50 = df50.sort_values(by = 'date')

df50.replace(' ', '0', inplace=True)
print("\n    22 dias durante la restriccion (Cuarentena)    \n")
print(df50)

max_value = df50.max()
print('\nValor Máximo Durante la restricción\n------')
print(max_value)

min_value = df50.min()
print('\nValor Mínimo Durante la restricción\n------')
print(min_value)



x = [0,22]
#Delete or replace the extra data.
#df50.drop(27, inplace=True)
#print(df50)

#Fill the empty spaces with zeros.


#plot the data of pm25

dates = df50['date']
pm25 = df50['pm25']
pm25 = [int(i) for i in pm25]

# so2 = df50['so2']
# so2 = [int(i) for i in so2]

# plt.figure(figsize=(10,8))

# plt.plot(dates,pm25,color='green')

# plt.title('PM 2.5 values in lockdown days')
# plt.xlabel('Dates of lockdown')
# plt.ylabel('PM 2.5 values')
# # plt.show()

#!!!!!!!Extracting days before coronavirus, to compare. PM 2.5

mask = (df['date'] >= '2020-02-20') & (df['date'] < '2020-03-13')

past22 = df.loc[mask]
past22 = past22.sort_values(by = 'date')
print("\n    22 dias antes de la restriccion (Cuarentena)    \n")
print(past22)


max_value_2 = past22.max()
print('\nValor Máximo Anterior a la restricción\n------')
print(max_value_2)

min_value_2 = past22.min()
print('\nValor Mínimo Anterior a la restricción\n------')
print(min_value_2)


mask_2 = (df['date'] > '2019-03-12') & (df['date'] < '2019-04-04')

all_2019 = df.loc[mask_2]
all_2019 = all_2019.sort_values(by = 'date')

print("\n    Variable de control [Periodo 2019]   \n")
print(all_2019)

max_value_3 = all_2019.max()
print('\nValor Máximo V.Control 2019\n------')
print(max_value_3)

min_value_3 = all_2019.min()
print('\nValor Mínimo V.Control 2019\n------')
print(min_value_3)


#Comparing both segmentations

inputq = input("¿Quieres visualizar y guardar los graficos del análisis?[Y/N] ")

if 'Y' in inputq:
    print("Opening Data...")
    dates = df50['date']
    pm25_l = df50['pm25']
    pm25_l = [float(i) for i in pm25_l]

    pm25_date2 = (df['date'] > '2019-03-12') & (df['date'] < '2019-04-04')
    pm25_date2 = df.loc[pm25_date2]
    pm25_date2 = pm25_date2.sort_values(by = 'date')

    pm25_during2019 = pm25_date2['pm25']
    pm25_during2019 = [float(i) for i in pm25_during2019]


    pm25_n = past22['pm25']
    pm25_n = [float(i) for i in pm25_n]

    plt.figure(figsize=(10,8))

    length = [i for i in range(1,len(dates)+1)]
    plt.plot(length,pm25_l,linestyle="-" , color = '#407294', marker='o',linewidth=4,label='Durante la restriccion (Cuarentena)')
    plt.plot(length,pm25_n, linestyle ='-',marker='.',color='#FFA500',linewidth=4,label='Antes de la restriccion (Cuarentena)')
    plt.plot(length,pm25_during2019,linestyle = '--',color='#66CDAA',label='Variable de Control 2019')

    plt.xticks(np.arange(min(x), max(x)+1, 1.0))

    plt.legend()
    plt.title('Polución por: (PM 2.5), antes y durante la restricción')
    plt.xlabel('Dias')
    plt.ylabel('Valores de PM 2.5[AQI]')
    plt.plot()
    #plt.grid(True)
    #plt.savefig('C:\\Users\\gaibo\\OneDrive\\Escritorio\\Python Projects\\Air Pollution\\Historical_data_analysis\\Graficos\\Plot PM25.png')

    plt.show()

    #Comparing CO values.

    co_date = (df['date'] > '2020-03-12') & (df['date'] < '2020-04-04')
    co_date = df.loc[co_date]

    co_date = co_date.sort_values(by = 'date')

    co_date2 = (df['date'] > '2019-03-12') & (df['date'] < '2019-04-04')
    co_date2 = df.loc[co_date2]
    co_date2 = co_date2.sort_values(by = 'date')

    co_during2019 = co_date2['co']
    
    co_during2019 = [float(i) for i in co_during2019]

    co_during = co_date['co']
    # co_during.replace(' ','0', inplace=True)
    co_during = [float(i) for i in co_during]



    co_past = past22['co']
    co_past = [float(i) for i in co_past]

    plt.figure(figsize=(10,8))

    length = [i for i in range(1,len(dates)+1)]
    plt.plot(length,co_during,linestyle ="-",marker='o',color='#407294',linewidth=4,label='Durante la restriccion (Cuarentena)')
    plt.plot(length,co_past,linestyle ='-',marker='.',color='#FFA500',linewidth=4,label='Antes de la restriccion (Cuarentena)')
    plt.plot(length,co_during2019,linestyle = '--',color='#66CDAA',label='Variable de Control 2019')

    plt.xticks(np.arange(min(x), max(x)+1, 1.0))

    plt.legend()
    plt.title('Polución por: (co), antes y durante la restricción')
    plt.xlabel('Dias')
    plt.ylabel('Valores de CO[AQI]')
    plt.plot()
    #plt.grid(True)
    #plt.savefig('C:\\Users\\gaibo\\OneDrive\\Escritorio\\Python Projects\\Air Pollution\\Historical_data_analysis\\Graficos\\Plot CO.png')
    plt.show()

    #Comparing O3 values.

    o3_date = (df['date'] > '2020-03-12') & (df['date'] < '2020-04-04')
    o3_date = df.loc[o3_date]

    o3_date = o3_date.sort_values(by = 'date')

    o3_date2 = (df['date'] > '2019-03-12') & (df['date'] < '2019-04-04')
    o3_date2 = df.loc[o3_date2]
    o3_date2 = o3_date2.sort_values(by = 'date')

    o3_during2019 = o3_date2['o3']
    o3_during2019 = [float(i) for i in o3_during2019]

    o3_during = o3_date['o3']
    o3_during = [float(i) for i in o3_during]

    # co_during.replace(' ','0', inplace=True)

    o3_past = past22['o3']
    o3_past = [float(i) for i in o3_past]

    plt.figure(figsize=(10,8))

    length = [i for i in range(1,len(dates)+1)]

    plt.plot(length,o3_during, linestyle = "-",marker='o',color='#407294',linewidth=4,label='Durante la restriccion (2020)')
    plt.plot(length,o3_past,linestyle ='-',marker='.',color='#FFA500',linewidth=4,label='Antes de la restriccion (2020)')
    plt.plot(length,o3_during2019,linestyle = '--',color='#66CDAA',label='Variable de Control 2019')

    plt.xticks(np.arange(min(x), max(x)+1, 1.0))

    plt.legend()
    plt.title('Polución por: (o3), antes y durante la restricción')
    plt.xlabel('Dias')
    plt.ylabel('Valores de O3[AQI]')

    plt.plot()
    #plt.grid(True)
    #plt.savefig('C:\\Users\\gaibo\\OneDrive\\Escritorio\\Python Projects\\Air Pollution\\Historical_data_analysis\\Graficos\\Plot o3.png')
    plt.show()



    #Comparing So2
    so2_date = (df['date'] > '2020-03-12') & (df['date'] < '2020-04-04')
    so2_date = df.loc[so2_date]

    so2_date = so2_date.sort_values(by = 'date')

    so2_date2 = (df['date'] > '2019-03-12') & (df['date'] < '2019-04-04')
    so2_date2 = df.loc[so2_date2]
    so2_date2 = so2_date2.sort_values(by = 'date')

    so2_during2019 = so2_date2['so2']
    so2_during2019 = [float(i) for i in so2_during2019]

    so2_during = so2_date['so2']
    so2_during = [float(i) for i in so2_during]


    # co_during.replace(' ','0', inplace=True)

    so2_past = past22['so2']
    so2_past = [float(i) for i in so2_past]

    plt.figure(figsize=(10,8))

    length = [i for i in range(1,len(dates)+1)]

    x_indexes = np.arange(len(length))
    width = 0.25

    plt.plot(length,so2_during,linestyle = "-",marker='o', color='#407294',linewidth=4,label='Durante la restriccion (2020)')
    plt.plot(length,so2_past, linestyle ='-',marker='.',color='#FFA500',linewidth=4,label='Antes de la restriccion (2020)')
    plt.plot(length,so2_during2019,linestyle = '--',color='#66CDAA',label='Variable de Control 2019')


    plt.xticks(np.arange(min(x), max(x)+1, 1.0))

    plt.legend()

    plt.title('Polución por: (so2), antes y durante la restricción')

    plt.xlabel('Dias')
    plt.ylabel('Valores de SO2[AQI]')

    plt.plot()
    #plt.grid(True)
    #plt.savefig('C:\\Users\\gaibo\\OneDrive\\Escritorio\\Python Projects\\Air Pollution\\Historical_data_analysis\\Graficos\\Plot so2.png')
    plt.show()


    ###BAR CHARTS###


    #PM25 BAR CHART

    plt.bar(x_indexes - width,pm25_l, color='#407294', width=width, label='Durante la restriccion (2020)')
    plt.bar(x_indexes,pm25_n, width=width, color='#FFA500', label='Antes de la restriccion (2020)')
    plt.bar(x_indexes + width,pm25_during2019, width=width, color='#66CDAA',label='Variable de Control 2019')

    plt.xticks(np.arange(min(x), max(x)+1, 1.0))

    plt.legend()

    plt.title('Polución por: (PM 2.5), antes y durante la restricción')

    plt.xlabel('Dias')
    plt.ylabel('Valores de PM 2.5[AQI]')

    plt.xticks(ticks=x_indexes, labels=length)

    plt.plot()
    #plt.grid(True)
    #plt.savefig('C:\\Users\\gaibo\\OneDrive\\Escritorio\\Python Projects\\Air Pollution\\Historical_data_analysis\\Graficos\\BAR pm25.png')
    plt.show()

    #CO BAR CHART

    plt.bar(x_indexes - width,co_during, color='#407294', width=width, label='Durante la restriccion (2020)')
    plt.bar(x_indexes,co_past, width=width, color='#FFA500', label='Antes de la restriccion (2020)')
    plt.bar(x_indexes + width,co_during2019, width=width, color='#66CDAA',label='Variable de Control 2019')

    plt.xticks(np.arange(min(x), max(x)+1, 1.0))

    plt.legend()

    plt.title('Polución por: (co), antes y durante la restricción')

    plt.xlabel('Dias')
    plt.ylabel('Valores de CO[AQI]')

    plt.xticks(ticks=x_indexes, labels=length)

    plt.plot()
    #plt.grid(True)
    #plt.savefig('C:\\Users\\gaibo\\OneDrive\\Escritorio\\Python Projects\\Air Pollution\\Historical_data_analysis\\Graficos\\BAR co.png')
    plt.show()


    #O3 BAR CHART

    plt.bar(x_indexes - width,o3_during, color='#407294', width=width, label='Durante la restriccion (2020)')
    plt.bar(x_indexes,o3_past, width=width, color='#FFA500', label='Antes de la restriccion (2020)')
    plt.bar(x_indexes + width,o3_during2019, width=width, color='#66CDAA',label='Variable de Control 2019')

    plt.xticks(np.arange(min(x), max(x)+1, 1.0))

    plt.legend()

    plt.title('Polución por: (o3), antes y durante la restricción')

    plt.xlabel('Dias')
    plt.ylabel('Valores de O3[AQI]')

    plt.xticks(ticks=x_indexes, labels=length)

    plt.plot()
    #plt.grid(True)
    #plt.savefig('C:\\Users\\gaibo\\OneDrive\\Escritorio\\Python Projects\\Air Pollution\\Historical_data_analysis\\Graficos\\BAR o3.png')
    plt.show()

    #SO2 BAR CHART

    plt.bar(x_indexes - width,so2_during, color='#407294', width=width, label='Durante la restriccion (2020)')
    plt.bar(x_indexes,so2_past, width=width, color='#FFA500', label='Antes de la restriccion (2020)')
    plt.bar(x_indexes + width,so2_during2019, width=width, color='#66CDAA',label='Variable de Control 2019')

    plt.xticks(np.arange(min(x), max(x)+1, 1.0))

    plt.legend()

    plt.title('Polución por: (so2), antes y durante la restricción')

    plt.xlabel('Dias')
    plt.ylabel('Valores de SO2[AQI]')

    plt.xticks(ticks=x_indexes, labels=length)

    plt.plot()
    #plt.grid(True)
    #plt.savefig('C:\\Users\\gaibo\\OneDrive\\Escritorio\\Python Projects\\Air Pollution\\Historical_data_analysis\\Graficos\\BAR so2.png')
    plt.show()

    #PIE CHART DURING RESTRICCION

    mean_pm25d = np.mean(pm25_l)
    mean_so2d = np.mean(so2_during)
    mean_cod = np.mean(co_during)
    mean_o3d = np.mean(o3_during)

    values = [mean_pm25d, mean_so2d, mean_cod, mean_o3d]
    labels = ['PM 2.5[AQI]', 'SO2[AQI]', 'CO[AQI]', 'O3[AQI]']
    explode = [0 , 0, 0, 0.1]
    colors = ['#49A7C3', '#99AAB5', '#F3C366', '#93B793']


    plt.pie(values, colors=colors,autopct='%1.1f%%',explode=explode, shadow=True)

    plt.legend(labels, loc='best')
    plt.title("Factores de contaminacion(Durante la restriccion 2020)")
    #plt.savefig('C:\\Users\\gaibo\\OneDrive\\Escritorio\\Python Projects\\Air Pollution\\Historical_data_analysis\\Graficos\\Pastel durante.png')
    plt.show()

    #PIE CHART AFTER RESTRICCION
    mean_pm25 = np.mean(pm25_n)
    mean_so2 = np.mean(so2_past)
    mean_co = np.mean(co_past)
    mean_o3 = np.mean(o3_past)

    values = [mean_pm25, mean_so2, mean_co, mean_o3]
    labels = ['PM 2.5[AQI]', 'SO2[AQI]', 'CO[AQI]', 'O3[AQI]']
    explode = [0.1, 0, 0, 0]

    plt.pie(values, explode=explode, autopct='%1.1f%%', shadow=True)

    plt.legend(labels, loc="best")
    plt.title("Factores de contaminacion (Antes de la restriccion 2020)")

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #plt.savefig('C:\\Users\\gaibo\\OneDrive\\Escritorio\\Python Projects\\Air Pollution\\Historical_data_analysis\\Graficos\\Pastel antes.png')

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    plt.show()

    print("---------Los graficos se abrieron y guardaron correctamente---------")
    
    a = input('Necesita mas informacion?[Y/N] ')

    if 'Y' in a:
        pregunta = input("Que otros datos necesitas saber sobre el análisis? ------> (Escribe 'help' para acceder a los commandos.) o (Escribe 'exit' para salir del programa.)  ")

        if 'help' in pregunta:
            print("\n------Lista de comandos: ------\n")
            print("media durante = Este comando imprimira el valor de la media de los datos para el factor contaminante que necesites durante la restriccion 2020.\n")
            print('-----------------------------------------------------------------')
            print("media antes = Este comando imprimira el valor de la media de los datos para el factor contaminante que necesites antes de la restriccion 2020.\n")
            print('-----------------------------------------------------------------')
            print("media 2019 = Este comando imprimira el valor de la media de los datos para el factor contaminante que necesites durante la temporada 2019, es decir la Variable de Control.\n")
            print('-----------------------------------------------------------------')
            print("moda durante = Este comando imprimira el valor de la moda de los datos para el factor contaminante que necesites durante la restriccion 2020.\n")
            print('-----------------------------------------------------------------')
            print("moda antes = Este comando imprimira el valor de la moda de los datos para el factor contaminante que necesites antes de la restriccion 2020.\n")
            print('-----------------------------------------------------------------')
            print("moda 2019 = Este comando imprimira el valor de la moda de los datos para el factor contaminante que necesites durante la temporada 2019, es decir la Variable de Control.\n")
            print('-----------------------------------------------------------------')
            print("mediana durante = Este comando imprimira el valor de la mediana de los datos para el factor contaminante que necesites durante la restriccion 2020.\n")
            print('-----------------------------------------------------------------')
            print("mediana antes = Este comando imprimira el valor de la mediana de los datos para el factor contaminante que necesites antes de la restriccion 2020.\n")
            print('-----------------------------------------------------------------')
            print("mediana 2019 = Este comando imprimira el valor de la mediana de los datos para el factor contaminante que necesites durante la temporada 2019, es decir la Variable de control.\n")
            print('-----------------------------------------------------------------')
            print("desviacion estandar durante = Este comando imprimira el valor de la desviacion estandar de los datos, para el factor contaminante que necesites durante la restriccion 2020.\n")
            print('-----------------------------------------------------------------')
            print("desviacion estandar antes = Este comando imprimira el valor de la desviacion estandar de los datos, para el factor contaminante que necesites antes de la restriccion 2020.\n")
            print('-----------------------------------------------------------------')
            print("desviacion estandar 2019 = Este comando imprimira el valor de la desviacion estandar de los datos para el factor contaminante que necesites durante la temporada 2019, es decir la Variable de control.\n")
            print('****************************************')
            respuesta = input("Ingrese el comando aqui: ")
            print('****************************************')

            if 'media durante' in respuesta:
                ask = input("De que factor necesitas la informacion?: ")
                if 'pm25' in ask:
                    print('La media es ',mean_pm25d)
                elif 'o3' in ask:
                    print('La media es ',mean_o3d)
                elif 'so2' in ask:
                    print('La media es ',mean_so2d)
                elif 'co' in ask:
                    print('La media es ',mean_cod)
                else:
                    print("Sorry, we dont have information for that, please try again.")

            if 'media antes' in respuesta:
                ask = input("De que factor necesitas la informacion?: ")
                if 'pm25' in ask:
                    print('La media es ',mean_pm25)
                elif 'o3' in ask:
                    print('La media es ',mean_o3)
                elif 'so2' in ask:
                    print('La media es ',mean_so2)
                elif 'co' in ask:
                    print('La media es ',mean_co)
                else:
                    print("Sorry, we dont have information for that, please try again.")

            if 'media 2019' in respuesta:
                ask = input("De que factor necesita la informacion?: ")
                if 'pm25' in ask:
                    print('La media es ',np.mean(pm25_during2019))
                elif 'o3' in ask:
                    print('La media es ',np.mean(o3_during2019))
                elif 'so2' in ask:
                    print('La media es ',np.mean(so2_during2019))
                elif 'co' in ask:
                    print('La media es ', np.mean(co_during2019))
                else:
                    print("Sorry, we dont have information for that, please try again.")

            if 'moda durante' in respuesta:
                ask = input("De que factor necesitas la informacion?: ")
                if 'pm25' in ask:
                    mod_pm25d = stats.mode(pm25_l)
                    print('La moda es ', mod_pm25d[0])
                elif 'o3' in ask:
                    mod_o3d = stats.mode(o3_during)
                    print('La moda es ', mod_o3d[0])
                elif 'so2' in ask:
                    mod_so2d = stats.mode(so2_during)
                    print('La moda es ', mod_so2d[0])
                elif 'co' in ask:
                    mod_cod = stats.mode(co_during)
                    print('La moda es ', mod_cod[0])
                else:
                    print("Sorry, we dont have information for that, please try again.")
            
            if 'moda antes' in respuesta:
                ask = input("De que factor necesitas la informacion?: ")
                if 'pm25' in ask:
                    mod_pm25a = stats.mode(pm25_n)
                    print('La moda es ', mod_pm25a[0])
                elif 'o3' in ask:
                    mod_o3a = stats.mode(o3_past)
                    print('La moda es ', mod_o3a[0])
                elif 'so2' in ask:
                    mod_so2a = stats.mode(so2_past)
                    print('La moda es ', mod_so2a[0])
                elif 'co' in ask:
                    mod_coa = stats.mode(co_past)
                    print('La moda es ', mod_coa[0])
                else:
                    print("Sorry, we dont have information for that, please try again.")

            if 'moda 2019' in respuesta:
                ask = input("De que factor necesitas la informacion?: ")
                if 'pm25' in ask:
                    mod_pm252 = stats.mode(pm25_during2019)
                    print('La moda es ', mod_pm252[0])
                elif 'o3' in ask:
                    mod_o32 = stats.mode(o3_during2019)
                    print('La moda es ', mod_o32[0])
                elif 'so2' in ask:
                    mod_so22 = stats.mode(so2_during2019)
                    print('La moda es ', mod_so22[0])
                elif 'co' in ask:
                    mod_co2 = stats.mode(co_during2019)
                    print('La moda es ', mod_co2[0])
                else:
                    print("Sorry, we dont have information for that, please try again.")

            if 'mediana durante' in respuesta:
                ask = input("De que factor necesitas la informacion?: ")
                if 'pm25' in ask:
                    med_pm25 = np.median(pm25_l)
                    print('La mediana es ',med_pm25)
                elif 'o3' in ask:
                    med_o3 = np.median(o3_during)
                    print('La mediana es ',med_o3)
                elif 'so2' in ask:
                    med_so2 = np.median(so2_during)
                    print('La mediana es ',med_so2)
                elif 'co' in ask:
                    med_co = np.median(co_during)
                    print('La mediana es ',med_co)
                else:
                    print("Sorry, we dont have information for that, please try again.")

            if 'mediana antes' in respuesta:
                ask = input("De que factor necesitas la informacion?: ")
                if 'pm25' in ask:
                    med_pm25a = np.median(pm25_n)
                    print('La mediana es ',med_pm25a)
                elif 'o3' in ask:
                    med_o3a = np.median(o3_past)
                    print('La mediana es ',med_o3a)
                elif 'so2' in ask:
                    med_so2a = np.median(so2_past)
                    print('La mediana es ',med_so2a)
                elif 'co' in ask:
                    med_coa = np.median(co_past)
                    print('La mediana es ',med_coa)
                else:
                    print("Sorry, we dont have information for that, please try again.")

            if 'mediana 2019' in respuesta:
                ask = input("De que factor necesitas la informacion?: ")
                if 'pm25' in ask:
                    med_pm252 = np.median(pm25_during2019)
                    print('La mediana es ',med_pm252)
                elif 'o3' in ask:
                    med_o32 = np.median(o3_during2019)
                    print('La mediana es ',med_o32)
                elif 'so2' in ask:
                    med_so22 = np.median(so2_during2019)
                    print('La mediana es ',med_so22)
                elif 'co' in ask:
                    med_co2 = np.median(co_during2019)
                    print('La mediana es ',med_co2)
                else:
                    print("Sorry, we dont have information for that, please try again.")

            if 'desviacion estandar durante' in respuesta:
                ask = input("De que factor necesita la informacion?: ")
                if 'pm25' in ask:
                    des_pmd = np.std(pm25_l)
                    print('La desviacion estandar es ',des_pmd)
                elif 'o3' in ask:
                    des_od = np.std(o3_during)
                    print('La desviacion estandar es ',des_od)
                elif 'so2' in ask:
                    des_sod = np.std(so2_during)
                    print('La desviacion estandar es ',des_sod)
                elif 'co' in ask:
                    des_cod = np.std(co_during)
                    print('La desviacion estandar es ',des_cod)
                else:
                    print("Sorry, we dont have information for that, please try again.")

            if 'desviacion estandar antes' in respuesta:
                ask = input("De que factor necesita la informacion?: ")
                if 'pm25' in ask:
                    des_pma = np.std(pm25_n)
                    print('La desviacion estandar es ', des_pma)
                elif 'o3' in ask:
                    des_oa = np.std(o3_past)
                    print('La desviacion estandar es ',des_oa)
                elif 'so2' in ask:
                    des_soa = np.std(so2_past)
                    print('La desviacion estandar es ',des_soa)
                elif 'co' in ask:
                    des_coa = np.std(co_past)
                    print('La desviacion estandar es ', des_coa)
                else:
                    print("Sorry, we dont have information for that, please try again.")

            if 'desviacion estandar 2019' in respuesta:
                ask = input("De que factor necesita la informacion?: ")
                if 'pm25' in ask:
                    des_pm2 = np.std(pm25_during2019)
                    print('La desviacion estandar es ', des_pm2)
                elif 'o3' in ask:
                    des_o32 = np.std(o3_during2019)
                    print('La desviacion estandar es ',des_o32)
                elif 'so2' in ask:
                    des_so2 = np.std(so2_during2019)
                    print('La desviacion estandar es ',des_so2)
                elif 'co' in ask:
                    des_co2 = np.std(co_during2019)
                    print('La desviacion estandar es ',des_co2)
                else:
                    print("Sorry, we dont have information for that, please try again.")
        elif 'exit' in pregunta:
            print('\n******************************************************************************\n')
            exit()

for inputq in inputq:
    if "N" in inputq:
        print('\n******************************************************************************\n')
        continue

    else:
        print('\n******************************************************************************\n')
        exit()