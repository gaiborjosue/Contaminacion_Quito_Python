import requests
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

#!!!!!!!
city = 'quito'
#!!!!!!!
url = 'https://api.waqi.info/feed/' + city + '/?token=959cd82964b69946f7b1aad10e6f56be40dd549e'

r = requests.get(url)

data = r.json()['data']


aqi = data['aqi']
iaqi = data['iaqi']

del iaqi['p']


# for i in iaqi.items():
#     print(i[0],':',i[1]['v'])

dew = iaqi.get('dew', 'Nil')
so2 = iaqi.get('so2', 'Nil')
o3 = iaqi.get('o3', 'Nil')
co = iaqi.get('co', 'Nil')
h = iaqi.get('h', 'Nil')
pm25 = iaqi.get('pm25', 'Nil')


# print(f'{city} AQI :',aqi,'\n')
# print('Individual Air quality')
# print('Dew :',dew)
# print('co :',co)
# print('Ozone :',o3)
# print('sulphur :',so2)
# print('Hidrogen :',h)
# print('pm25 :',pm25)

pollutants = [i for i in iaqi]
values = [i['v'] for i in iaqi.values()]

# print(pollutants, values)

#plot a pie chart
explode = [0 for i in pollutants]
mx = values.index(max(values))
explode[mx] = 0.1


plt.figure(figsize=(8,6))

plt.pie(values,labels=pollutants,autopct='%1.1f%%', shadow=True, explode=explode)
plt.show()

#geographic location in map
geo = data['city']['geo']

fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.stock_img()

plt.scatter(geo[1],geo[0], color='blue')
plt.text(geo[1] + 3,geo[0]-2,f'{city} AQI \n    {aqi}',color='red')

plt.show()