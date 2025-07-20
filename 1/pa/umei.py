import requests
import re
import bs4 as bs

url = 'http://umei.cc'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0',
    'cookie': '__51vcke__K0KOUvCHIpTH8Vt6=ee43f8e8-98d7-571e-8109-a6451cc3a89f; __51vuft__K0KOUvCHIpTH8Vt6=1752761425028; __51uvsct__K0KOUvCHIpTH8Vt6=3; gxgefecookieinforecord=%2C61-338399%2C14-332342%2C; __vtins__K0KOUvCHIpTH8Vt6=%7B%22sid%22%3A%20%22c3492287-24a7-5c07-b23d-811532fe730d%22%2C%20%22vd%22%3A%204%2C%20%22stt%22%3A%201236384%2C%20%22dr%22%3A%201083322%2C%20%22expires%22%3A%201753000813380%2C%20%22ct%22%3A%201752999013380%7D'
}


response = requests.get(url, headers=headers)
response.encoding = 'utf-8'
main_page = bs.BeautifulSoup(response.text, 'html.parser')
a_list = main_page.find('div', class_='index-list-c').find_all('a')
div_list = []
for a in a_list:
    href = url + a['href']
    resp = requests.get(href)
    resp.encoding = 'utf-8'
    child_page = bs.BeautifulSoup(resp.text, 'html.parser')
    div = child_page.find('div', class_='big-pic').find('img')
    div_list.append(div['src'])

img_resp = requests.get(div_list[0])
with open('img', 'wb') as f:
    f.write(img_resp.content)

#  BYD 教程里的网址给封了[😀]

