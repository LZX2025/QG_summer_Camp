import requests
import re


domain = 'https://www.dytt8899.com'

resp = requests.get(domain)
resp.encoding = 'gb2312'
#print(resp.text)

obj1 = re.compile(r'2025必看热片.*?<ul>(?P<ul>.*?</ul>)', re.S)
obj2 = re.compile(r"a href='(?P<href>.*?)' title=(?P<name>.*?)>", re.S)
obj3 = re.compile(r'<td style="WORD-WRAP: break-word" bgcolor="#fdfddf"><a href="(?P<download>.*?)"', re.S)

dic = {}
href_list = []
name_list = []
it1 = obj1.finditer(resp.text)
for it in it1:
    ul = it.group('ul')
    print(ul)
    result = obj2.finditer(ul)
    for r in result:
        name = r.group('name')
        name_list.append(name)
        href = domain + r.group('href')
        href_list.append(href)

dic = {}
for href, name in zip(href_list, name_list):
    print(href, name)
    child_resp = requests.get(href)
    child_resp.encoding = 'gb2312'
    result3 = obj3.search(child_resp.text)
    dic[name] = result3.group('download')

print(dic)


