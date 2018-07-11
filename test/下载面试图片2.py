import requests
import urllib
import os, re
from os.path import join
import time


def getPages(keyword, pages=5):
    params = []
    for i in range(30, 30 * pages + 30, 30):
        # 通过网上资料，可以使用 requests.get() 解析 json 数据，能够得到对应 url
        # 其中一个坑是，原来并不是以下的每个数据都是需要的，某些不要也可以！
        params.append({
            'tn': 'resultjson_com',
            'ipn': 'rj',
            'ct': 201326592,
            'is': '',
            'fp': 'result',
            'queryWord': keyword,
            'cl': 2,
            'lm': -1,
            'ie': 'utf-8',
            'oe': 'utf-8',
            'adpicid': '',
            'st': -1,
            'z': '',
            'ic': 0,
            'word': keyword,
            's': '',
            'se': '',
            'tab': '',
            'width': '',
            'height': '',
            'face': 0,
            'istype': 2,
            'qc': '',
            'nc': 1,
            'fr': '',
            'pn': i,
            'rn': 30,
            # 'gsm': '1e',
            # '1488942260214': ''
        })
    url = 'https://image.baidu.com/search/acjson'
    urls = []
    for param in params:
        # url 与 param 合成完整 url
        urls.append(requests.get(url, param, headers=headers, timeout=3).url)  #
    # print (urls)
    return urls


def get_Img_url(keyword, pages=5):
    # 每页的 URL 集合
    pageUrls = getPages(keyword, pages)
    # 图片url : "thumbURL":"https://ss0.bdstatic.com/70cFuHSh_Q1YnxGkpoWK1HF6hhy/it/u=1789805203,3542215675&fm=27&gp=0.jpg"
    # 正则写的很差！
    exp = re.compile(r'"thumbURL":"[\:\,\_\w\d\=\.\+\s\/\%\$\&]+\.jpg')
    imgUrls = []
    for url in pageUrls:
        # 逐个读取每页 URL
        try:
            with urllib.request.urlopen(url, timeout=3) as pageUrl:
                imgUrl = pageUrl.read().decode('utf-8')
                urls = re.findall(exp, imgUrl)
                for url in urls:
                    # 除去 thumbURL":"
                    imgUrls.append(url.replace('"thumbURL":"', ''))
        # 正则提取 ImgUrl
        except:
            print('SomePage is not opened!')
            continue
    # 所有照片的 urls
    return imgUrls


def getImg(urlList, localPath):
    if not os.path.exists(localPath):
        os.makedirs(localPath)
    x = 1
    for url in urlList:
        # 将 for 循环写在 try 外面
        try:
            # 什么时候应该转义？这点还没弄明白
            # 没有打开特定文件夹！
            with open(keyword + str(x) + '.jpg', 'wb') as f:  # 原本写 ‘\.jpg’ 会出错，打印 \\.jpg
                img = urllib.request.urlopen(url, timeout=3).read()
                f.write(img)
            print('%d.jpg is downloaded!' % x)
            x += 1
        except Exception:
            print("\n  Failed downloading NO. %d image\n" % x)


if __name__ == '__main__':
    keyword = '美女'
    pages = 500
    localPath = 'D:/train_img/'
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
    urlList = get_Img_url(keyword, pages)
    getImg(urlList, localPath)
