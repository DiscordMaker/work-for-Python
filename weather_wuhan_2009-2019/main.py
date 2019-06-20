import math
import re

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator, FuncFormatter
from pandas import DataFrame

ISTEST = False

pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)






# hsv转rgb
def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b


# rgb转16#
def color(value):
    digit = list(map(str, range(10))) + list("ABCDEF")
    if isinstance(value, tuple):
        string = '#'
        for i in value:
            a1 = i // 16
            a2 = i % 16
            string += digit[a1] + digit[a2]
        return string


# 月份格式
def month_formatter(x, pos):
    v = int(x / 31) + 1
    if v > 12:
        return '-'
    else:
        return str(v)



stationID='CHM00057494'

df_PRCP = DataFrame(columns=['datetime', 'year', 'month', 'day', 'dayIndex', 'value', 'color'])  # 降雨数据
df_TAVG = DataFrame(columns=['datetime', 'year', 'month', 'day', 'dayIndex', 'value', 'color'])  # 平均气温数据



# 导入数据
with open('data/' + stationID + '.txt', 'r') as f1:
    dataList = f1.readlines()

pdi = 0  # 降雨天数索引
tdi = 0  # 气温天数索引

_d: str
for _d in dataList:

    _res = re.split("\s+|s|S", _d)  # 使用正则对字符串进行分割
    _date_res = list(filter(lambda x: x != '', _res))  # 清除结果集中的空字符串

    _title = _date_res[0]  # 结果数组中的第一位为索引
    _year = _title.replace(stationID, '')[0:4]  # 获取年
    _month = _title.replace(stationID, '')[4:6]  # 获取月
    _type = _title.replace(stationID, '')[6:]  # 获取数据类型（PRCP，TAVG)
    print('loding ' + str(_year) + str(_month))

    # 结果集中的后续内容为日数据值
    for i in range(1, len(_date_res)):
        _datetime = str(_year) + '-' + str(_month) + '-' + str(i)

        # 降雨记录
        if _type == 'PRCP':

            if _month == '01' and i == 1:
                pdi = 1
            else:
                pdi += 1

            _value = float(_date_res[i]) / 10  # 降雨量为 value / 10


            _color = '#FFFFFF'
            if _value > 0:
                _color = color(tuple(hsv2rgb(int(_value * 0.75) + 180, 1, 1)))  # 设置数据颜色值
            elif _value != -999.9:
                _color = '#FFFFFF'
            _df = DataFrame({'datetime': [_datetime], 'year': [_year], 'month': [_month], 'day': [i], 'dayIndex': [pdi],
                             'value': [_value], 'color': [_color]})
            df_PRCP = df_PRCP.append(_df, ignore_index=True)  # 将单行数据添加至总DF中

        # 平均气温记录
        elif _type == 'TAVG':

            if _month == '01' and i == 1:
                tdi = 1
            else:
                tdi += 1

            _value = str(_date_res[i]).find('T') == True and -9999 or float(_date_res[i]) / 10  # 温度为 value / 10

            _color = '#FFFFFF'
            if _value != -999.9:
                _color = color(tuple(hsv2rgb(int(-7 * _value) + 240, 1, 1)))  # 设置数据颜色值
            _df = DataFrame({'datetime': [_datetime], 'year': [_year], 'month': [_month], 'day': [i], 'dayIndex': [tdi],
                             'value': [_value], 'color': [_color]})
            df_TAVG = df_TAVG.append(_df, ignore_index=True)

    # 调试使用，避免数据加载时间过长
    if len(df_PRCP) and len(df_TAVG) > 100000:
        break

df_PRCP = df_PRCP.set_index('datetime')
df_TAVG = df_TAVG.set_index('datetime')



# 画布基础设置
plt.rcParams['figure.figsize'] = (10.0, 6.0)  # 设置figure_size尺寸

plt.rcParams['image.cmap'] = 'gray'  # 设置 颜色 style

plt.rcParams['figure.dpi'] = 200  # 分辨率
plt.rcParams['savefig.dpi'] = 200  # 图片像素
plt.rcParams['image.interpolation'] = 'nearest'  # 设置 interpolation style

key = '1'

while(key!='0'):
    key = input('''请输入0-5
1:2009-2019.6 TAVG散点图带颜色
2:2009-2019.6 PRCP
3.TAVG补温度显示
4.2019年1-6月TAVG进行线性回归分析
5.2019年1-6月TAVG进行回归分析（sklearn）
0.退出
''')
    if (key == '1'):
        # TAVG
        plt.gca().xaxis.set_major_locator(MultipleLocator(31))
        plt.gca().xaxis.set_major_formatter(FuncFormatter(month_formatter))
        plt.gca().grid()  # 网格
        # plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.scatter(df_TAVG['dayIndex'], df_TAVG['value'], s=80, c=df_TAVG['color'], marker='.', alpha=0.05)
        plt.xlim(0, 365)
        plt.ylim(-30, 50)
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(10))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, y: str(x) + 'C°'))

        plt.title('TAVG : 2009-2019.6')
        plt.xlabel("Month")
        plt.ylabel("Temperature C°")

        if ISTEST == False:
            plt.savefig("./data/TAVG.png")  # 保存图片

        plt.show()
    elif (key == '2'):
        # PRCP散点图
        plt.gca().xaxis.set_major_locator(MultipleLocator(31))  # 设置x轴的刻度间隔
        plt.gca().xaxis.set_major_formatter(FuncFormatter(month_formatter))  # 设置x轴的显示数值
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(10))  # 设置y轴的刻度间隔
        plt.scatter(df_PRCP['dayIndex'], df_PRCP['value'], s=40, c=df_PRCP['color'], marker='.', alpha=0.5)  # 散点图绘制
        plt.xlim(0, 365)  # 设置x轴的坐标范围
        plt.ylim(1, 250)  # 设置y轴的坐标范围

        plt.title('PRCP : 2009-2019.6')
        plt.xlabel("Month")
        plt.ylabel("Precipitation mm")

        if ISTEST == False:
            plt.savefig("./data/PRCP.png")  # 保存图片

        plt.show()
    elif (key == '3'):
        # TAVG补温度显示
        plt.scatter(df_TAVG['dayIndex'], df_TAVG['year'], s=20, c=df_TAVG['color'], marker='|', alpha=1)  # 散点图绘制
        plt.xlabel("Month")
        plt.ylabel("Year")
        plt.title("Historical TAVG Show : 2009-2019.6")

        # 设置主刻度
        plt.xlim(0, 365)
        plt.gca().xaxis.set_major_locator(MultipleLocator(31))
        plt.gca().xaxis.set_major_formatter(FuncFormatter(month_formatter))
        plt.gca().yaxis.set_major_locator(MultipleLocator(5))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, y: int(x + 2009)))
        # 设置次刻度
        plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(1))

        # # 打开网格
        plt.gca().xaxis.grid(True, which='major')  # x坐标轴的网格使用主刻度
        plt.gca().yaxis.grid(True, which='minor')  # y坐标轴的网格使用次刻度
        if ISTEST == False:
            plt.savefig("./data/Historical TAVG Show.png")  # 保存图片
        plt.show()
    elif (key == '4'):
        print(df_TAVG)
        # 线性回归分析
        print(DataFrame)
        data = df_TAVG[-558:-186]
        print(df_TAVG[-558:-186]["value"])
        print(data)
        plt.xlim(0, 180)  # 设置x轴的坐标范围
        plt.ylim(-20, 50)  # 设置y轴的坐标范围
        x = df_TAVG['dayIndex']
        y = df_TAVG['value']
        # 截取2019年数据
        x_data = x[-181:-122].append(x[-119:-22])
        y_data = y[-181:-122].append(y[-119:-22])
        plt.scatter(x_data, y_data, s=16)
        plt.gca().xaxis.set_major_locator(MultipleLocator(31))
        plt.gca().xaxis.set_major_formatter(FuncFormatter(month_formatter))
        plt.gca().yaxis.set_major_locator(MultipleLocator(5))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
        plt.xlabel("Day")
        plt.ylabel("TAVG")
        plt.title("2019.1-2019.6 TAVG")
        plt.show()
        # 学习率
        lr = 0.0001
        # 截距
        b = 0
        # 斜率
        k = 0
        # 最大迭代次数
        epochs = 100


        # 梯度下降法

        def compute_error(b, k, x_data, y_data):
            totalError = 0
            for i in range(0, len(x_data)):
                totalError += (y[i] - (k * x_data[i] + b)) ** 2
            return totalError / float(len(x_data)) / 2.0


        def gradient_descent_runner(x_data, y_data, b, k, lr, epochs):
            # 计算总数据量
            m = float(len(x_data))
            # 循环epochs次
            for i in range(epochs):
                b_grad = 0
                k_grad = 0
                # 计算梯度的总和再求平均
                for j in range(0, len(x_data)):
                    b_grad += (1 / m) * (((k * x_data[j]) + b) - y_data[j])
                    k_grad += (1 / m) * x_data[j] * (((k * x_data[j]) + b) - y_data[j])
                    # 更新b和k
                    b = b - (lr * b_grad)
                    k = k - (lr * k_grad)
                    # # 每迭代5次，输出一次图像
                    # if (i%5==0):
                    #     print("epochs：",i)
                    #     plt.plot(x,y,'b.')
                    #     plt.plot(x,k*x+b,'r')
                    #     plt.show()
            return b, k


        # print(x_data, y_data)

        print("Strating b={0},k={1},error={2}".format(b, k, compute_error(b, k, x_data, y_data)))
        print("Running....")
        b, k = gradient_descent_runner(x_data, y_data, b, k, lr, epochs)
        print("after {0} iterations b={1},k={2},error={3}".format(epochs, b, k, compute_error(b, k, x_data, y_data)))
        plt.xlim(0, 180)  # 设置x轴的坐标范围
        plt.ylim(-20, 50)  # 设置y轴的坐标范围
        plt.gca().xaxis.set_major_locator(MultipleLocator(31))
        plt.gca().xaxis.set_major_formatter(FuncFormatter(month_formatter))
        plt.gca().yaxis.set_major_locator(MultipleLocator(5))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
        plt.plot(x_data, y_data, 'b.', )
        plt.plot(x_data, k * x_data + b, 'r')
        plt.xlabel("Day")
        plt.ylabel("TAVG")
        plt.title("2019.1-2019.6 TAVG LinearRegression")
        plt.show()
    elif (key == '5'):
        #sklearn回归分析

        plt.xlim(0, 180)  # 设置x轴的坐标范围
        plt.ylim(-20, 50)  # 设置y轴的坐标范围
        x = df_TAVG['dayIndex']
        y = df_TAVG['value']
        #截取2019年数据
        x_data = x[-181:-122].append(x[-119:-22])
        y_data = y[-181:-122].append(y[-119:-22])

        #画出散点图
        plt.scatter(x_data, y_data, s=16)
        plt.gca().xaxis.set_major_locator(MultipleLocator(31))
        plt.gca().xaxis.set_major_formatter(FuncFormatter(month_formatter))
        plt.gca().yaxis.set_major_locator(MultipleLocator(5))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
        plt.xlabel("Day")
        plt.ylabel("TAVG")
        plt.title("2019.1-2019.6 TAVG")
        plt.show()

        # 数据处理符合输入
        lengh = len(x_data)
        # x_data = x_data[:, np.newaxis]
        x_data=np.array(x_data).reshape([lengh, 1])
        # y_data = y_data[:, np.newaxis]
        y_data = np.array(y_data)
        print(x_data)
        print(y_data)

        # minX = min(x_data)
        # maxX = max(x_data)
        # X = np.arange(minX, maxX).reshape([-1, 1])

        #sklearn线性回归
        model=LinearRegression()
        model.fit(x_data,y_data)
        plt.xlim(0, 180)  # 设置x轴的坐标范围
        plt.ylim(-20, 50)  # 设置y轴的坐标范围
        plt.gca().xaxis.set_major_locator(MultipleLocator(31))
        plt.gca().xaxis.set_major_formatter(FuncFormatter(month_formatter))
        plt.gca().yaxis.set_major_locator(MultipleLocator(5))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
        plt.scatter(x_data,y_data,c='b')
        plt.plot(x_data,model.predict(x_data),'r')
        plt.xlabel("Day")
        plt.ylabel("TAVG")
        plt.title("2019.1-2019.6 TAVG LinearRegression")
        plt.show()

        #多项式回归分析



        #定义多项巨回归式子，degree值用来调整多项式的特征
        poly_reg =PolynomialFeatures(degree=3)
        #特证处理
        x_poly=poly_reg.fit_transform((x_data))
        #定义回归模型
        lin_reg=LinearRegression()
        #训练模型
        lin_reg.fit(x_poly,y_data)

        print(x_poly)

        plt.xlim(0, 180)  # 设置x轴的坐标范围
        plt.ylim(-20, 50)  # 设置y轴的坐标范围
        plt.scatter(x_data,y_data,c='b')
        plt.plot(x_data,lin_reg.predict(poly_reg.fit_transform(x_data)),'r')
        plt.gca().xaxis.set_major_locator(MultipleLocator(31))
        plt.gca().xaxis.set_major_formatter(FuncFormatter(month_formatter))
        plt.gca().yaxis.set_major_locator(MultipleLocator(5))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
        plt.xlabel("Day")
        plt.ylabel("TAVG")
        plt.title("2019.1-2019.6 Polynomial Regression")
        plt.show()
        print("Cofficients:",lin_reg.coef_)
        print("intercept",lin_reg.intercept_)
    elif (key=='0'):
        break
    else:
        print("错误的输入")

print("end")













