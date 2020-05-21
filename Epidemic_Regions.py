#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Для работы необходимы файлы:
# epidemic_params.py
# epidemic_jup.py
# regions.py
#
import epidemic_jup
import regions
from epidemic_params import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime
import math
#
import sqlalchemy
import dotenv
import os
#
dotenv.load_dotenv('.env')
DB_USER = os.environ.get('DASHBOARD_USER')
DB_PASSWORD = os.environ.get('DASHBOARD_PASSWORD')
DB_NAME = os.environ.get('DASHBOARD_DB')
DB_HOST = os.environ.get('DASHBOARD_HOST')
DB_PORT = os.environ.get('DASHBOARD_PORT')
#
DAYS_TO_SIM = 306
FLAG_PRINT = False # Флаг вывода результатов моделирования на печать
SIM_START = datetime.date(2020, 3, 1)
dates = pd.date_range(date(2020, 1, 1), end = SIM_START + datetime.timedelta(days = DAYS_TO_SIM), freq = 'D')
#
# Выбор региона
region = 'Москва'
region = 'Санкт-Петербург'
# region = 'Пермский край'


# In[2]:


def Res_Reg(Type_Err, DC_L, DC_R, AT_IN):
    # Type_Err - тип расчёта ошибок: 1 - логарифм, 0 - средняя квадратичная
    # DC - даты начального и конечного лагов для расчёта ошибки
    # AT_IN - таблица дат "изломов" и значений отношения R0 на "изломе" к начальному R0
    # JC - номера начального и конечного лагов для расчёта ошибки
    DAYS_REAL = len(Ser_Real)
    JC_L = (DC_L - SIM_START).days
    JC_R = (DC_R - SIM_START).days
    if JC_L < 15: JC_L = 15
    if JC_R <= JC_L: JC_R = DAYS_REAL
    #
    if FLAG_PRINT:
        print(region)
        print('R0 = %.2f; lags = [%.0f; %.0f]' % (R0 , JC_L, JC_R) )
        for i in range(len(AT_IN)): print(str(AT_IN[i][0]) + ' AT = ' + str(AT_IN[i][1]))
    #
    # Задаём функцию зависимости R0 по дням
    ms = []
    j_ST = 1
    # Значения на левом и правом краях текущего интервала
    a_L = abs(AT_IN[0][1])
    a_R = abs(AT_IN[1][1])
    ST = AT_IN[1][0]-AT_IN[0][0]
    DT = 0
    for simdate in dates:
        if simdate <= AT_IN[0][0]: a = AT_IN[0][1]
        elif j_ST < len(AT_IN):
            # j_ST - номер текущего промежутка
            DT += 1
            a = a_L - ((a_L - a_R) * DT / ST.days)
            #
            if simdate >= AT_IN[j_ST][0]:
                # Достигли правого края отрезка - переходим на следующий
                DT = 0
                j_ST += 1
                a_L = abs(AT_IN[j_ST-1][1])
                if j_ST < len(AT_IN): 
                    a_R = abs(AT_IN[j_ST][1])
                    ST = AT_IN[j_ST][0]-AT_IN[j_ST-1][0]
        else: a = abs(AT_IN[-1][1])
        ms.append( (simdate, a) )
    #
    # Рассчитываем динамику эпидемии
    EParams = epidemic_jup.EpidemicParams(mitigation_strategy = ms, overflow_severity = 2.0, simulation_start_date = SIM_START, simulation_days = DAYS_TO_SIM, average_r0 = R0)
    pop = np.multiply(regions.regional_age_distribution[region], regions.regional_population[region])
    try:
        sim = epidemic_jup.simulate_path_with_maxima(
            params = EParams, icu_beds = regions.regional_ventilators[region],
            imported_cases = epidemic_jup.regional_imports[region], initial_population = pop)
    except BaseException as e:
        print(R0)
        raise e
    #
    # Сравниваем результат моделирования с наблюдаемым
    TRes = 0
    Ser_Sim = sim[0].Infectious[:DAYS_REAL]
    #
    # Расчёт ошибки
    if Type_Err == 1:
        sim_1 = np.log(Ser_Sim[JC_L:JC_R].to_numpy().astype(np.float64) + 0.01)
        sim_r = np.log(Ser_Real[JC_L:JC_R].to_numpy().astype(np.float64) + 0.01)
        TRes = math.sqrt(np.sum((sim_1 - sim_r) ** 2) / (JC_R - JC_L))
    else:
        # TRes = sum(((Ser_Sim - Ser_Real) / Ser_Real) ** 2)
        TRes = math.sqrt(sum((Ser_Sim - Ser_Real) ** 2) / (JC_R - JC_L))
    #
    Max_R = max(EParams.r0_trajectory)
    Min_R = min(EParams.r0_trajectory)
    # Выводим (если нужно) результаты сравнения c 16 марта и динамику R0 с 01 марта
    if FLAG_PRINT:
        plt.figure(figsize = (12, 6) )
        plt.scatter(EParams.dates[15:DAYS_REAL], Ser_Sim[15:DAYS_REAL], label = 'Расчёт')
        plt.scatter(EParams.dates[15:DAYS_REAL], Ser_Real[15:DAYS_REAL], label = 'Наблюдение')
        #
        plt.title(f'{region}: число вирулентных (R0: max=%.2f; min=%.2f)' % (Max_R, Min_R), fontsize = 16)
        plt.legend(loc = (0.04, 0.7), fontsize = 14)
        plt.grid(True)
        plt.show()
        #
        plt.figure(figsize = (12, 6) )
        plt.scatter(EParams.dates, sim[0].Infectious)
        plt.title(f'{region}: число вирулентных (R0: max=%.2f; min=%.2f)' % (Max_R, Min_R), fontsize = 16)
        plt.grid(True)
        plt.show()
        #
        plt.figure(figsize = (12, 6) )
        plt.scatter(EParams.dates[15:DAYS_REAL], EParams.r0_trajectory[15:DAYS_REAL])
        plt.title(f'{region}: траектория R0: max=%.2f; min=%.2f' % (Max_R, Min_R), fontsize = 16)
        plt.grid(True)
        plt.show()    
        #
        print(f'{region}: {str(int(sim[0].Dead[-1]))} погибших')
        print(f'Максимальное число вирулентных {sim[4]} на дату: {str(sim[3].date())}')
        print(f'Максимальное число не попавших в реанимацию {sim[2]} на дату: {str(sim[1].date())}')
        # for t in range(DAYS_REAL):
        #    DT = EParams.dates[t].date() - SIM_START
        #    print(str(t) + ' ' + str(DT.days) + ' ' + str(EParams.dates[t].date()) + ' (' + str(EParams.dates[t].isoweekday()) + ') %.3f' % (EParams.r0_trajectory[t]/R0))
    return (TRes, EParams.dates, sim[0].Infectious)


# In[3]:


# Чтение данных DASHBOARD
db_conn = sqlalchemy.create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
# case_type case_event case_value
# инфицировано; выздоровело; умерло
all_new = pd.read_sql(""" 
    SELECT  date(date_trunc('day', tcr.actual_at)) as "date", sum(tcr.case_value) as "case_value", 
            tcr.case_event as "case_event", tmr.region_alias as "region"
    FROM tab_case_records tcr, tab_master_regions tmr
    WHERE tmr.region_code = tcr.region_code
    GROUP by "region", "date", "case_event"
    ORDER by "region", "date", "case_event"
    """, con = db_conn)
#
# for reg_T in all_new['region'].unique(): print(reg_T)
#
print(region)
df_case = all_new[(all_new['case_event'] == 'инфицировано') & (all_new['region'] == region)].set_index('date')['case_value']
df_rec = all_new[(all_new['case_event'] == 'выздоровело') & (all_new['region'] == region)].set_index('date')['case_value']
df_dead = all_new[(all_new['case_event'] == 'умерло') & (all_new['region'] == region)].set_index('date')['case_value']
#
df_case_F = df_case.reindex(pd.date_range(SIM_START, df_case.index.max()))
df_case_F.sort_index(inplace = True)
df_case_F.fillna(0, inplace = True)
#
df_rec_F = df_rec.reindex(pd.date_range(SIM_START, df_rec.index.max()))
df_rec_F.sort_index(inplace = True)
df_rec_F.fillna(0, inplace = True)
#
df_dead_F = df_dead.reindex(pd.date_range(SIM_START, df_dead.index.max()))
df_dead_F.sort_index(inplace = True)
df_dead_F.fillna(0, inplace = True)
#
Ser_Real = df_case_F.cumsum() - df_rec_F.cumsum() - df_dead_F.cumsum()
#
pd.set_option('display.max_rows', None)
# pd.DataFrame(Ser_Real)


# In[4]:


if region == 'Москва':
    # Москва
    R0 = 6.3
    ATM = []
    ATM.append((date(2020, 3, 15), 1))
    ATM.append((date(2020, 3, 22), 0.889))
    ATM.append((date(2020, 3, 29), 0.823))
    ATM.append((date(2020, 4, 5), 0.7))
    ATM.append((date(2020, 4, 12), 0.6))
    ATM.append((date(2020, 4, 19), 0.43))
    ATM.append((date(2020, 4, 26), 0.37))
    ATM.append((date(2020, 5, 1), 0.4))
    ATM.append((date(2020, 5, 3), 0.38))
    #
    ATM.append((date(2020, 5, 10), 0.24))
    ATM.append((date(2020, 5, 15), 0.22))
    ATM.append((date(2020, 5, 19), 0.15))
    #
elif region == 'Санкт-Петербург':
    # Санкт-Петербург
    R0 = 3
    ATM = []
    ATM.append((date(2020, 3, 15), 1))
    ATM.append((date(2020, 5, 1), 0.8))
    ATM.append((date(2020, 5, 10), 0.7))
    ATM.append((date(2020, 5, 15), 0.65))
#
# Ручной запуск без оптимизации по выбранным параметрам
FLAG_PRINT = True
TRes = Res_Reg(1, date(2020, 1, 1), date(2020, 1, 1), ATM)
print('SD: %.0f' % TRes[0])


# In[8]:


# Оптимизационная функция
def Res_Mos_Opt(X):
    AS, AR = X
    AT[IT+1] = (AT[IT+1][0], AS)
    AT[IT+2] = (AT[IT+2][0], AR)
    ResT = Res_Reg(0, DL, DR, AT)
    return ResT[0]


# In[ ]:


# Оптимизация
from scipy.optimize import basinhopping
#
R0 = 6.3
AT = []
AT.append((date(2020, 3, 15), 1))
AT.append((date(2020, 3, 22), 0.9))
AT.append((date(2020, 3, 29), 0.8))
AT.append((date(2020, 4, 5), 0.7))
AT.append((date(2020, 4, 12), 0.6))
AT.append((date(2020, 4, 19), 0.43))
AT.append((date(2020, 4, 26), 0.37))
AT.append((date(2020, 5, 1), 0.4))
AT.append((date(2020, 5, 3), 0.38))
AT.append((date(2020, 5, 10), 0.24))
#
FLAG_PRINT = False
for IT in range(len(AT) - 2):
    DL = AT[IT][0]
    DR = AT[IT+2][0]
    X = [AT[IT+1][1], AT[IT+2][1]]
    Res_Mos_Opt(X)
    print(f'Начальные значения {AT[IT+1][0]}: %.3f;  {DR}: %.3f' % (X[0], X[1]) )
    Opt = basinhopping(Res_Mos_Opt, X)
    print(f'{IT} {AT[IT+1][0]}: A = %.3f;  {AT[IT+2][0]}: A = %.3f; Err = %.3f' % (abs(Opt.x[0]), abs(Opt.x[1]), Opt.fun) )
    print('ERR: %.3f' % Opt.fun)    
    AT[IT+1] = (AT[IT+1][0], round(abs(Opt.x[0]), 3) )
    AT[IT+2] = (AT[IT+2][0], round(abs(Opt.x[1]), 3) )
#
FLAG_PRINT = True
TRes = Res_Mos(0, date(2020, 1, 1), date(2020, 1, 1), AT)
print('ERR: %.3f' % TRes[0])

