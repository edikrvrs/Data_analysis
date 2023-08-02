import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind,f_oneway

# Загрузка данных
data = pd.read_csv('Forbes.csv')

# Преобразование столбца 'Previous Year Rank' в числовой формат
data['Previous Year Rank'] = pd.to_numeric(data['Previous Year Rank'], errors='coerce')

# 1. Количество строк и столбцов
num_rows, num_cols = data.shape
print(f"Количество строк: {num_rows}")
print(f"Количество столбцов: {num_cols}")

# 2. a) EDA числовых переменных
numeric_cols = ['Current Rank', 'Previous Year Rank', 'Year', 'earnings ($ million)']
numeric_data = data[numeric_cols]

for col in numeric_cols:
    # Доля пропусков
    missing_ratio = numeric_data[col].isnull().mean()
    print(f"Доля пропусков в {col}: {missing_ratio:.2%}")

    # Максимальное и минимальное значения
    max_value = numeric_data[col].max()
    min_value = numeric_data[col].min()
    print(f"Максимальное значение в {col}: {max_value}")
    print(f"Минимальное значение в {col}: {min_value}")

    # Среднее значение
    mean_value = numeric_data[col].mean()
    print(f"Среднее значение в {col}: {mean_value:.2f}")

    # Медиана
    median_value = numeric_data[col].median()
    print(f"Медиана в {col}: {median_value}")

    # Дисперсия
    variance_value = numeric_data[col].var()
    print(f"Дисперсия в {col}: {variance_value:.2f}")

    # Квантили 0.1 и 0.9
    quantile_01 = numeric_data[col].quantile(0.1)
    quantile_09 = numeric_data[col].quantile(0.9)
    print(f"Квантиль 0.1 в {col}: {quantile_01}")
    print(f"Квантиль 0.9 в {col}: {quantile_09}")

    # Квартили 1 и 3
    quartile_1 = numeric_data[col].quantile(0.25)
    quartile_3 = numeric_data[col].quantile(0.75)
    print(f"Квартиль 1 в {col}: {quartile_1}")
    print(f"Квартиль 3 в {col}: {quartile_3}")

    print('\n')

# b). EDA категориальных переменных
categorical_cols = ['Name', 'Nationality', 'Sport']
categorical_data = data[categorical_cols]

for col in categorical_cols:
    # Доля пропусков
    missing_ratio = categorical_data[col].isnull().mean()
    print(f"Доля пропусков в {col}: {missing_ratio:.2%}")

    # Количество уникальных значений
    unique_values = categorical_data[col].nunique()
    print(f"Количество уникальных значений в {col}: {unique_values}")

    # Мода
    mode_value = categorical_data[col].mode().values[0]
    print(f"Мода в {col}: {mode_value}")

    print('\n')



# 3.
# Гипотеза 1: Заработки спортсменов в сфере бокса выше, чем в других видах спорта
boxing_earnings = data[data['Sport'] == 'boxing']['earnings ($ million)']
other_sports_earnings = data[data['Sport'] != 'boxing']['earnings ($ million)']
t_statistic, p_value = ttest_ind(boxing_earnings, other_sports_earnings)
print('Гипотеза 1: Заработки спортсменов в сфере бокса выше, чем в других видах спорта')
if p_value < 0.05:
    print('Статистически значимое различие')
else:
    print('Статистически значимого различия не обнаружено')
print('t-статистика:', t_statistic)
print('p-значение:', p_value)

# Гипотеза 2: Заработки спортсменов из разных стран различаются
country_earnings = []
for country in data['Nationality'].unique():
    country_earnings.append(data[data['Nationality'] == country]['earnings ($ million)'])
f_statistic, p_value = f_oneway(*country_earnings)
print('Гипотеза 2: Заработки спортсменов из разных стран различаются')
if p_value < 0.05:
    print('Статистически значимое различие')
else:
    print('Статистически значимого различия не обнаружено')
print('F-статистика:', f_statistic)
print('p-значение:', p_value)

# 4. Построение информативных графиков
# Пример построения графика, где по оси x - год, а по оси y - заработок
plt.plot(data['Year'], data['earnings ($ million)'])
plt.xlabel('Year')
plt.ylabel('Earnings ($ million)')
plt.title('Earnings Over Time')
plt.show()

# Построение графика, показывающего изменение заработков по годам для разных спортсменов
plt.figure(figsize=(12, 6))
for name, group in data.groupby('Name'):
    plt.plot(group['Year'], group['earnings ($ million)'], label=name)
plt.xlabel('Year')
plt.ylabel('Earnings ($ million)')
plt.title('Earnings Over Time by Athlete')
plt.subplots_adjust(left=0.05,right=0.7, wspace=0.2)
plt.legend(bbox_to_anchor=(1.05, 1.1), loc='upper left', ncol=2)
plt.subplots_adjust(left=0.1, right=0.5)
plt.show()







