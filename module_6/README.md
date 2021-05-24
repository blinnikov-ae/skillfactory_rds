![Title PNG "AUTO.RU"](https://github.com/blinnikov-ae/skillfactory_rds/blob/master/module_6/carprice_logo.png)
# Проект №3. Credit Scoring.
* [Задача](#задача)
* [Данные](#данные)
* [Ноутбуки](#ноутбуки)
* [Коментарии](#коментарии)


[Актуальный Leaderboard](https://www.kaggle.com/c/sf-dst-car-price-prediction/leaderboard)
Команда:
[Alexey Blinnikov](https://www.kaggle.com/alexeyblinnikov)
[Lyubov Utkina](https://www.kaggle.com/lemura)


## Задача

Предсказание риска дефолта заемщиков банка.
> https://www.kaggle.com/c/sf-dst-restaurant-rating

## Данные
### Входные данные
- [train.csv](train.csv) - Тренировочная выборка данных
- [test.csv](test.csv) - Тестовая выборка данных для предсказаний на Kaggle
- [sample_submission.csv](sample_submission.csv) - Файл содержит формат ответа
- [submission.csv](submission.csv) - Выходной файл ноутбука с предсказаниями
- [requirements.txt](requirements.txt) - Минимальные требования к версиям библиотек
#### Описание полей основного датасета
- client_id - идентификатор клиента
- education - уровень образования
- sex - пол заемщика
- age - возраст заемщика
- car - флаг наличия автомобиля
- car_type - флаг автомобиля иномарки
- decline_app_cnt - количество отказанных прошлых заявок
- good_work - флаг наличия “хорошей” работы
- bki_request_cnt - количество запросов в БКИ
- home_address - категоризатор домашнего адреса
- work_address - категоризатор рабочего адреса
- income - доход заемщика
- foreign_passport - наличие загранпаспорта
- sna - связь заемщика с клиентами банка
- first_time - давность наличия информации о заемщике
- score_bki - скоринговый балл по данным из БКИ
- region_rating - рейтинг региона
- app_date - дата подачи заявки
- default - флаг дефолта по кредиту

## Ноутбуки
- [Credit-Scoring-Blinnikov-AE.ipynb](Credit-Scoring-Blinnikov-AE.ipynb) - Основной Kaggle Notebook, в котором производилась работа.
- [Credit-Scoring-Blinnikov-AE(Jupyter_version).ipynb](Credit-Scoring-Blinnikov-AE(Jupyter_version).ipynb) - Версия ноутбука для Jupyter. Без операционной и фаловой ситем Kaggle.

## Коментарии
Ноутбук грамоздкий, в нем присутствуют артефакты, которые при удалении снижают целевую метрику. К примеру, данные обрабатывались исходя из ROC AUC на фиксированной выборке. Вариант с кросс-валидацией снизил итоговый результат и был отвергнут. 

В целом, поскольку разница в метрике очень мала, в верхней части лидерборда небольшое изменение параметров модели влияет на итоговый результат. По крайней мере, в ходе работы, были созданы довольно информативные синтетические параметры.