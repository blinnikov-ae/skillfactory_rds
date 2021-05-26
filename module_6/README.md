![Title PNG "AUTO.RU"](https://github.com/blinnikov-ae/skillfactory_rds/blob/master/module_6/carprice_logo.png)
# Проект №6. Car Price.
* [Задача](#задача)
* [Данные](#данные)
* [Папки](#папки)
* [Ноутбуки](#ноутбуки)
* [Коментарии](#коментарии)


[Актуальный Leaderboard](https://www.kaggle.com/c/sf-dst-car-price-prediction/leaderboard)

Команда 'BALU':  

[Alexey Blinnikov](https://www.kaggle.com/alexeyblinnikov)  

[Lyubov Utkina](https://www.kaggle.com/lemura)


## Задача

Прогнозирование стоимости автомобиля по характеристикам.
> https://www.kaggle.com/c/sf-dst-car-price-prediction/overview

## Данные
### Входные данные
- [test.csv](https://www.kaggle.com/c/sf-dst-car-price-prediction/data?select=test.csv) - Тестовая выборка данных для предсказаний на Kaggle
- [all_auto_ru_09_09_2020.csv](thttps://www.kaggle.com/sokolovaleks/parsing-all-moscow-auto-ru-09-09-2020?select=all_auto_ru_09_09_2020.csv) - Старый парсинг данных, предоставленный с baseline проекта.
- [extended_train.csv](https://www.kaggle.com/alexeyblinnikov/car-price-spring-2021?select=extended_train.csv) - Результат собственноручного парсинга
- [sample_submission.csv](https://www.kaggle.com/c/sf-dst-car-price-prediction/data?select=sample_submission.csv) - Файл содержит формат ответа

## Папки
- [Parsing](Parsing) - Папка сожержит ноутбук парсинга данных и входной файл в ноутбук

## Ноутбуки
- [Car_Price_BALU.ipynb](Car_Price_BALU.ipynb) - Основной Kaggle Notebook, в котором производилась работа.
- [CarPrice_parsing_april_2021.ipynb](Parsing/CarPrice_parsing_april_2021.ipynb) - Ноутбук с парсингом данных.
- [Model_testing.ipynb](Model_testing.ipynb) - Ноутбук с поиском моделей по Lazy Predict и продбор оптимальных гиперпараметров моделей.

## Коментарии
На GitHub не загружались входные и выходные данные в силу их веса. Основной ноутбук довольно грамоздкий.  
Итоговый результат по проекту невысокий. Это, частично, может быть объяснено тем, что с каждым днем парсинг акктуальных данных, близких к тестовой выборке, становится всё сложнее.  
Работая в команде, старались параллельно учавствовать во всех этапах разработки.