![Title PNG "TripAdvisor"](https://github.com/blinnikov-ae/skillfactory_rds/skillfactory_logo.png)
# Проект №3. Restaurant Rating prediction.
* [Задача](#задача)
* [Данные](#данные)
* [Ноутбуки](#ноутбуки)
* [Коментарии](#коментарии)

Профиль Kaggle:

[Alexey Blinnikov](https://www.kaggle.com/alexeyblinnikov)

[Актуальный Leaderboard](https://www.kaggle.com/c/sf-dst-restaurant-rating/leaderboard)

## Задача

Предсказание рейтинга ресторанов. По легенде, может быть использовано для определния аномально высоких рейтингов, свидетельствующих о накрутке отзывов.
> https://www.kaggle.com/c/sf-dst-restaurant-rating

## Данные
### Входные данные
- [main_task.csv](main_task.csv) - Тренировочная выборка данных
- [kaggle_task.csv](kaggle_task.csv) - Тестовая выборка данных для предсказаний на Kaggle
- [sample_submission.csv](sample_submission.csv) - Файл содержит формат ответа
- [submission.csv](submission.csv) - Выходной файл ноутбука с предсказаниями
- [requirements.txt](requirements.txt) - Минимальные требования к версиям библиотек
- [worldcities.csv](worldcities.csv) - DataSet с Kaggle, содержит информацию о городах мира (https://www.kaggle.com/juanmah/world-cities)
#### Описание полей основного датасета
- Restaurant_id - ID
- City - Город 
- Cuisine Style - Кухня
- Ranking - Ранг ресторана относительно других ресторанов в этом городе
- Price Range - Цены в ресторане в 3 категориях
- Number of Reviews - Количество отзывов
- Reviews - 2 последних отзыва и даты этих отзывов
- URL_TA - Cтраница ресторана на 'www.tripadvisor.com' 
- ID_TA - ID ресторана в TripAdvisor
- Rating - Рейтинг ресторана
#### Вспомогательные файлы
- [model_selection.csv](model_selection.csv) - Обработанная тренировочная выборка, подается на вход в [ноутбук](TripAdvisor_model_selection.ipynb) подбора параметров модели
- [TA_ratings_04_2021.csv](TA_ratings_04_2021.csv) - Резултат [ноутбука](TripAdvisor_actual_rating_april_2021.ipynb), запрашивающего актуальные рейтинги с https://www.tripadvisor.com/
- [TripAdvisor-Rating-Blinnikov-AE.html](TripAdvisor-Rating-Blinnikov-AE.html) - Поскольку ноутбук довольно длинный, HTML-файл позволяет быстрее загрузить и просмотреть ноутбук на GitHub
## Ноутбуки
- [TripAdvisor-Rating-Blinnikov-AE.ipynb](TripAdvisor-Rating-Blinnikov-AE.ipynb) - Основной Kaggle Notebook, в котором производилась работа.
- [TripAdvisor-Rating-Blinnikov-AE(Jupyter_version).ipynb](TripAdvisor-Rating-Blinnikov-AE(Jupyter_version).ipynb) - Версия ноутбука для Jupyter. Без операционной и фаловой ситем Kaggle. Так же отличается синтексис Word2Vec.
### Вспомогательные ноутбуки
- [TripAdvisor_actual_rating_april_2021.ipynb](TripAdvisor_actual_rating_april_2021.ipynb) - Внутри ноутбука выполнялись HTML-запросы. Выполнение заняло около суток.
- [TripAdvisor_model_selection.ipynb](TripAdvisor_model_selection.ipynb) - Ноутбук для подбора параметров моделей.

## Коментарии
Работа с API TripAdvisor и Google, могла бы существенно ускорить процесс и сократить код. На момент последнего сабмита, ноутбук неплохо показывал себя в соревнии.