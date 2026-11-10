# Лабораторная работа 4. MLOps Вариант 18

## 1. Цель работы и постановка задачи с указанием индивидуального варианта
**Цель:** Освоить основы MLOps: учёт и сравнение экспериментов в MLflow, регистрацию и управление версиями моделей, развёртывание сервиса предсказаний с использованием Docker и Docker Compose, проверку его работоспособности, обновление и откат модели.

**Индивидуальный вариант:** Вариант 18. Учебная классификация типа опухоли.
**Данные:** Встроенный набор sklearn.datasets.load_breast_cancer.
**Задача:** Измерения -> метка diagnosis. Бинарная классификация (malignant/benign).
**Алгоритмы и критерий:** LogisticRegression и RandomForest; основная метрика — val_f1_macro.

## 2. Описание набора данных
*   **Источник:** UCI ML Repository (Breast Cancer Wisconsin Diagnostic), доступен через scikit-learn.
*   **Признаки:** 30 числовых признаков (радиус, текстура, периметр, площадь, гладкость и т.д.).
*   **Целевая переменная:** target (0 — злокачественная malignant, 1 — доброкачественная benign).
*   **Подготовка и разбиение:** Данные разделены на 3 части (60/20/20) с использованием train_test_split и stratify. 20% отложено под финальный test, из оставшихся 80% выделено 25% под validation. Масштабирование StandardScaler обучается только на train внутри Pipeline.

## 3. Описание программного окружения и структуры проекта
*   **ОС:** Ubuntu 24.04.4 LTS
*   **Python:** 3.12.3
*   **Основные библиотеки:** MLflow 3.5.0, scikit-learn 1.7.2, FastAPI, uvicorn, pandas, numpy.
*   **Docker:** Docker Engine 29.1.3, Docker Compose v2.

![Проверка системы](screenshots/Screenshot_1.png)
![Установка Docker Compose](screenshots/Screenshot_2.png)
![Установка зависимостей](screenshots/Screenshot_3.png)
![Структура проекта в VS Code](screenshots/Screenshot_4.png)

## 4. Последовательность выполнения работы с командами и скриншотами

**4.1. Запуск MLflow**
Сервер запущен локально с использованием SQLite и локальной папки артефактов:

    mlflow server \
      --backend-store-uri sqlite:////home/vlad/mlops_lab/mlflow-data/mlflow.db \
      --artifacts-destination ./mlflow-data/artifacts \
      --serve-artifacts \
      --host 127.0.0.1 \
      --port 5000

![Запуск сервера MLflow](screenshots/Screenshot_5.png)
![Логи инициализации базы MLflow](screenshots/Screenshot_6.png)

**4.2. Проведение экспериментов**
Обучение запускалось через train.py с разными гиперпараметрами (6 запусков):

    python src/train.py --algorithm lr --c 0.1
    python src/train.py --algorithm lr --c 1
    python src/train.py --algorithm lr --c 10
    python src/train.py --algorithm rf --trees 50 --depth 3
    python src/train.py --algorithm rf --trees 100 --depth 5
    python src/train.py --algorithm rf --trees 200 --depth 8

![Сравнение экспериментов в MLflow](screenshots/Screenshot_7.png)

**4.3. Сборка Docker-контейнера v1 и проверка API**
Собран образ и запущен контейнер:

    sudo docker build -t breast-api:v1 .
    sudo docker run -d --name breast-api -p 127.0.0.1:8000:8000 breast-api:v1
    sudo docker ps
    sudo docker logs breast-api

![Сборка Docker-образа v1](screenshots/Screenshot_8.png)
![Запуск контейнера и проверка запросов](screenshots/Screenshot_11.png)

Проверка некорректных запросов (возвращают 422 Unprocessable Entity):
![Ответы 422 на некорректные запросы](screenshots/Screenshot_9.png)

Проверка корректного запроса через Swagger UI:
![Проверка API через Swagger UI](screenshots/Screenshot_12.png)

Смоук-тест check_api.py (подтверждает совпадение контейнера и модели):
    python src/check_api.py

![Успешный смоук-тест v1](screenshots/Screenshot_10.png)

**4.4. Обновление и откат**
Обновление на версию 2 и откат на версию 1 (без переобучения):

    # Обновление до v2
    python src/promote.py --version 2
    python src/export_model.py
    sudo docker build -t breast-api:v2 .
    sudo docker stop breast-api && sudo docker rm breast-api
    sudo docker run -d --name breast-api -p 127.0.0.1:8000:8000 breast-api:v2

    # Откат на v1
    sudo docker stop breast-api && sudo docker rm breast-api
    sudo docker run -d --name breast-api -p 127.0.0.1:8000:8000 breast-api:v1
    python src/promote.py --version 1
    python src/export_model.py

![Назначение alias champion на v2 и экспорт](screenshots/Screenshot_13.png)
![Сборка и проверка контейнера v2](screenshots/Screenshot_14.png)
![Откат к версии 1](screenshots/Screenshot_15.png)

**4.5. Запуск через Docker Compose**
    sudo docker compose up -d --build
    sudo docker compose ps
    sudo docker compose logs --tail 20

![Запуск через Docker Compose и healthcheck](screenshots/Screenshot_16.png)

## 5. Сравнительная таблица не менее шести экспериментов
| № | Алгоритм | Гиперпараметры | val_accuracy | val_f1_macro |
| :--- | :--- | :--- | :--- | :--- |
| 1 | LogisticRegression | C=0.1 | ~0.9737 | ~0.9715 |
| 2 | LogisticRegression | C=1 | **~0.9912** | **~0.9906** |
| 3 | LogisticRegression | C=10 | ~0.9649 | ~0.9626 |
| 4 | RandomForest | trees=50, depth=3 | ~0.9561 | ~0.9531 |
| 5 | RandomForest | trees=100, depth=5 | ~0.9474 | ~0.9439 |
| 6 | RandomForest | trees=200, depth=8 | ~0.9561 | ~0.9531 |

## 6. Обоснование выбора модели и результаты итоговой оценки
По результатам валидации победителем признана **LogisticRegression с C=1**, показавшая наивысший val_f1_macro (≈0.9906). Итоговая оценка на отложенном тесте (test set):
*   test_accuracy: **0.9824**
*   test_f1_macro: **0.9811**

## 7. Регистрация версий модели и назначение alias champion
В MLflow Model Registry зарегистрированы две версии модели breast-cancer-classifier. Версии 1 (LogisticRegression) назначен alias champion с помощью команды python src/promote.py --version 1. Версия 2 (RandomForest) использовалась для демонстрации обновления.

## 8. Описание API, формата входных данных и результатов проверки
Сервис реализован на FastAPI (app.py).
*   GET /health — проверка статуса (возвращает версию модели).
*   GET /metadata — список 30 признаков и имена классов.
*   POST /predict — принимает JSON {"instances": [{...}]}.
**Проверка:** Корректный запрос возвращает код 200. Некорректные запросы (нехватка признаков, пустой массив) возвращают 422 Unprocessable Entity.

## 9. Сборка и запуск Docker-контейнера, проверка совпадений
Образ собран командой sudo docker build -t breast-api:v1 . Смоук-тест check_api.py подтвердил (PASS), что предсказания контейнера полностью совпадают с локальной экспортированной моделью.

## 10. Демонстрация обновления и отката
*   **Обновление:** Alias переведен на v2. Собран образ breast-api:v2, проверка /metadata подтвердила версию модели 2.
*   **Откат:** Контейнер v2 остановлен, запущен сохраненный образ breast-api:v1. Проверка подтвердила работу версии 1. Alias в MLflow возвращен на версию 1.

## 11. Запуск через Docker Compose и проверка healthcheck
Создан compose.yaml с настройкой healthcheck (проверка /health каждые 10 секунд). Запуск выполнен командой sudo docker compose up -d --build. Статус контейнера — healthy.

## 12. Выводы по работе
В ходе работы был освоен полный цикл MLOps. Настроено отслеживание экспериментов, выбрана лучшая модель по метрике F1-macro на валидации, зарегистрированы версии и настроен alias. Модель упакована в Docker-контейнер. Продемонстрирован механизм обновления и отката версий без переобучения, а также запуск с контролем состояния через Docker Compose.
