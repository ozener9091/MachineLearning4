# Лабораторная работа 4. MLOps

**Вариант 18. Учебная классификация типа опухоли**
Данные: `sklearn.datasets.load_breast_cancer` (UCI ML Repository, Breast Cancer Wisconsin Diagnostic).
Задача: бинарная классификация (malignant/benign) по 30 числовым признакам.

## 1. Цель работы
Освоить основы MLOps: учёт и сравнение экспериментов в MLflow, регистрацию и управление версиями моделей, развёртывание сервиса предсказаний с использованием Docker и Docker Compose, проверку его работоспособности, обновление и откат модели.

## 2. Программное окружение
*   **ОС:** Ubuntu 24.04.4 LTS
*   **Python:** 3.12.3
*   **Основные библиотеки:** MLflow 3.5.0, scikit-learn 1.7.2, FastAPI, uvicorn, pandas, numpy.
*   **Docker:** Docker Engine 29.1.3, Docker Compose v2.
*   *Примечание:* В `requirements.txt` версия SQLAlchemy зафиксирована как `>=2.0,<2.1`, так как ветка 2.1 несовместима с MLflow 3.5.0.

![Проверка системы](screenshots/01_system_check.png)
![Установка Compose](screenshots/02_compose_install.png)
![Структура проекта](screenshots/03_project_tree.png)

## 3. Подготовка данных и разбиение
Данные разделены на 3 части (60/20/20) с использованием `train_test_split` и параметра `stratify` для баланса классов. Сначала 20% отложено под финальный `test`, затем из оставшихся 80% выделено 25% под `validation`. Итог: 341 объект в train, 114 в validation, 114 в test. `StandardScaler` обучается только на train внутри `Pipeline`.

## 4. Сравнительная таблица экспериментов
Оценка проводилась по метрике `val_f1_macro` на валидационной выборке.

| № | Алгоритм | Гиперпараметры | val_accuracy | val_f1_macro |
| :--- | :--- | :--- | :--- | :--- |
| 1 | LogisticRegression | C=0.1 | ~0.9737 | ~0.9715 |
| 2 | LogisticRegression | C=1 | ~0.9912 | **~0.9906** |
| 3 | LogisticRegression | C=10 | ~0.9649 | ~0.9626 |
| 4 | RandomForest | trees=50, depth=3 | ~0.9561 | ~0.9531 |
| 5 | RandomForest | trees=100, depth=5 | ~0.9474 | ~0.9439 |
| 6 | RandomForest | trees=200, depth=8 | ~0.9561 | ~0.9531 |

**Обоснование выбора:** Победителем признана модель **LogisticRegression с C=1** (наивысший `val_f1_macro` ≈ 0.9906).

## 5. Регистрация версий и итоговая оценка
В MLflow Model Registry зарегистрированы две версии модели `breast-cancer-classifier`: версия 1 (LogisticRegression, победитель) и версия 2 (RandomForest). С помощью скрипта `promote.py` версии 1 был назначен alias `champion`.

**Итоговая оценка на отложенном тесте (test set):**
*   `test_accuracy`: 0.9824
*   `test_f1_macro`: 0.9811

![Запуск MLflow](screenshots/04_mlflow_start.png)
![Логи MLflow](screenshots/05_mlflow_logs.png)

## 6. Описание API и проверка запросов
Реализован сервис на базе FastAPI (`app.py`).
*   `GET /health` — проверка статуса (возвращает версию модели).
*   `GET /metadata` — список 30 признаков и имена классов.
*   `POST /predict` — принимает JSON `{"instances": [...]}`, возвращает предсказания (0 или 1), имена классов (`malignant`/`benign`), версию модели и `latency_ms`.

**Проверка:**
1. Корректный запрос через `sample_request.json` возвращает код 200 и правильные предсказания. Смоук-тест `check_api.py` подтверждает совпадение с локальной моделью (PASS).
2. Некорректные запросы (нехватка признаков, пустой массив) возвращают код `422 Unprocessable Entity` с детальным описанием ошибки валидации.

![Сборка Docker v1](screenshots/06_docker_build_v1.png)
![Ошибка 422](screenshots/07_api_422.png)
![Смоук-тест v1](screenshots/08_check_api_v1.png)
![Swagger UI](screenshots/09_swagger.png)

## 7. Сборка Docker, обновление и откат
1. Собран образ `breast-api:v1`.
2. Запущен контейнер, проверен через `curl` и Swagger UI.
3. **Обновление:** Назначен alias `champion` на версию 2. Выполнен экспорт модели, собран образ `breast-api:v2`, контейнер перезапущен. Проверка `/metadata` подтвердила версию 2.
4. **Откат:** Контейнер v2 остановлен, запущен сохраненный образ `breast-api:v1`. Alias в MLflow возвращен на версию 1. Проверка подтвердила работу версии 1.

![Переключение на v2](screenshots/10_promote_export_v2.png)
![Сборка Docker v2](screenshots/11_docker_build_v2.png)
![Откат к версии 1](screenshots/12_rollback_v1.png)
![Смоук-тест после отката](screenshots/13_check_api_rollback.png)

## 8. Запуск через Docker Compose
Создан `compose.yaml` с настройкой `healthcheck` (проверка эндпоинта `/health` каждые 10 секунд). Запуск выполнен командой `sudo docker compose up -d --build`. Статус контейнера — `healthy`.

![Docker Compose healthcheck](screenshots/14_compose_healthcheck.png)

## 9. Выводы
В ходе работы был освоен полный цикл MLOps. Настроено отслеживание экспериментов, выбрана лучшая модель (LogisticRegression) по метрике `F1-macro` на валидации, зарегистрированы версии и настроен alias. Модель упакована в Docker-контейнер с FastAPI. Продемонстрирован механизм бесшовного обновления и отката версий без переобучения, а также запуск с контролем состояния через Docker Compose.
