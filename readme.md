# Модель по предсказнию цены автомобиля

## О проекте:

* Расчеты представлены в файле calculation.ihynb.
* Модель/пайплайн для внедрения представлена в в файле pipeline.py, а код для вставки в airflow в файлe dag.py.
* Пайплайн в аирфлоу представлен следующей видом: модель обучается на тренировочном датасете,
а далее выполняет предсказания для анкет в папке test после чего сохраняет csv файл в папку predictions.
