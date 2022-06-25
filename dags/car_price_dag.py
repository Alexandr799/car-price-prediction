import datetime as dt
from airflow.models import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import sys
import os
path = os.path.expanduser('~/cat_price_predict')
sys.path.insert(0, path)
from modules.pipeline import make_model
from modules.predict import predict

args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2022, 6, 23),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1),
    'depends_on_past': False,
}

with DAG(
        dag_id='car_price_prediction',
        schedule_interval="00 15 * * *",
        default_args=args,
) as dag:

    start = BashOperator(
        task_id='first_task',
        bash_command='echo "Lets go!"',
        dag=dag
    )

    pipeline = PythonOperator(
        task_id='pipe',
        python_callable=make_model,
        dag=dag
    )
    predict = PythonOperator(
        task_id='predict',
        python_callable=predict,
        dag=dag
    )
    end = BashOperator(
        task_id='last_task',
        bash_command='echo "Directed by Robert B. Weide"',
        dag=dag
    )


    start >> pipeline >> predict >> end