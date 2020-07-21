FROM python:3.6

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY client_interface_clone /usr/src/app/client_interface_clone

COPY api/utils /usr/src/app/api/utils

COPY api/globalserver_task_controller.py /usr/src/app/api
COPY static_variables.json /usr/src/app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r client_interface_clone/requirements.txt

# run the application
CMD ["python", "client_interface_clone/client_interface_controller.py"]
