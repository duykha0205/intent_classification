FROM service_requirements
RUN mkdir /app

# Install dependencies
WORKDIR /app
RUN mkdir /.cache
RUN chmod 777 /.cache
RUN apt-get update
RUN apt install libgl1-mesa-glx -y
#RUN apt-get install ffmpeg libsm6 libxext6  -y


COPY framework /app/framework
COPY interface /app/interface
COPY utilities /app/utilities
COPY parameters /app/parameters
COPY app.py /app


RUN mkdir /.mxnet
RUN chmod 777 /.mxnet

# Switch back to a non-root user
USER 1001
# Expose port 5000
EXPOSE 5000

# Run gunicorn
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:5000", "-w","1","--timeout", "30", "app:app"]
