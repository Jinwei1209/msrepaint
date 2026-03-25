FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git

COPY . /tmp/msrepaint

RUN pip install /tmp/msrepaint && rm -rf /tmp/msrepaint

ENTRYPOINT ["msrepaint"]
