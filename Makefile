IMAGE_NAME = causal_ai
CONTAINER_NAME = causal_ai_container

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -it --rm -v $(PWD):/app --name $(CONTAINER_NAME) $(IMAGE_NAME)

jupyter:
	docker run -it --rm -p 8888:8888 -v $(PWD):/app --name $(CONTAINER_NAME) $(IMAGE_NAME) \
	jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token=''

clean:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true
	docker rmi $(IMAGE_NAME) || true
