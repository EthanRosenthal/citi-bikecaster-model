SHELL = /bin/bash
MAKEFLAGS = "-s"
.DEFAULT_GOAL := build


REGION = "us-east-1"
REGISTRY = "069757761977.dkr.ecr.us-east-1.amazonaws.com"
COMMIT = $(shell git rev-parse HEAD)

PROJECT = citi-bikecaster-train
VERSION = "0.1.0"
DOCKER = docker
BUILD_TIMESTAMP = $(shell date +%s)

export VERSION

TAG = "latest"

.build: $(shell find .  -path ./.git -prune -o -path ./.artifacts -prune -o  -path ./.import -prune -o -path ./.build -prune -o -path */*.pyc -prune -o -type f -print )
		@aws ecr get-login --no-include-email | sh
		@$(DOCKER) build -t $(REGISTRY)/$(PROJECT):latest --build-arg GIT_COMMIT=$(COMMIT) --build-arg BUILD_TIMESTAMP=$(BUILD_TIMESTAMP) --build-arg SEM_VERSION=$(VERSION) .
		@touch .build

build: .build ## build the docker container

shell: build ## build container and run a shell inside it
		@docker run --rm -ti -v ~/.aws/:/root/.aws/ $(REGISTRY)/$(PROJECT):latest /bin/sh

push: build tag ## build and push container into ECR repository
		@$(DOCKER) push $(REGISTRY)/$(PROJECT):$(VERSION)
		@$(DOCKER) push $(REGISTRY)/$(PROJECT):$(TAG)

tag: build
		@$(DOCKER) tag $(REGISTRY)/$(PROJECT):latest $(REGISTRY)/$(PROJECT):$(VERSION)
		@$(DOCKER) tag $(REGISTRY)/$(PROJECT):latest $(REGISTRY)/$(PROJECT):$(TAG)
