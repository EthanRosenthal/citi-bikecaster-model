SHELL = /bin/bash
MAKEFLAGS = "-s"
.DEFAULT_GOAL := build


REGION = "us-east-1"
REGISTRY = "069757761977.dkr.ecr.us-east-1.amazonaws.com/citi-bikecaster-train"
COMMIT = $(shell git rev-parse HEAD)

PROJECT = citi-bikecaster-train
VERSION = $(shell python setup.py --version)
DOCKER = docker
BUILD_TIMESTAMP = $(shell date +%s)

export VERSION

ifeq ($(DIACO_ENV), )
        DIACO_ENV="local"
endif


ifeq ($(DIACO_ENV),production)
        TAG="latest"
else
        TAG=$(DIACO_ENV)
endif


help: ## Show this help page
        @python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

init-deploy: ## Creates an ECR Repository and initializes the Terraform backend
        @aws ecr create-repository --repository-name recommender-api
        @cd infrastructure && terraform init && terraform workspace new staging && terraform workspace new production


.build: $(shell find .  -path ./.git -prune -o -path ./.artifacts -prune -o  -path ./.import -prune -o -path ./.build -prune -o -path */*.pyc -prune -o -type f -print )
        @aws ecr get-login --no-include-email | sh
        @$(DOCKER) build -t $(REGISTRY)/$(PROJECT):latest --build-arg GIT_COMMIT=$(COMMIT) --build-arg BUILD_TIMESTAMP=$(BUILD_TIMESTAMP) --build-arg SEM_VERSION=$(VERSION) .
        @touch .build

build: .build ## build the docker container

pyshell: build  ## build container and run a python shell inside it
        @docker-compose run --rm api python


test: build ## run build image and run tests in the test environment
        @docker-compose rm -f tests
        @docker-compose  run --name recommender-api-tests tests
        @docker cp recommender-api-tests:/app/.artifacts .
        @docker-compose rm -f tests

start: build  ## Create and start api containers
        @docker-compose up -d api

down: ## Stop and remove containers, networks, images, and volumes
        @docker-compose down

shell: build ## build container and run a shell inside it
        @docker-compose run --rm api /bin/sh


push: build tag ## build and push container into ECR repository
        @$(DOCKER) push $(REGISTRY)/$(PROJECT):$(VERSION)
        @$(DOCKER) push $(REGISTRY)/$(PROJECT):$(TAG)

tag: build
        @$(DOCKER) tag $(REGISTRY)/$(PROJECT):latest $(REGISTRY)/$(PROJECT):$(VERSION)
        @$(DOCKER) tag $(REGISTRY)/$(PROJECT):latest $(REGISTRY)/$(PROJECT):$(TAG)
