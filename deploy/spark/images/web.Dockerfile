# syntax=docker/dockerfile:1.7
# web: demo-ui `npm run build:web` 산출물을 nginx로 서빙하고 /api 를 agent-api로 same-origin 프록시한다.
# 컨텍스트는 저장소 루트, 포함 경로는 web.Dockerfile.dockerignore가 연다.

# 개발 서버의 빌드 기준 node v24.18.0 / npm 11.16.0 (2026-09-15).
ARG NODE_IMAGE=node:24.18.0-bookworm-slim
ARG NGINX_IMAGE=nginx:1.29.4-alpine

# 정적 산출물은 아키텍처와 무관하므로 빌드 호스트 플랫폼에서 만든다.
FROM --platform=$BUILDPLATFORM ${NODE_IMAGE} AS build
ENV CI=1 \
    ELECTRON_SKIP_BINARY_DOWNLOAD=1
WORKDIR /src
COPY demo-ui/package.json demo-ui/package-lock.json ./
RUN npm ci
COPY demo-ui/ ./
# VITE_API_BASE_URL을 주지 않는다: index.html의 %VITE_API_BASE_URL%가 남아 window.__DEMO_API_BASE_URL__ = undefined,
# api.ts는 Vite env도 없으므로 '' = same-origin 으로 /api/* 를 부른다.
RUN npm run build:web

FROM ${NGINX_IMAGE}
COPY deploy/spark/web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /src/dist /usr/share/nginx/html
EXPOSE 80
