# Builds the static site and serves it with nginx. Everything runs in the visitor's browser; nginx only serves files.
FROM node:22-alpine AS web
WORKDIR /src
COPY frontend/package.json frontend/package-lock.json frontend/
RUN cd frontend && npm ci
COPY backend/app backend/app
COPY frontend frontend
RUN cd frontend && npm run build

FROM nginx:1.27-alpine
COPY --from=web /src/frontend/dist /usr/share/nginx/html
EXPOSE 80
