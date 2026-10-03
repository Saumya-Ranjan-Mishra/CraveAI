FROM node:22-alpine AS build

WORKDIR /src/ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.27-alpine
LABEL org.opencontainers.image.description="CraveAI React menu search UI"

COPY --from=build /src/ui/dist/ /usr/share/nginx/html/
COPY ui/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080