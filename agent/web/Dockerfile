FROM node:20-alpine AS base
WORKDIR /app

COPY package.json package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

COPY . .

ARG NEXT_PUBLIC_API_BASE
ARG NEXT_PUBLIC_DEBUG_UI
ENV NEXT_PUBLIC_API_BASE=${NEXT_PUBLIC_API_BASE}
ENV NEXT_PUBLIC_DEBUG_UI=${NEXT_PUBLIC_DEBUG_UI}

RUN npm run build

EXPOSE 3000
CMD ["npm", "run", "start"]
