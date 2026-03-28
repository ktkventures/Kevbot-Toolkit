FROM node:20-slim

WORKDIR /app

# Install dependencies (patches/ must be present for postinstall → patch-package)
COPY package.json package-lock.json ./
COPY patches/ patches/
RUN npm ci

# Copy source
COPY . .

# Build-time env vars (Railway passes these as build args)
ARG NEXT_PUBLIC_SUPABASE_URL
ARG NEXT_PUBLIC_SUPABASE_ANON_KEY
ARG NEXT_PUBLIC_API_URL

ENV NEXT_PUBLIC_SUPABASE_URL=$NEXT_PUBLIC_SUPABASE_URL
ENV NEXT_PUBLIC_SUPABASE_ANON_KEY=$NEXT_PUBLIC_SUPABASE_ANON_KEY
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

# Build
RUN npm run build

# Run — Railway sets PORT automatically
ENV NODE_ENV=production
CMD sh -c "npx next start -p ${PORT:-3000}"
