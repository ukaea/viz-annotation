# Use official Node.js image as base
FROM node:18-alpine

# Set working directory inside the container
WORKDIR /app

# Copy package.json first (to optimize caching)
COPY package.json ./

# Install dependencies using Yarn
RUN yarn install --frozen-lockfile

# Copy the rest of the application files
COPY . .

# Build the Next.js application
# RUN yarn build

# Expose the port Next.js runs on
EXPOSE 3001

# Start the application
CMD ["yarn", "dev"]