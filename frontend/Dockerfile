FROM node:lts-alpine

WORKDIR /src

COPY package.json ./
COPY package-lock.json ./
RUN npm install

COPY . . 

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]