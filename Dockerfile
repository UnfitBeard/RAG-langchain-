FROM node:alpine
WORKDIR /app
COPY RAG2Practise.ts .
RUN npm install
CMD ["npm", "start"]