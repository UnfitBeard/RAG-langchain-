import { ChatGoogleGenerativeAI } from '@langchain/google-genai'
import { HumanMessage, SystemMessage } from '@langchain/core/messages'
import { ChatPromptTemplate } from "@langchain/core/prompts"
import dotenv from 'dotenv'

dotenv.config()

const model = new ChatGoogleGenerativeAI({
    model: "gemini-2.0-flash",
    temperature: 0,
    apiKey: process.env.GOOGLE_API_KEY
})

const messages = [
    new SystemMessage("Translate the following from English to Italian"),
    new HumanMessage("hi!")
]

// invoke Runnable Interface
// console.log(await model.invoke(messages))
// console.log(await model.invoke("Hello"))
// console.log(await model.invoke([{role: "user", content: "Hello"}]))

// Streaming Runnable interface
// const stream = await model.stream(messages);

// const chunks = [];
// for await (const chunk of stream) {
//     chunks.push(chunk);
//     console.log(`${chunk.content}`)
// }

// Prompt Templates
const systemTemplate = "Translate the following from English to {language}";

const promptTemplate = ChatPromptTemplate.fromMessages([
    ["system", systemTemplate],
    ["user", "{text}"]
])

const promptValue = await promptTemplate.invoke({
    language: "italian",
    text: "hi"
})

// console.log(promptValue.toChatMessages()) // To print the messages directly
const response = await model.invoke(promptValue)
console.log(`${response.content}`)