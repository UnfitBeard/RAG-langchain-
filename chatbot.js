import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import dotenv from 'dotenv'
import { START, END} from '@langchain/langgraph'

dotenv.config()

const llm = new ChatGoogleGenerativeAI({
    model: "gemini-2.0-flash",
    temperature: 0,
    apiKey: process.env.GOOGLE_API_KEY,
});

// console.log(await llm.invoke([{ role: "user", content: "Hi im bob" }]))
// console.log(await llm.invoke([{ role: "user", content: "Whats my name" }]))
// no concept of state yet

// How about we pass the entire histroy into a model
const response = await llm.invoke([
    { role: "user", content: "Hi im bob" },
    { role: "assistant", content: "Hello Bob! How can I assist you today?" },
    { role: "user", content: "Whats my name" }
])

console.log(response)

// Using a checkpointer to persist messages
