import dotenv from "dotenv"
import { v4 as uuidv4 } from "uuid";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
// Removed unused import TaskType
// Removed unused import MemoryVectorStore
// Removed unused import CheerioWebBaseLoader
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { MemorySaver, START, END, Annotation, MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import mongoose from "mongoose";
import express from "express";
import cors from "cors";

// Load environment variables from .env file
dotenv.config()
const lApp = express()

lApp.use(express.json());
lApp.use(cors({
    origin: "*", // Allow all origins for simplicity, adjust as needed
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
    credentials: true,
}))

// Import necessary modules and types

// Connect to MongoDB
await mongoose.connect(process.env.MONGO_DB_URI)
    .then(() => {
        console.log("Connected to MongoDB");
    })
    .catch((error) => {
        console.error("Error connecting to MongoDB:", error);
    });

// Check if the environment variable is set
console.log("GEMINI_API_KEY:", process.env.GEMINI_API_KEY);

// Initialize the Google Generative AI model
const llm = new ChatGoogleGenerativeAI({
    model: "gemini-2.0-flash",
    temperature: 0,
    apiKey: process.env.GEMINI_API_KEY,
})

// Define the prompt template for evaluating prompts
const promptTemplate = ChatPromptTemplate.fromMessages([
    [
        "system",
        `You are a helpful prompt evaluator that evaluates the 
        quality of prompts based on their clarity, specificity, 
        and potential to elicit useful responses. You will be given 
        a prompt and you should provide feedback and recommedation prompts on its quality based 
        on tone( a score of 1 to 5), clarity(a score of 1 to 5), relevance(a score of 1 to 5).
        also the recommendations should start with "Recommendations for Improved Prompts" : then the prompts
        and the overall feedback should start with "Overall Feedback": then the feedback`,
    ],
    [
        "placeholder", "{messages}"
    ]
])

// Define the annotation for the state graph
const GraphAnnotation = Annotation.Root({
    ...MessagesAnnotation.spec,
    language: Annotation<string>(),
})

// Define the state graph for the prompt evaluation workflow
const callModel = async (state: typeof GraphAnnotation.State) => {
    const prompt = await promptTemplate.invoke(state)
    const response = await llm.invoke(prompt)
    return { messages: [response] };
}

// Define the state graph workflow
const workflow = new StateGraph(GraphAnnotation)
    .addNode("model", callModel)
    .addEdge(START, "model")
    .addEdge("model", END)

// Add the language input to the state graph
const app = workflow.compile({ checkpointer: new MemorySaver() });

// Set the language for the state graph
const config = { configurable: { thread_id: uuidv4() } }

// Define the input for the prompt evaluation
const input = {
    messages: [
        {
            role: "user",
            content: "Generate me a node js app for langchain"
        },
    ],
    language: "English",
}
const output = await app.invoke(input, config);
console.log("Output:", output.messages[output.messages.length - 1].content);

// Extract the last message from the output
const lastMessage = output.messages[output.messages.length - 1].content
console.log("Last Message:", lastMessage);

// Extract scores using regex
const toneMatch = typeof lastMessage === "string" ? lastMessage.match(/Tone.*?(\d)\/5/) : null;
const clarityMatch = typeof lastMessage === "string" ? lastMessage.match(/Clarity.*?(\d)\/5/) : null;
const relevanceMatch = typeof lastMessage === "string" ? lastMessage.match(/Relevance.*?(\d)\/5/) : null

// Extract overall feedback and recommendations
const overallFeedbackMatch = typeof lastMessage === "string"
  ? lastMessage.match(/\*\*Overall Feedback:\*\*\n([\s\S]*?)(?=\n\*\*Recommendations)/)
  : null;

const recommendationsMatch = typeof lastMessage === "string"
  ? lastMessage.match(/\*\*Recommendations for Improved Prompts:\*\*\n([\s\S]*)/)
  : null;

console.log("overall Feedback: ", overallFeedbackMatch)
// Build the JSON object
const cleanMarkdown = (text: string) =>
    text.replace(/\*\*(.*?)\*\*/g, "$1"); // Remove **bold**
  
  const feedbackJSON = {
    tone: toneMatch ? parseInt(toneMatch[1]) : null,
    clarity: clarityMatch ? parseInt(clarityMatch[1]) : null,
    relevance: relevanceMatch ? parseInt(relevanceMatch[1]) : null,
    overallFeedback: overallFeedbackMatch
      ? cleanMarkdown(overallFeedbackMatch[1].trim())
      : null,
    recommendations: recommendationsMatch
      ? recommendationsMatch[1]
          .trim()
          .split(/\n\s*\*\s+/) // Match lines starting with "*"
          .filter(Boolean) // Remove empty entries
          .map(line => cleanMarkdown(line.trim()))
      : null
  };
// Log the JSON object
console.log("Feedback JSON:", JSON.stringify(feedbackJSON, null, 2));

lApp.listen(8080, () => {
    console.log("Server is running on port 8080");
})