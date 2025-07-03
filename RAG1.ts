import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import 'cheerio'
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import dotenv from "dotenv";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { pull } from 'langchain/hub'
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { Document } from "@langchain/core/documents";
import { Annotation } from "@langchain/langgraph";
import { concat } from "@langchain/core/utils/stream";
import { StateGraph } from "@langchain/langgraph";
import z from 'zod'
import { contents } from "cheerio/dist/commonjs/api/traversing";
// Initialize with an embedding model

dotenv.config()

const llm = new ChatGoogleGenerativeAI({
    model: "gemini-2.0-flash",
    temperature: 0,
    apiKey: process.env.GEMINI_API_KEY,
});

const embeddings = new GoogleGenerativeAIEmbeddings({
    model: "text-embedding-004", // 768 dimensions
    taskType: TaskType.RETRIEVAL_DOCUMENT,
    title: "Document title",
    apiKey: process.env.GEMINI_API_KEY,
});

const vectorStore = new MemoryVectorStore(embeddings);

const pTagSelector = `p`;
const cheerioLoader = new CheerioWebBaseLoader(
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    {
        selector: pTagSelector,
    }
)

const docs = await cheerioLoader.load();

console.assert(docs.length === 1)
console.log(`Total characters: ${docs[0].pageContent.length}`);
console.log(`${docs[0].pageContent.slice(0, 500)}`);

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
})

const allSplits = await splitter.splitDocuments(docs);
console.log(`Total splits: ${allSplits.length} subDocuments.`);
await vectorStore.addDocuments(allSplits)

// Retrieval Stage
const promptTemplate = await pull<ChatPromptTemplate>("rlm/rag-prompt")

// Example
const examplePrompt = await promptTemplate.invoke({
    context: "(context from vector store)",
    question: "(What is the difference between RAG and LLM?)",
})
const example_messages = examplePrompt.messages;

console.assert(example_messages.length === 1)
console.log(example_messages[0].content)

const InputStateAnnotation = Annotation.Root({
    question: Annotation<string>,
})

const StateAnnotation = Annotation.Root({
    question: Annotation<string>,
    context: Annotation<Document[]>,
    answer: Annotation<string>,
})

const retrieve = async (state: typeof StateAnnotation.State) => {
    const retrivedDocs = await vectorStore.similaritySearch(state.question)
    return {
        context: retrivedDocs
    }
}

const generate = async (state: typeof StateAnnotation.State) => {
    const docsContent = state.context.map(doc => doc.pageContent).join("\n")
    const messages = await promptTemplate.invoke({
        question: state.question,
        context: docsContent,
    })

    const response = await llm.invoke(messages)
    return {
        answer: response.content
    }
}

const graph = new StateGraph(StateAnnotation)
    .addNode("retrieve", retrieve)
    .addNode("generate", generate)
    .addEdge("__start__", "retrieve")
    .addEdge("retrieve", "generate")
    .addEdge("generate", "__end__")
    .compile();

let inputs = { question: "What is task decomposition?" };

const result = await graph.invoke(inputs);
console.log(result.context.slice(0, 2))
console.log(`Answer: ${result["answer"]}`);

// stream steps
console.log('inputs')
console.log("\n====\n")
for await (const chunk of await graph.stream(inputs, {
    streamMode: "updates",
})) {
    console.log(chunk);
    console.log("\n====\n");
}

// stream tokens
// const stream = await graph.stream(inputs, { streamMode: "messages" });
// for await (const key of Object.keys(stream)) {
//     const message = stream[key];
//     process.stdout.write(message + '|');
// }

const template = `Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:`;

const promptTemplateCustom = ChatPromptTemplate.fromMessages([
    ["user", template],
]);

// Query analysis
const totalDocuments = allSplits.length;
const third = Math.floor(totalDocuments / 3)
allSplits.forEach((document, i) => {
    if (i < third) {
        document.metadata["section"] = "beginning";
    } else if (i < 2 * third) {
        document.metadata["section"] = "middle";
    } else {
        document.metadata["section"] = "end";
    }
})

console.log(allSplits[0].metadata)

const vectorStoreQA = new MemoryVectorStore(embeddings);
await vectorStoreQA.addDocuments(allSplits)

// query schema
const searchSchema = z.object({
    query: z.string().describe("Search query to run."),
    section: z.enum(["beginning", "middle", "end"]).describe("Section to query.")
});

const structuredLlm = llm.withStructuredOutput(searchSchema);

const StateAnnotationQA = Annotation.Root({
    question: Annotation<string>,
    search: Annotation<z.infer<typeof searchSchema>>,
    context: Annotation<Document[]>,
    answer: Annotation<string>,
});

const analyzeQuery = async (state: typeof InputStateAnnotation.State) => {
    const result = await structuredLlm.invoke(state.question)
    return { search: result }
}

const retrieveQA = async (state: typeof StateAnnotationQA.State) => {
    const filter = (doc) => doc.metadata.section === state.search.section;
    const retrivedDocs = await vectorStore.similaritySearch(
        state.search.query,
        2,
        filter
    )
    return { context: retrivedDocs }
}

const generateQA = async (state: typeof StateAnnotationQA.State) => {
    const docsContent = state.context.map((doc) => doc.pageContent).join("\n")
    const messages = await promptTemplate.invoke({
        question: state.question,
        context: docsContent,
    })

    const response = await llm.invoke(messages)
    return {answer: response.content}
}

const graphQA = new StateGraph(StateAnnotationQA)
.addNode("analyzeQuery", analyzeQuery)
.addNode("retrieveQA", retrieveQA)
.addNode("generateQA", generateQA)
.addEdge("__start__", "analyzeQuery")
.addEdge("analyzeQuery", "retrieveQA")
.addEdge("retrieveQA", "generateQA")
.addEdge("generateQA", "__end__")
.compile()

// asking for info from the end of the document
let inputsQA = {
    question: "What does the end of the post say about Task Decomposition?"
}

console.log(inputsQA)
console.log("\n====\n");
for await (const chunk of await graphQA.stream(inputsQA, {
    streamMode: "updates",
})) {
    console.log(chunk);
    console.log("\n====\n");
}