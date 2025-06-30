import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import dotenv from 'dotenv'
import { START, END, MessagesAnnotation, StateGraph, MemorySaver, Annotation } from '@langchain/langgraph'
import { v4 as uuidv4 } from 'uuid'
import { ChatPromptTemplate, PromptTemplate } from "@langchain/core/prompts";
import {
    SystemMessage,
    HumanMessage,
    AIMessage,
    trimMessages,
} from "@langchain/core/messages";

const trimmer = trimMessages({
    maxTokens: 10,
    strategy: "last",
    tokenCounter: (msgs) => msgs.length,
    includeSystem: true,
    allowPartial: false,
    startOn: "human",
});
dotenv.config()

const llm = new ChatGoogleGenerativeAI({
    model: "gemini-2.0-flash",
    temperature: 0,
    apiKey: process.env.GEMINI_API_KEY,
});

const messages = [
    new SystemMessage("you're a good assistant"),
    new HumanMessage("hi! I'm bob"),
    new AIMessage("hi!"),
    new HumanMessage("I like vanilla ice cream"),
    new AIMessage("nice"),
    new HumanMessage("whats 2 + 2"),
    new AIMessage("4"),
    new HumanMessage("thanks"),
    new AIMessage("no problem!"),
    new HumanMessage("having fun?"),
    new AIMessage("yes!"),
];

await trimmer.invoke(messages);
// console.log(await llm.invoke([{ role: "user", content: "Hi im bob" }]))
// console.log(await llm.invoke([{ role: "user", content: "Whats my name" }]))
// no concept of state yet

// How about we pass the entire histroy into a model
// const response = await llm.invoke([
//     { role: "user", content: "Hi im bob" },
//     { role: "assistant", content: "Hello Bob! How can I assist you today?" },
//     { role: "user", content: "Whats my name" }
// ])

// console.log(response)

// Using a checkpointer to persist messages
// // Define the function that calls the model
// const callModel = async (state: typeof MessagesAnnotation.State) => {
//     const response = await llm.invoke(state.messages)
//     return { messages: response }
// }

// // Define a state graph
// const workflow = new StateGraph(MessagesAnnotation)
//     .addNode("model", callModel)
//     .addEdge(START, "model")
//     .addEdge("model", END);

// // add memory
// const memory = new MemorySaver()
// const app = workflow.compile({ checkpointer: memory })

// // create config - support multiple convo threads e.g when app has multiple users
// const config = { configurable: { thread_id: uuidv4() } }

// // how about we invoke tha appp now
// const input = [
//     {
//         role: "user",
//         content: "Hi! I'm Bob."
//     }
// ]

// const output = await app.invoke({ messages: input }, config)
// // // output has all messages in the state
// // // This will log the last message in the conversation
// // console.log(output.messages[output.messages.length - 1])

// const input2 = [
//     {
//         role: "user",
//         content: "What's my name?",
//     },
// ];
// const output2 = await app.invoke({ messages: input2 }, config);
// // console.log(output2.messages[output2.messages.length - 1]);

// // Lets change conversation thread
// const config2 = { configurable: { thread_id: uuidv4() } }
// const input3 = [
//     {
//         role: "user",
//         content: "What's my name?",
//     },
// ];
// const output3 = await app.invoke({ messages: input3 }, config2);
// // console.log(output3.messages[output3.messages.length - 1]);

// // Going back to the intiial convo
// const input4 = [
//     {
//         role: "user",
//         content: "Whats my name?",
//     }
// ]

// // const output4 = await app.invoke({ messages: input4 }, config);
// console.log(output4.messages[output4.messages.length - 1]);

// How about changing the prompt template

// const promptTemplate = ChatPromptTemplate.fromMessages([
//     [
//         "system",
//         "You talk like a pirate. Answer the user's questions to the best of your ability.",
//     ],
//     [
//         "placeholder", "{messages}"
//     ]
// ])

// const callModel2 = async (state: typeof MessagesAnnotation.State) => {
//     const prompt = await promptTemplate.invoke(state)
//     const response = await llm.invoke(prompt.messages)
//     // update messages history with response
//     return { messages: [response] }
// }

// // Define a new Graph
// const workflow2 = new StateGraph(MessagesAnnotation)
//     .addNode("model", callModel2)
//     .addEdge(START, "model")
//     .addEdge("model", END)

// // Add memory
// const app2 = workflow2.compile({ checkpointer: new MemorySaver() })

// const config3 = { configurable: { thread_id: uuidv4() } }
// const input5 = [
//     {
//         role: "user",
//         content: "Hi! I'm Jim."
//     }
// ]

// const output5 = await app2.invoke({ messages: input5 }, config3)
// console.log(output5.messages[output5.messages.length - 1])
// const input6 = [
//     {
//         role: "user",
//         content: "What is my name?"
//     }
// ]

// const output6 = await app2.invoke({ messages: input6 }, config3)
// console.log(output6.messages[output6.messages.length - 1])

// Increase prompt templates complexity
const promptTemplate2 = ChatPromptTemplate.fromMessages([
    [
        "system",
        "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
    ],
    ["placeholder", "{messages}"],
]);


// Define the State
const GraphAnnotation = Annotation.Root({
    ...MessagesAnnotation.spec,
    language: Annotation<string>(),
});

// // Define the function that calls the model
// const callModel3 = async (state: typeof GraphAnnotation.State) => {
//     const prompt = await promptTemplate2.invoke(state);
//     const response = await llm.invoke(prompt);
//     return { messages: [response] };
// };

// const workflow3 = new StateGraph(GraphAnnotation)
//     .addNode("model", callModel3)
//     .addEdge(START, "model")
//     .addEdge("model", END);

// const app3 = workflow3.compile({ checkpointer: new MemorySaver() });
// const config4 = { configurable: { thread_id: uuidv4() } };
// const input6 = {
//     messages: [
//         {
//             role: "user",
//             content: "Hi im bob",
//         },
//     ],
//     language: "Spanish",
// };
// const output7 = await app3.invoke(input6, config4);
// console.log(output7.messages[output7.messages.length - 1]);

// const input7 = {
//     messages: [
//         {
//             role: "user",
//             content: "What is my name?",
//         },
//     ],
// };
// const output8 = await app3.invoke(input7, config4);
// console.log(output8.messages[output8.messages.length - 1]);


const promptTemplate3 = ChatPromptTemplate.fromMessages([
    [
        "system",
        "Your task is to help the user refine their prompts and provide suggestions for improvement in {language}. ",
    ],
    [
        "placeholder",
        "{messages}",
    ]
])
// const configPromptHelper = { configurable: { thread_id: uuidv4() } };

// // Define the function that calls the model
// const callModel4 = async (state: typeof GraphAnnotation.State) => {
//     const trimmedMessages = await trimmer.invoke(state.messages);
//     const prompt = await promptTemplate3.invoke({
//         messages: trimmedMessages,
//         language: state.language || "English",
//     });
//     const response = await llm.invoke(prompt);
//     return { messages: [response] };
// };

// const workflow4 = new StateGraph(GraphAnnotation)
//     .addNode("model", callModel4)
//     .addEdge(START, "model")
//     .addEdge("model", END);

// const app4 = workflow4.compile({ checkpointer: new MemorySaver() });
// const inputPrompt = {
//     messages: [
//         {
//             role: "user",
//             content: "Hi im trying to write a prompt for a chatbot. Can you help me?",
//         },
//     ],
//     language: "English",
// };
// const outputPrompt = await app4.invoke(inputPrompt, configPromptHelper);
// console.log(outputPrompt.messages[outputPrompt.messages.length - 1]);

// const inputPrompt1 = {
//     messages: [
//         {
//             role: "user",
//             content: "Am trying to build a UI dashboard for my app. Can you help me?",
//         },
//     ],
// };
// const outputPrompt1 = await app4.invoke(inputPrompt1, configPromptHelper);
// console.log(outputPrompt1.messages[outputPrompt1.messages.length - 1]);

// // trimmed output
// const inputPrompt2 = {
//     messages: [
//         {
//             role: "user",
//             content: "What did i ask you to do help me with?",
//         },
//     ],
// };
// const outputPrompt2 = await app4.invoke(inputPrompt2, configPromptHelper);
// console.log(outputPrompt2.messages[outputPrompt2.messages.length - 1]);
