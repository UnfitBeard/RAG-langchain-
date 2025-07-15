import { Schema, model } from "mongoose";

const promptSchema = new Schema({
    prompt:String,
    feedback: String,
    scores: {
        tone: Number,
        clarity: Number,
        relevance: Number,
    },
    createdAt: {
        type: Date,
        default: Date.now,
    }
})

const promptModel = model("Prompt", promptSchema);
export default promptModel;