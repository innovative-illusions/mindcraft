import OpenAIApi from 'openai';
import { getKey, hasKey } from '../utils/keys.js';
import { strictFormat } from '../utils/text.js';

export class GPT {
    constructor(model_name, url) {
        this.model_name = model_name;

        let config = {};
        if (url)
            config.baseURL = url;

        if (hasKey('OPENAI_ORG_ID'))
            config.organization = getKey('OPENAI_ORG_ID');

        config.apiKey = getKey('OPENAI_API_KEY');

        this.openai = new OpenAIApi(config);
    }

    async sendRequest(turns, systemMessage, stop_seq='***') {
        let messages = [{'role': 'system', 'content': systemMessage}].concat(turns);

        const pack = {
            model: this.model_name || "gpt-4o",
            messages,
            response_format: {
                type: "json_schema",
                json_schema: {
                        name: "reasoning",
                        schema: {
                            type: "object",
                            properties: {
                                cot1: {
                                    type: "string",
                                    description: "Think step by step about what you will do next.",
                                },
                                cot2: {
                                    "type": "string",
                                    "description": "Determine the optimal approach.",
                                },
                                cot3: {
                                    type: "string",
                                    description: "Plan ahead.",
                                },
                                cot4: {
                                    type: "string",
                                    description: "Explain your reasoning behind this choice of action.",
                                },
                                cot5: {
                                    "type": "string",
                                    "description": "Plan ahead further.",
                                },
                                output: {
                                    "type": "string",
                                    "description": "This is the final response !withThisSyntax(params) commands are permitted IF THE SYSTEM ASKS FOR CODE OUTPUT ONLY CODE.",
                                },
                            },
                            "required": [
                                "cot1",
                                "cot2",
                                "cot3",
                                "cot4",
                                "cot5",
                                "output",
                            ],
                            "additionalProperties": false,
                        },
                        "strict": true,
                    },
            },
            temperature: 0.7,
            top_p: 0.9,
            frequency_penalty: 0.2,
            presence_penalty: 0.5,
            stop: stop_seq,
        };
        if (this.model_name.includes('o1')) {
            pack.messages = strictFormat(messages);
            delete pack.stop;
        }

        let res = null;
        try {
            console.log('Awaiting openai api response...')
            // console.log('Messages:', messages);
            let completion = await this.openai.beta.chat.completions.parse(pack);
            if (completion.choices[0].finish_reason == 'length')
                throw new Error('Context length exceeded');
            console.log('Received.')
            if (completion.choices[0].message) {
                res = completion.choices[0].message.parsed;
                console.log('Final answer:', res);
                res = completion.choices[0].message.parsed.output;

            } else {
                res = 'final_answer is missing from the response.';
            }
        }
        catch (err) {
            if ((err.message == 'Context length exceeded' || err.code == 'context_length_exceeded') && turns.length > 1) {
                console.log('Context length exceeded, trying again with shorter context.');
                return await this.sendRequest(turns.slice(1), systemMessage, stop_seq);
            } else {
                console.log(err);
                return res;
                res = 'My brain disconnected, try again.';
            }
        }
        return res;
    }

    async embed(text) {
        const embedding = await this.openai.embeddings.create({
            model: this.model_name || "text-embedding-3-small",
            input: text,
            encoding_format: "float",
        });
        return embedding.data[0].embedding;
    }
}
