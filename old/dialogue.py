import json
import os
import textwrap
from src.vectorstore import VectorstoreHandler
from src.models import init_emb, init_llm
from src.rag_tools import build_rag_chain



# Function to process RAG based on config
def create_rag(config, handler):
    subfolder = config["subfolder"]
    llm_name = config["llm_name"]
    emb_name = config["emb_name"]
    k = config["k"]

    dir_path = os.path.join(SOURCES_DIR, subfolder)
    emb = init_emb(emb_name)
    llm = init_llm(llm_name)
    vs = handler.build_vectorstore(dir_path, emb, emb_name)
    retriever = handler._init_retriever(vs, dir_path, k)
    chain = build_rag_chain(retriever, llm)
    return chain, retriever


def generate_answers(prompt, rags):
    rag_responses = {}
    print(f"\nQuerying all RAGs for prompt: '{prompt}'\n{'=' * 80}")
    for rag_name, (chain, _) in rags.items():
        print(f"--- Querying RAG: {rag_name} ---")
        output = chain.invoke(prompt)
        rag_responses[rag_name] = {"answer": output["answer"], "docs": output["docs"]}
    return rag_responses


def generate_follow_up_questions(aggregation_llm, agent_answers):
    follow_up_questions = {}
    for agent_name, answer in agent_answers.items():
        prompt = prompt = (
        f"Here is the response from the agent '{agent_name}':\n{answer}\n\n"
        "Your task is to act as an intermediary and generate a follow-up question to ask the agent. "
        "The follow-up question should clarify or expand on the information provided by the agent, "
        "seeking either broader context or deeper detail. This question is intended for the agent "
        "to answer, not the user, so do not include questions directed at the user. "
        "Include nothing in the answer but the question from a first-person perspective."
        )
        response = aggregation_llm.invoke(prompt)
        follow_up_questions[agent_name] = response["answer"] if isinstance(response, dict) else response
    return follow_up_questions


def aggregate_final_response(aggregation_llm, all_responses, prompt):
    aggregation_prompt = (
    f"Here are the collected responses (initial and follow-up) for the prompt: '{prompt}':\n"
    )
    for agent_name, responses in all_responses.items():
        aggregation_prompt += (
            f"\n[{agent_name} Initial Answer]:\n{responses['initial']}\n"
            f"[{agent_name} Follow-Up Answer]:\n{responses['follow_up']}\n"
        )

    aggregation_prompt += (
        "\nUsing the information above, please provide a **concise, clear, and structured** "
        "final answer in the following format:\n\n"
        "1. **Introduction**: Briefly summarize the context or problem.\n"
        "2. **Key Points**: Highlight the main insights or steps (use bullet points).\n"
        "3. **Examples/Applications**: Provide concrete examples or use cases if relevant.\n"
        "4. **Conclusion**: Summarize the takeaways in one or two sentences.\n\n"
        "If the answer includes code or mathematical notation, use appropriate Markdown formatting."
    )
    aggregated_output = aggregation_llm.invoke(aggregation_prompt)
    return aggregated_output["answer"] if isinstance(aggregated_output, dict) else aggregated_output


def organize_into_json(prompt, initial_responses, follow_up_questions, follow_up_responses, final_answer):
    def serialize_docs(docs):
        """Converts non-serializable Document objects into JSON-serializable dictionaries."""
        return [
            {
                "content": getattr(doc, "content", "No content available"),
                "metadata": getattr(doc, "metadata", {})
            }
            for doc in docs
        ]

    return {
        "prompt": prompt,
        "initial_responses": {
            agent_name: {
                "answer": response["answer"],
            }
            for agent_name, response in initial_responses.items()
        },
        "follow_up_questions": follow_up_questions,
        "follow_up_responses": {
            agent_name: {
                "answer": response["answer"],
            }
            for agent_name, response in follow_up_responses.items()
        },
        "final_answer": final_answer,
    }


def display_from_json(conversation_json):
    print(f"\nUser Prompt: {conversation_json['prompt']}\n{'=' * 80}")
    for agent_name, response in conversation_json["initial_responses"].items():
        print(f"\n--- {agent_name} Initial Answer ---")
        print(textwrap.fill(response["answer"], width=80))

    print("\nFollow-Up Questions and Answers:\n" + "=" * 80)
    for agent_name, question in conversation_json["follow_up_questions"].items():
        print(f"\n--- {agent_name} Follow-Up Question ---")
        print(textwrap.fill(question, width=80))
        print(f"\n--- {agent_name} Follow-Up Answer ---")
        print(
            textwrap.fill(
                conversation_json["follow_up_responses"][agent_name]["answer"], width=80
            )
        )

    print("\nFinal Aggregated Answer:\n" + "=" * 80)
    print(textwrap.fill(conversation_json["final_answer"], width=80))

#####################################################################################################################
## SETUP ############################################################################################################
#####################################################################################################################
SOURCES_DIR = "sources"
JSON_PATH = "conversation.json"
AGGREGATION_LLM_NAME = "Llama3.2-3b"
RAG_CONFIG = {
    "book": {
        "subfolder": "book",
        "llm_name": "Llama3.2-3b",
        "emb_name": "hf-minilm-l6-v2",
        "k": 10,
    },
    "lectures": {
        "subfolder": "lectures",
        "llm_name": "Llama3.2-3b",
        "emb_name": "hf-minilm-l6-v2",
        "k": 10,
    },
}

test_prompt = "Explain R-CNN"
#####################################################################################################################
#####################################################################################################################


handler = VectorstoreHandler(SOURCES_DIR, force_rebuild=False)

# Initialize RAGs
rags = {}
for rag_name, rag_config in RAG_CONFIG.items():
    rags[rag_name] = create_rag(rag_config, handler)

# Initialize Aggregation LLM
aggregation_llm = init_llm(AGGREGATION_LLM_NAME)

# Prompt and First Round
initial_responses = generate_answers(test_prompt, rags)

# Follow-Up Questions
agent_answers = {k: v["answer"] for k, v in initial_responses.items()}
follow_up_questions = generate_follow_up_questions(aggregation_llm, agent_answers)

# Second Round
follow_up_responses = {}
for agent_name, question in follow_up_questions.items():
    follow_up_responses[agent_name] = generate_answers(question, {agent_name: rags[agent_name]})[agent_name]

# Final Aggregation
all_responses = {
    agent_name: {
        "initial": initial_responses[agent_name]["answer"],
        "follow_up": follow_up_responses[agent_name]["answer"],
    }
    for agent_name in rags.keys()
}
final_answer = aggregate_final_response(aggregation_llm, all_responses, test_prompt)

# Organize and Display
conversation_json = organize_into_json(
    test_prompt, initial_responses, follow_up_questions, follow_up_responses, final_answer
)
display_from_json(conversation_json)

json.dump(conversation_json, open(JSON_PATH, "w"), indent=4)