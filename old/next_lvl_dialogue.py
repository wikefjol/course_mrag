

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


def generate_follow_up_questions(aggregation_llm, original_prompt, agent_answers):
    """
    Generates follow-up questions for each agent based on the original prompt and their initial answers,
    including the original question in the output for clarity.
    
    Args:
        aggregation_llm: The aggregation LLM used to generate the follow-up questions.
        original_prompt (str): The original question posed to all agents.
        agent_answers (dict): A dictionary where keys are agent names and values are their initial answers.
    
    Returns:
        dict: A dictionary where keys are agent names and values are follow-up questions, 
              prefixed with the original question.
    """
    follow_up_questions = {}
    for agent_name, answer in agent_answers.items():
        prompt = (
        f"The user originally asked:\n"
        f"'{original_prompt}'\n\n"
        f"The agent '{agent_name}' provided the following response:\n"
        f"{answer}\n\n"
        "Your task is to generate a follow-up question to ask this agent. The follow-up question should:\n"
        "1. Clarify unclear points in the response.\n"
        "2. Expand on details relevant to the original question.\n"
        "3. Ensure alignment with the user's original query.\n\n"
        "Do not include questions directed at the user. Provide only the follow-up question in your response."
        )
        llm_response = aggregation_llm.invoke(prompt)
        follow_up_question = (
            f"The original question was:\n{original_prompt}\n\n"
            f"Here's the follow-up question:\n{llm_response['answer'] if isinstance(llm_response, dict) else llm_response}"
        )
        follow_up_questions[agent_name] = follow_up_question
    return follow_up_questions

# Cross-Agent Verification
def cross_agent_verification(rags, follow_up_responses, original_prompt,follow_up_quesionts):
    cross_agent_responses = {}
    for target_agent, follow_up_response in follow_up_responses.items():
        cross_agent_responses[target_agent] = {}
        for verifying_agent, (chain, _) in rags.items():
            if verifying_agent != target_agent:
                prompt = (
                f"The user originally asked:\n"
                f"'{original_prompt}'\n\n"
                f"The follow-up question for the agent '{target_agent}' was:\n"
                f"'{follow_up_questions[target_agent]}'\n\n"
                f"The agent provided the following follow-up response:\n"
                f"{follow_up_response['answer']}\n\n"
                "Your task is to critically evaluate this response in light of the original question and the follow-up question. Specifically:\n"
                "1. Identify any inaccuracies, gaps, or ambiguities in the response. If errors are found, correct them and provide a clear explanation.\n"
                "2. If the response is accurate, offer additional nuance, context, or clarity to enhance its completeness and value.\n\n"
                "Focus on ensuring the response is both correct and comprehensive while addressing the user's original and follow-up questions. Provide a concise yet thorough critique."
                )
                output = chain.invoke(prompt)
                cross_agent_responses[target_agent][verifying_agent] = {
                    "answer": output["answer"],
                    "docs": output["docs"],
                }
    return cross_agent_responses

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
JSON_PATH = "next_lvl_conversation.json"
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

original_prompt = "Give a point process model that is similar to the Ising model and explain what the similarities are"
#####################################################################################################################
#####################################################################################################################

# Main Execution Workflow
handler = VectorstoreHandler(SOURCES_DIR, force_rebuild=False)

# Initialize RAGs
rags = {rag_name: create_rag(config, handler) for rag_name, config in RAG_CONFIG.items()}

# Initialize Aggregation LLM
aggregation_llm = init_llm(AGGREGATION_LLM_NAME)

# Step 1: Initial Question to All Agents
initial_responses = generate_answers(original_prompt, rags)

# Step 2: Follow-Up Questions from Aggregation LLM
agent_answers = {k: v["answer"] for k, v in initial_responses.items()}
follow_up_questions = generate_follow_up_questions(aggregation_llm, original_prompt, agent_answers)

# Step 3: Follow-Up Answers from Original Agents
follow_up_responses = {
    agent_name: generate_answers(question, {agent_name: rags[agent_name]})[agent_name]
    for agent_name, question in follow_up_questions.items()
}

# Step 4: Cross-Agent Verification
cross_agent_responses = cross_agent_verification(rags, follow_up_responses, original_prompt, follow_up_questions)
print(cross_agent_responses)
# Step 5: Final Aggregation
all_responses = {
    agent_name: {
        "initial": initial_responses[agent_name]["answer"],
        "follow_up": follow_up_responses[agent_name]["answer"],
    }
    for agent_name in rags.keys()
}
final_answer = aggregate_final_response(
    aggregation_llm, all_responses, original_prompt
)

# Step 6: Organize and Display
conversation_json = organize_into_json(
    original_prompt, initial_responses, follow_up_questions, follow_up_responses, final_answer
)
conversation_json["cross_agent_responses"] = {
    target_agent: {
        verifier: response.get("answer", "No answer provided")
        for verifier, response in cross_agent_responses.get(target_agent, {}).items()
    }
    for target_agent in cross_agent_responses.keys()
}
display_from_json(conversation_json)

# Save JSON
json.dump(conversation_json, open(JSON_PATH, "w"), indent=4)