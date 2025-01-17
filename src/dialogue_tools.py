import json, os
from datetime import datetime
import json
import os
import textwrap
from src.vectorstore import VectorstoreHandler
from src.models import init_emb, init_llm
from src.rag_tools import build_rag_chain

def create_rag(sources_dir,config, handler):
    subfolder = config["subfolder"]
    llm_name = config["llm_name"]
    emb_name = config["emb_name"]
    k = config["k"]

    dir_path = os.path.join(sources_dir, subfolder)
    emb = init_emb(emb_name)
    llm = init_llm(llm_name)
    vs = handler.build_vectorstore(dir_path, emb, emb_name)
    retriever = handler._init_retriever(vs, dir_path, k)
    chain = build_rag_chain(retriever, llm)
    return chain, retriever


def generate_answers(prompt, rags):
    rag_responses = {}
    #print(f"\nQuerying all RAGs for prompt: '{prompt}'\n{'=' * 80}")
    for rag_name, (chain, _) in rags.items():
       # print(f"--- Querying RAG: {rag_name} ---")
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

def cross_reference(rags, follow_up_questions, follow_up_responses, original_prompt):
    cross_agent_responses = {}

    agent_names = list(rags.keys())
    for i, target_agent in enumerate(agent_names):
        verifier_agent = agent_names[(i + 1) % len(agent_names)]

        follow_up_question = follow_up_questions.get(target_agent, "")
        follow_up_response = follow_up_responses.get(target_agent, {}).get("answer", "")
        docs = follow_up_responses.get(target_agent, {}).get("docs", [])

        prompt = (
            f"The user originally asked:\n'{original_prompt}'\n\n"
            f"The follow-up question for the agent '{target_agent}' was:\n"
            f"'{follow_up_question}'\n\n"
            f"The agent provided the following follow-up response:\n"
            f"{follow_up_response}\n\n"
            "Your task is to critically evaluate this response in light of the original question and the follow-up question. Specifically:\n"
            "1. Identify any inaccuracies, gaps, or ambiguities in the response. If errors are found, correct them and provide a clear explanation.\n"
            "2. If the response is accurate, offer additional nuance, context, or clarity to enhance its completeness and value.\n\n"
            "Focus on ensuring the response is both correct and comprehensive while addressing the user's original and follow-up questions. Provide a concise yet thorough critique."
        )

        verifier_chain = rags[verifier_agent][0]
        output = verifier_chain.invoke(prompt)

        cross_agent_responses.setdefault(target_agent, {})[verifier_agent] = {
            "answer": output.get("answer", "No answer"),
            "docs": output.get("docs", [])
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


def process_question(question, rags, aggregation_llm, log_folder):
    # 1) Ask each RAG for initial answer
    initial_responses = generate_answers(question, rags)
    # 2) Aggregation LLM -> follow-up question for each RAG
    agent_answers = {rag_name: data["answer"] for rag_name, data in initial_responses.items()}
    follow_up_questions = generate_follow_up_questions(aggregation_llm, question, agent_answers)
    # 3) Each RAG answers its follow-up question
    follow_up_responses = {}
    for agent_name, fq in follow_up_questions.items():
        ans = generate_answers(fq, {agent_name: rags[agent_name]})
        follow_up_responses[agent_name] = ans[agent_name]
    # 4) Cross-reference
    cross_agent_responses = cross_reference(rags, follow_up_questions, follow_up_responses, question)
    # 5) Final aggregated answer
    all_resps = {}
    for agent_name in rags.keys():
        all_resps[agent_name] = {
            "initial": initial_responses[agent_name]["answer"],
            "follow_up": follow_up_responses[agent_name]["answer"] if agent_name in follow_up_responses else "",
        }
    final_answer = aggregate_final_response(aggregation_llm, all_resps, question)
    # 6) Organize results & save
    conversation_json = organize_into_json(
        question,
        initial_responses,
        follow_up_questions,
        follow_up_responses,
        final_answer
    )
    conversation_json["cross_agent_responses"] = {
        target_rag: {
            verifier: resp.get("answer","No answer")
            for verifier, resp in cross_agent_responses[target_rag].items()
        }
        for target_rag in cross_agent_responses
    }
    filename = f"{log_folder}/multi_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json.dump(conversation_json, open(filename,"w"), indent=2)
    return final_answer

def run_all_questions(questions, rags, aggregation_llm, log_folder):
    qa = {}
    for q in questions:
        qa[q] = process_question(q, rags, aggregation_llm, log_folder)
    # Optionally, produce a single JSON with all Q&A
    result_json = {"questions_and_answers": qa}
    combined_filename = f"{log_folder}/all_questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json.dump(result_json, open(combined_filename,"w"), indent=2)
    return qa