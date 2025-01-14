1. **Crete venv and install requirements.txt**  
   requirements.txt should be complete
2. **organize data**  
   - Create a directory (e.g., `sources`) with subfolders for each dataset (e.g., `lectures`, `articles`, `notes`).
   - Place your PDF, PPTX, and text files in the appropriate subfolders.
3. **access**
 For now, only the notebook has been set up and two rags have been constructed in parallel. 
 4. **Dialogue Notebook Update**

### **Overview**
The dialogue_notebook introduces a new workflow for using multiple (2 for now) Retrieval-Augmented Generation (RAG) agents to collaboratively refine and validate answers. This setup enhances the accuracy, depth, and reliability of the responses by leveraging cross-agent interactions and aggregation. 

---

### **Workflow Highlights**
1. **Initial Query:**  
   - Each RAG agent is asked the initial prompt, and their responses are captured.
   
2. **Follow-Up Questions:**  
   - An aggregation LLM generates follow-up questions for each RAG based on the initial prompt and their answers.  
   - Follow-up questions aim to clarify ambiguities, expand on details, and ensure alignment with the original query.  

3. **Follow-Up Responses:**  
   - Each RAG answers the follow-up question generated specifically for them.  

4. **Cross-Agent Verification:**  
   - RAG agents review and critique each other's follow-up responses.  
   - Agents identify inaccuracies, suggest corrections, and add nuanced insights to improve the overall response quality.  

5. **Final Aggregation:**  
   - An aggregation LLM synthesizes all initial responses, follow-up answers, and cross-verification insights into a single, concise, and structured final response.
