# AutoAgents
As part of the AI for Science initiative at Argonne, I built an agent-based LLM system to retrieve and generate deep science questions for building Q&A datasets. These questions will be used to help train the AuroraGPT project at ANL. 

# Work
The main focus of this project can be found in questionGenerationRAG, where I utilized an AutoGen agent system to generate deep science questions on academic research papers from the fields of Economics, Computer Science, Chemistry, Physics, Biology, and more. 
I also extended work done by Dr. Nicholas Chia to wrap the current Argonne LLM Argo for use in AutoGen and other agent frameworks. This can be found in Argo, CustomLLMAutogen, and CustomLLMAutogen2, with a wrapper test in argoWrapperTest.
