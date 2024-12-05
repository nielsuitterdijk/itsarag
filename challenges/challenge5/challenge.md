# GraphRAG

https://github.com/microsoft/graphrag

There's a GraphRAG repo from msft. 
Though it has many stars, the API seems not documented. 
Few examples are given, most of which use CLI. 
The accellerator uses Azure APIM and AKS and is designed for scale, something we are not interested in for this hackathon.

## GraphRAG in a nutshell

Extract all entities, relationships, and key claims from the TextUnits using an LLM.
Perform a hierarchical clustering of the graph using the Leiden technique. 
Generate summaries of each community and its constituents from the bottom-up. 
This aids in holistic understanding of the dataset.

At query time, these structures are used to provide materials for the LLM context window when answering a question. The primary query modes are:
- Global Search for reasoning about holistic questions about the corpus by leveraging the community summaries.
- Local Search for reasoning about specific entities by fanning-out to their neighbors and associated concepts.
- DRIFT Search for reasoning about specific entities by fanning-out to their neighbors and associated concepts, but with the added context of community information.

### Global Search

https://microsoft.github.io/graphrag/examples_notebooks/global_search/git 

Given a user query and, optionally, the conversation history, the global search method uses a collection of LLM-generated community reports from a specified level of the graph's community hierarchy as context data to generate response in a map-reduce manner. At the map step, community reports are segmented into text chunks of pre-defined size. Each text chunk is then used to produce an intermediate response containing a list of point, each of which is accompanied by a numerical rating indicating the importance of the point. At the reduce step, a filtered set of the most important points from the intermediate responses are aggregated and used as the context to generate the final response.

The quality of the global search’s response can be heavily influenced by the level of the community hierarchy chosen for sourcing community reports. Lower hierarchy levels, with their detailed reports, tend to yield more thorough responses, but may also increase the time and LLM resources needed to generate the final response due to the volume of reports.

### Local Search

Given a user query and, optionally, the conversation history, the local search method identifies a set of entities from the knowledge graph that are semantically-related to the user input. These entities serve as access points into the knowledge graph, enabling the extraction of further relevant details such as connected entities, relationships, entity covariates, and community reports. Additionally, it also extracts relevant text chunks from the raw input documents that are associated with the identified entities. These candidate data sources are then prioritized and filtered to fit within a single context window of pre-defined size, which is used to generate a response to the user query.
