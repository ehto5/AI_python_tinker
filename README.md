-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
AI_python_tinker - Personal AI/ML experimentation projects, primarily focused on building a local vector search system over large text corpora using open source tooling.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**import_wiki_xml.py**
Processes a full Wikipedia XML dump and loads it into a Milvus vector database for semantic search. Streams the XML file one article at a time to keep memory usage manageable, cleans Wikipedia markup, splits articles into token-limited chunks by section, generates vector embeddings using the mxbai-embed-large-v1 sentence transformer model on Intel XPU hardware, and batch inserts the results into Milvus with metadata including article title, section, quality rating, and source URL. Filters out stubs, disambiguation pages, redirects, and low-value articles automatically.

**import_wikipediaapi.py**
Earlier prototype version that fetches individual Wikipedia articles via the Wikipedia API rather than processing a bulk XML dump. Used for initial development and testing of the chunking and embedding pipeline before scaling to full dataset processing.
