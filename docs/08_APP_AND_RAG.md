# Application and RAG

python demo.py launches the existing Streamlit dashboard on loopback. Use verified TPG CSV upload and select the available RPT. The app displays measured curves, same-RPT SoH estimation when required modalities and a verified historical checkpoint are available, reference error when a valid label exists, and Charge-based PyDMA diagnoses when supported.

The integrated RAG component combines dense and lexical retrieval and cites retrieved material. It requires the index and configured local embedding and generation endpoints by default. Diagnosis and RAG are independent of the neural model. There is no promoted updated-Physics checkpoint.

See the [dashboard](../app/final2_streamlit_dashboard.py) and [RAG service](../rag/rag_service.py).
