import chromadb
from chromadb.utils import embedding_functions

class PersonaRouter:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./data")

        self.model_name = "all-MiniLM-L6-v2"
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.model_name
        )
        
        self.collection = self.client.get_or_create_collection(
            name="bot_personas", 
            embedding_function=self.emb_fn,
            metadata={"hnsw:space": "cosine"} 
        )
        
        self._seed_personas()

    def _seed_personas(self):
        """Phase 1.1: Store the three personas in the vector DB"""
        if self.collection.count() == 0:
            personas = [
                "Bot A (Tech Maximalist): I believe AI and crypto will solve all human problems. I am highly optimistic about technology, Elon Musk, and space exploration. I dismiss regulatory concerns.",
                "Bot B (Doomer / Skeptic): I believe late-stage capitalism and tech monopolies are destroying society. I am highly critical of AI, social media, and billionaires. I value privacy and nature.",
                "Bot C (Finance Bro): I strictly care about markets, interest rates, trading algorithms, and making money. I speak in finance jargon and view everything through the lens of ROI."
            ]
            ids = ["bot_a", "bot_b", "bot_c"]
            self.collection.add(documents=personas, ids=ids)
            print("✅ Personas seeded into the local database.")

    def route_post_to_bots(self, post_content: str, threshold: float = 0.85):
        """Phase 1.2: Cosine Similarity Matching"""
        results = self.collection.query(
            query_texts=[post_content],
            n_results=3,
            include=["distances", "documents"]
        )

        matched_bots = []
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            similarity = 1 - distance
            
            if similarity >= threshold:
                matched_bots.append({
                    "bot_id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "similarity": round(similarity, 4)
                })
        
        return matched_bots