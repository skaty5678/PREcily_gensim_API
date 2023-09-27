# PREcily_gensim_API


This Flask API calculates the similarity between two text sentences using the Word2Vec model. It preprocesses the text, removes stop words, and computes the cosine similarity between the word embeddings of the sentences.


### Prerequisites

- Python 3.x
- Flask
- NLTK
- Gensim

You can install the required Python packages using the following command:

```bash
pip install flask nltk gensim
```

4. Run the Flask API using the following command:

   ```bash
   python app.py
   ```

   This will start the API locally on `http://localhost:8080`.

5. Use a tool like `curl` or `Postman` to send POST requests to the API with JSON data containing two text sentences for which you want to calculate similarity. Example:

   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"text1": "First sentence", "text2": "Second sentence"}' http://localhost:8080
   ```

6. The API will respond with a JSON object containing the similarity score between the two sentences.

### Endpoint

- **POST /**

   - Request:
   
     ```json
     {
       "text1": "First sentence",
       "text2": "Second sentence"
     }
     ```

   - Response:

     ```json
     {
       "similarity score": 0.85
     }
     ```
