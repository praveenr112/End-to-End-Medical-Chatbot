# End-to-End-Medical-Chatbot
End to End Medical Chatbot
Clone the repository

``` bash
https://github.com/praveenr112/End-to-End-Medical-Chatbot
```

### Step 01- Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.10 -y
```

``` bash
conda activate medibot
```

### Step 02- Install the requirements

``` bash
pip install -r requirements.txt
```

Create a .env file in the root directory and add your Pinecone & Openai credentials as follows:

``` bash
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "zzzzzzzzzzzzzzzzzzzzzzzzz"
```

```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
#Finally run the following command
python app.py
```