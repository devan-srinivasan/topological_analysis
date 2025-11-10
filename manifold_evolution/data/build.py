import json, openai, dotenv, os, tqdm, datetime, torch
from transformers import BertTokenizer, BertModel

ROOT_DIR = '/Users/mrmackamoo/Projects/topological_analysis/'
DATA_DIR = os.path.join(ROOT_DIR, 'manifold_evolution/data')
assert dotenv.load_dotenv(os.path.join(ROOT_DIR, '.env')), "Failed to load .env file"

client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("ORGANIZATION_ID"),
    project=os.getenv("PROJECT_ID")
)

with open(f'{DATA_DIR}/initial_summaries.json', 'r') as f:
    
    summaries = json.load(f)

min_word_len = 100
filtered_summaries = {title: summary for title, summary in summaries.items() if len(summary.split()) >= min_word_len}

def perturb(
        text: str,
        system_prompt: str,
        model_id: str,
    ) -> str:
    """
    This function paraphrases the text using a large language model, using OpenAI's API
    Arguments:
    - text: the text to paraphrase
    - prompt: the prompt to use for paraphrasing
    - model_id: the OpenAI model id to use
    Returns:
        paraphrased_text: the paraphrased text
    """
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

def build_dataset(
        summaries: dict[str, str],
        system_prompt_key: str,
        model_id: str,
        output_file: str
    ) -> dict[str, str]:
    """
    Build a dataset of paraphrased summaries using the provided LLM, incrementally saving progress.
    Arguments:
    - summaries: dictionary of {title: summary} pairs
    - system_prompt: the prompt to use for paraphrasing
    - model_id: the OpenAI model id to use
    - output_file: path to save/load the paraphrased summaries
    Returns:
        paraphrased_summaries: dictionary of {title: paraphrased_summary} pairs
    """
    with open(os.path.join(DATA_DIR, 'prompts.json'), 'r') as f:
        prompts = json.load(f)

    system_prompt = prompts[system_prompt_key]
    now = datetime.datetime.now()
    formatted_time = now.strftime("%m_%d_%H_%M")
    output_file = f"{DATA_DIR}/{system_prompt_key}_{formatted_time}"

    # Load existing paraphrased_summaries if file exists
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            paraphrased_summaries = json.load(f)
    else:
        paraphrased_summaries = {}
    
    # Process only titles not already paraphrased
    titles_to_process = [title for title in summaries if title not in paraphrased_summaries]
    
    for title in tqdm.tqdm(titles_to_process, desc="Paraphrasing summaries"):
        paraphrased_summary = perturb(
            text=summaries[title],
            system_prompt=system_prompt.format(title),
            model_id=model_id,
        )
        paraphrased_summaries[title] = paraphrased_summary
        # Save after each step
        with open(output_file, 'w') as f:
            json.dump(paraphrased_summaries, f, indent=4)
    
    return paraphrased_summaries

def vectorize_dataset(
    summaries: dict[str, str],
    output_dir: str = '/Users/mrmackamoo/Projects/topological_analysis/manifold_evolution/data/embeddings/initial',
):
    """
    We vectorize the dataset by passing it through BERT and extracting the hidden states.
    Arguments:
    - summaries: dictionary of {title: summary} pairs
    - output_dir: directory to save the embeddings and summaries
    
    We process each summary for each title:summary pair in the dataset. Then we pass it through BERT setting
    output_hidden_states=True. We extract the hidden states from the output, from each layer, and save the tensor 
    to embeddings/{title}.pt in output_dir.

    For each summary, we save the embeddings as a tensor of shape (L, d) where L is the number of layers and d is the hidden state dimension.
    We will also save the summaries as a dictionary in summaries.json for reference, within this directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    embeddings_dir = os.path.join(output_dir, 'embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

    for title, summary in tqdm.tqdm(summaries.items(), desc="Vectorizing summaries"):
        inputs = tokenizer(summary, return_tensors='pt', truncation=True, max_length=512)
        outputs = model(**inputs)
        hidden_states = torch.stack(outputs.hidden_states, dim=0).squeeze(1)  # Shape: (L, T, d)
        torch.save(hidden_states, os.path.join(embeddings_dir, f'{title}.pt'))

    with open(os.path.join(output_dir, 'summaries.json'), 'w') as f:
        json.dump(summaries, f, indent=4)


vectorize_dataset(filtered_summaries)
