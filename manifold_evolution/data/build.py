import json, openai, dotenv, os, tqdm, datetime

ROOT_DIR = '/Users/mrmackamoo/Projects/topological_analysis/'
DATA_DIR = os.path.join(ROOT_DIR, 'data')
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

