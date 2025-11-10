import os, re, json, time, wikipediaapi
from tqdm import tqdm


user_agent = 'EducationalScraper/1.0 (devan@cs.utoronto.ca)'

urls = []

def get_page_summary(page_title: str) -> str:
    """
    Fetch the summary of a Wikipedia page.
    - page_title: title of the Wikipedia page
    Returns:
        summary: summary text of the page, or an error message if the page does not exist
    """
    wiki = wikipediaapi.Wikipedia(
        user_agent=user_agent,
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )

    page = wiki.page(page_title)
    if page.exists():
        return page.summary
    else:
        return "Page not found."

def get_titles(html_content: str) -> list[str]:
    """
    This function is a result of my inability to work with BeautifulSoup. It's so annoying.
    So instead of something intelligent, we are going to grab the entire html from a url 
    as text. Then we are going to filter it to only in between left and right markets (excluding both from the string)
    Then we will grab all text present in between <li> and </li> tags.

    We will return a list of all such link text snippets. They happen to be the titles I need.
    """
    # Now parse the relevant_html to find all <li>...</li> contents using regex
    titles = re.findall(r'<li[^>]*>.*?<a[^>]*>(.*?)</a>.*?</li>', html_content, re.DOTALL | re.IGNORECASE)
    titles = [title.strip() for title in titles]

    return titles

def get_all_titles(html_dir: str = '/Users/mrmackamoo/Projects/topological_analysis/manifold_evolution/data/htmls') -> list[str]:
    """
    Go through each .txt file, and read in the html content.
    Then call get_titles on each, and aggregate all titles into a single list.
    Returns:
        all_titles: list of all titles found across all html files
    """
    all_titles = []
    for filename in os.listdir(html_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(html_dir, filename), 'r', encoding='utf-8') as f:
                html_content = f.read()
                titles = get_titles(html_content)
                all_titles.extend(titles)
    return all_titles

def process_titles_to_json(json_file: str, sleep: float = 1.0):
    """
    Process titles from get_all_titles(), fetch summaries for new ones, and save to JSON incrementally.
    Arguments:
    - json_file: path to the JSON file to save summaries
    - sleep: time to wait between requests to avoid overwhelming Wikipedia (in seconds)
    """
    all_titles = get_all_titles()
    
    # Load existing data if file exists
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {}
    
    # Get titles that haven't been processed yet
    processed_titles = set(data.keys())
    new_titles = [title for title in all_titles if title not in processed_titles]
    
    # Process new titles with progress bar, saving incrementally
    for title in tqdm(new_titles, desc="Fetching summaries"):
        summary = get_page_summary(title)
        data[title] = summary
        # Save after each title to avoid losing progress
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        time.sleep(sleep)  # Be polite and avoid hitting Wikipedia too hard

# Example usage
if __name__ == "__main__":
    process_titles_to_json('/Users/mrmackamoo/Projects/topological_analysis/manifold_evolution/data/initial_summaries.json')