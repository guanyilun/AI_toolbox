"""A first attempt at a literature review tool."""
import os
import re
import logging
import requests
import json
import time
from openai import OpenAI

logging.basicConfig(level=logging.INFO)

class LM:
    def __init__(self, model):
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.messages = []
        self.model = model

    def get_response(self, prompt, system_message=None, max_tokens=10000, add_to_history=False): 
        if system_message is None:
            system_message = (
                "You are a helpful assistant that answer questions and provide guidance."
            )
        if len(self.messages) == 0:
            self.messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]
        else:
            self.messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )
        response_content = response.choices[0].message.content
        if add_to_history:
            self.messages.append({"role": "assistant", "content": response_content})
        return response_content

    def clear_history(self, n=None):
        # clear the history of messages
        # if n is not None, clear the last n messages
        # if n is None, clear all messages
        if n is not None:
            self.messages = self.messages[:-n]
        else:
            self.messages = []

def get_query_suggestions(lm, question) -> list[str]:
    prompt = f"""
    You are a helpful research assistant that provides advice on literature review.
    Your task is to suggest search queries for a given research question. The question
    will be provided within <question> </question> tags.
    Please provide a list of search queries that can be used to find the most relevant
    papers related to this question. Provide your response in a structured format with each
    query suggestion wrapped within <query> </query> tag.

    Example:
    <query>cosmic microwave background lensing</query>
    <query>cmb polarization</query>

    Here is the research question:
    <question>{question}</question>

    Suggestions:
    """
    res = lm.get_response(prompt)
    # parse the response
    queries = re.findall(r'<query>(.*?)</query>', res)
    if len(queries) == 0:
        logging.debug(f"Response: {res}")
        raise ValueError("No query suggestions found.")
    return queries

def search_ads(query, num_results=50):
    """
    Search the ADS API for papers matching the given query.
    
    :param query: The search query string
    :param num_results: Number of results to return (default 20)
    :return: List of dictionaries containing paper information
    """
    ads_api_token = os.getenv("ADS_API_TOKEN")
    if not ads_api_token:
        raise ValueError("ADS API token not found. Please set the ADS_API_TOKEN environment variable.")

    headers = {
        "Authorization": f"Bearer {ads_api_token}",
        "Content-Type": "application/json",
    }

    params = {
        "q": query,
        "fl": "bibcode,title,abstract",
        "rows": num_results,
    }

    response = requests.get("https://api.adsabs.harvard.edu/v1/search/query", headers=headers, params=params)
    response.raise_for_status()

    results = response.json()["response"]["docs"]
    papers = []
    for paper in results:
        papers.append({
            "bibcode": paper["bibcode"],
            "title": paper.get("title", [""])[0],
            "abstract": paper.get("abstract", "")
        })

    return papers

def _get_title_relevance(lm, question, papers):
    prompt = f"""
    Here are the titles of the papers found in the search results. Please provide
    feedback on the relevance of each title to the research question provided
    within <question></question> tag. You can provide feedback in the following format
    for each paper in the list:

    <paper><bibcode>bibcode</bibcode><feedback>feedback</feedback></paper>

    where feedback can be one of the following:
    - Relevant
    - Somewhat relevant
    - Irrelevant
    - Unsure 

    Research Question:
    <question>{question}</question>
    
    Here is the list of titles from the search results:
    """
    prompt += "\n".join(
        [f"<paper><bibcode>{paper['bibcode']}</bibcode><title>{paper['title']}</title></paper>" for paper in papers]
    )
    prompt += "\nProvide your feedbacks:\n"
    prompt += """
    Example response:
    <paper><bibcode>2021A&A...647A..10P</bibcode><feedback>Relevant</feedback></paper>
    <paper><bibcode>2021A&A...647A..11P</bibcode><feedback>Irrelevant</feedback></paper>

    Your response:
    """

    res = lm.get_response(prompt)
    rset = re.findall(r'<paper>(.*?)</paper>', res)
    feedbacks = {}
    for r in rset:
        bibcode = re.search(r'<bibcode>(.*?)</bibcode>', r).group(1)
        feedback_label = re.search(r'<feedback>(.*?)</feedback>', r).group(1)
        try:
            feedback_score = {
                'Relevant': 2,
                'Somewhat relevant': 1,
                'Unsure': 1,
                'Irrelevant': 0,
            }[feedback_label]
        except KeyError:
            logging.warning(f"Invalid feedback label: {feedback_label}")
            feedback_score = 1
        feedbacks[bibcode] = feedback_score
    if len(feedbacks) == 0:
        logging.debug(f"Response: {res}")
        raise ValueError("No feedback parsed.")
    return feedbacks

def get_title_relevance(lm, question, papers, batch_size=10):
    batches = [papers[i:i+batch_size] for i in range(0, len(papers), batch_size)]
    feedbacks = {}
    for batch in batches:
        logging.info(f"\tProcessing batch of {len(batch)} papers...")
        _feedbacks_batch = _get_title_relevance(lm, question, batch)
        for bibcode, feedback in _feedbacks_batch.items():
            feedbacks[bibcode] = feedback
            logging.info(f"\t\t{bibcode}: {feedback}")
    return feedbacks

def _get_abstract_relevance(lm, question, papers):
    prompt = f"""
    Here are the abstracts of the papers found in the search results. Provided in the
    following format:

    <paper><bibcode>bibcode</bibcode><title>title</title><abstract>abstract</abstract></paper> 
    
    Please provide feedback on the relevance of each paper to the provided research question.
    The research question is provided within <question></question> tag. You can provide feedback
    in a structured format as follows:

    <paper><bibcode>bibcode</bibcode><feedback>feedback</feedback></paper>

    where feedback can be one of the following:
    - Highly relevant
    - Relevant
    - Somewhat relevant
    - Irrelevant
    - Unsure 

    Research Question:
    <question>{question}</question>
    
    Here is the list of papers from the search results:
    """
    prompt += "\n".join(
        [f"<paper><bibcode>{paper['bibcode']}</bibcode><title>{paper['title']}></title><abstract>{paper['abstract']}</abstract></paper>" for paper in papers]
    )
    prompt += "\nProvide your feedbacks:\n"
    prompt += """
    Example response:
    <paper><bibcode>2021A&A...647A..10P</bibcode><feedback>Relevant</feedback></paper>
    <paper><bibcode>2021A&A...647A..11P</bibcode><feedback>Irrelevant</feedback></paper>

    Your response:
    """
    res = lm.get_response(prompt)
    rset = re.findall(r'<paper>(.*?)</paper>', res)
    feedbacks = {}
    for r in rset:
        bibcode = re.search(r'<bibcode>(.*?)</bibcode>', r).group(1)
        feedback_label = re.search(r'<feedback>(.*?)</feedback>', r).group(1)
        try:
            feedback_score = {
                'Highly relevant': 3,
                'Relevant': 2,
                'Somewhat relevant': 1,
                'Unsure': 1,
                'Irrelevant': 0,
            }[feedback_label]
        except KeyError:
            logging.warning(f"Invalid feedback label: {feedback_label}")
            feedback_score = 1
        feedbacks[bibcode] = feedback_score
    if len(feedbacks) == 0:
        logging.debug(f"Response: {res}")
        raise ValueError("No feedback parsed.")
    return feedbacks

def get_abstract_relevance(lm, question, papers, batch_size=5):
    # split the papers into batches
    batches = [papers[i:i+batch_size] for i in range(0, len(papers), batch_size)]
    feedbacks = {}
    for batch in batches:
        logging.info(f"\tProcessing batch of {len(batch)} papers...")
        _feedbacks_batch = _get_abstract_relevance(lm, question, batch)
        for bibcode, feedback in _feedbacks_batch.items():
            feedbacks[bibcode] = feedback
            logging.info(f"\t\t{bibcode}: {feedback}")
    return feedbacks
    

if __name__ == '__main__':
    import dotenv
    dotenv.load_dotenv()
    
    # import argparse
    # parser = argparse.ArgumentParser(description='A literature review tool.')
    # args = parser.parse_args()
    # temporary test
    class args:
        question = "The effect of cosmic birefringence on the CMB polarization measurement."
        
    # lm = LM("openai/gpt-4o-mini")
    # model class: S > A > B > C > ...
    lm = {
        'S': LM("openai/gpt-4o"),
        'A': LM("openai/gpt-4o-mini"),
        'B': LM("nousresearch/hermes-3-llama-3.1-405b:free"), # 8k context, 4k output
        'C': LM("liquid/lfm-40b"),                            # 32k context, 32k output
    }

    # get some search suggestions
    # queries = get_query_suggestions(lm['A'], args.question)

    # temp
    # queries = [
    #     "cosmic birefringence CMB polarization",
    #     "impact of birefringence on cosmic microwave background",
    #     "CMB polarization measurement cosmic birefringence",
    #     "birefringence effects on cosmic microwave background polarization",
    #     "cosmic birefringence and its influence on CMB",
    #     "CMB polarization anisotropy and birefringence",
    #     "theoretical models of cosmic birefringence CMB",
    #     "observational constraints on cosmic birefringence CMB polarization",
    #     "polarization rotation due to cosmic birefringence",
    #     "effects of isotropic cosmic birefringence on CMB measurements"
    # ]

    # send each query to ads search api and get the top 50 papers
    # all_papers = {}
    # query_stats = {}
    # load intermediate data
    if not os.path.exists('papers.json'):
        all_papers = {}
    else:
        with open('papers.json', 'r') as f:
            all_papers = json.load(f)
    if not os.path.exists('query_stats.json'):
        query_stats = {}
    else:
        with open('query_stats.json', 'r') as f:
            query_stats = json.load(f)
    
    if not os.path.exists('queries.json'):
        queries = get_query_suggestions(lm['A'], args.question)
    else:
        with open('queries.json', 'r') as f:
            queries = json.load(f)

    while len(queries) > 0:
        query = queries.pop(0)
        logging.info(f"Processing query: {query}")

        if query not in query_stats:
            query_stats[query] = {}
        else:
            logging.warning(f"Query {query} already processed. Skipping...")
            continue

        try:
            candidates = search_ads(query, num_results=50)
        except Exception as e:
            logging.error(f"Error processing query: {query}")
            logging.error(e)
            query_stats[query]['error'] = 'bad ads response'
            continue

        logging.info(f"\tRetrieved {len(candidates)} papers for query: {query}")

        if len(candidates) == 0:
            logging.warning(f"No papers found for query: {query}")
            continue

        # filter out papers already in the list
        candidates = [p for p in candidates if p['bibcode'] not in all_papers]
        logging.info(f"\tRemoving duplicates, left with {len(candidates)} new papers")

        if len(candidates) == 0:
            logging.warning(f"No new papers found for query: {query}")
            query_stats[query]['no_new_papers'] = True
            continue

        # assess title relevance
        logging.info("\tAssessing title relevance for the papers...")
        try:
            title_relevances = get_title_relevance(lm['B'], args.question, candidates, batch_size=20)
        except Exception as e:
            logging.error("Error assessing title relevance.")
            logging.error(e)
            query_stats[query]['error'] = 'bad title relevance assessment'
            continue

        if len(title_relevances) == 0:
            logging.warning("No title relevance feedback received.")
            query_stats[query]['error'] = 'empty title relevance feedback'
            continue

        # update query stats
        n_relevant = len([p for p in title_relevances if title_relevances[p]>=2])
        n_irrelevant = len([p for p in title_relevances if title_relevances[p]==0])
        query_stats[query]['relevant_rate'] = n_relevant / len(title_relevances)
        query_stats[query]['irrelevant_rate'] = n_irrelevant / len(title_relevances)

        # filter out irrelevant papers
        candidates = [p for p in candidates if p['bibcode'] in title_relevances and title_relevances[p['bibcode']] >= 2]
        logging.info(f"\tKeeping only relevant papers, left with {len(candidates)} papers")

        if len(candidates) == 0:
            logging.warning(f"No relevant titles found for query: {query}")
            query_stats[query]['no_relevant_title'] = True

        # assess abstract relevance
        logging.info("\tAssessing abstract relevance for the papers...")
        try:
            abstract_relevances = get_abstract_relevance(lm['B'], args.question, candidates, batch_size=5)
        except Exception as e:
            logging.error("Error assessing abstract relevance.")
            logging.error(e)
            query_stats[query]['error'] = 'bad abstract relevance assessment'
            continue

        if len(abstract_relevances) == 0:
            logging.warning("No abstract relevance feedback received.")
            query_stats[query]['error'] = 'empty abstract relevance feedback'
            continue

        n_relevant = len([p for p in abstract_relevances if abstract_relevances[p]>=2])
        n_irrelevant = len([p for p in abstract_relevances if abstract_relevances[p]==0])
        logging.info(f"\tFound {n_relevant} relevant and {n_irrelevant} irrelevant papers.")

        # filter out irrelevant papers
        candidates = [p for p in candidates if p['bibcode'] in abstract_relevances and abstract_relevances[p['bibcode']] >= 2]
        logging.info(f"\tKeeping only relevant papers, left with {len(candidates)} papers")

        if len(candidates) == 0:
            logging.warning(f"No relevant papers found for query: {query}")
            query_stats[query]['no_relevant_abstract'] = True
            continue

        n_highly_relevant = len([p for p in abstract_relevances if abstract_relevances[p]==3])
        query_stats[query]['abstract_highly_relevant_rate'] = n_highly_relevant / len(abstract_relevances)

        # find papers cited by the highly relevant papers
        highly_relevant_papers = [p for p in candidates if p['bibcode'] in abstract_relevances and abstract_relevances[p['bibcode']] == 3]
        _citations_queries = [
            'citations(bibcode:{})'.format(p['bibcode']) for p in highly_relevant_papers
        ]
        _references_queries = [
            'references(bibcode:{})'.format(p['bibcode']) for p in highly_relevant_papers
        ]
        new_queries = _citations_queries + _references_queries
        logging.info(f"\tFound {len(highly_relevant_papers)} highly relevant papers. Adding {len(new_queries)} new queries.")
        queries.extend(new_queries)

        # add to all_papers
        for paper in candidates:
            all_papers[paper['bibcode']] = paper

        logging.info(f"\tAdded {len(candidates)} new papers to the list. Total papers: {len(all_papers)}")

        # save data
        logging.info("\tSaving intermediate data...")
        with open('papers.json', 'w') as f:
            json.dump(all_papers, f, indent=2)
        with open('query_stats.json', 'w') as f:
            json.dump(query_stats, f, indent=2) 
        with open('queries.json', 'w') as f:
            json.dump(queries, f, indent=2)
        
        time.sleep(5)

    logging.info("All queries processed.")
    logging.info(f"Total papers found: {len(all_papers)}")